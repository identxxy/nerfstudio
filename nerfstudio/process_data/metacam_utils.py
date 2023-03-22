"""Helper functions for processing metacam data."""

import json
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import io

CONSOLE = Console(width=120)


def inverse_transform(a2b):
    """Return the inverse transform `b2a`, given 4x4 R|t transform `a2b`"""
    assert a2b.shape == (4, 4)
    rot = a2b[0:3, 0:3]
    t = a2b[0:3, 3]
    b2a = np.eye(4)
    b2a[0:3, 0:3] = rot.T
    b2a[0:3, 3] = -rot.T @ t
    return b2a


def read_lidar_camera_calib(lidar_camera_calib: Path) -> np.ndarray:
    """Read lidar camera calibration file, return the transform matrix lidar to camera `l2c`."""
    lidar_camera_calib_np = np.loadtxt(lidar_camera_calib)
    l2c_rot = Rotation.from_euler("xyz", lidar_camera_calib_np[0:3]).as_matrix()
    l2c = np.eye(4)
    l2c[:3, :3] = l2c_rot
    l2c[:3, 3] = lidar_camera_calib_np[3:6]
    return l2c


def read_camera_intrisincs_from_json(file: Path) -> dict:
    """Read camera calibration files.

    Return dict keys format
    - front
        - intrinsics
        - x2f
    - left ...
    - right ...

    Args:
        file: path to camera_calibration.json

    Return:
        cameras: containing 3 cameras intrinsics and transform matrix from x to front camera.
    """
    with open(file) as f:
        prefixes = ["front", "left", "right"]
        cameras = {prefixes[0]: {}, prefixes[1]: {}, prefixes[2]: {}}
        data = json.load(f)["value0"]
        intrinsics = data["intrinsics"]
        extrinsics = data["T_imu_cam"]
        resolution = data["resolution"]
        for i in range(3):
            cameras[prefixes[i]]["intrinsics"] = {}
            cameras[prefixes[i]]["intrinsics"]["fl_x"] = intrinsics[i]["intrinsics"]["fx"]
            cameras[prefixes[i]]["intrinsics"]["fl_y"] = intrinsics[i]["intrinsics"]["fy"]
            cameras[prefixes[i]]["intrinsics"]["k1"] = intrinsics[i]["intrinsics"]["k1"]
            cameras[prefixes[i]]["intrinsics"]["k2"] = intrinsics[i]["intrinsics"]["k2"]
            cameras[prefixes[i]]["intrinsics"]["k3"] = intrinsics[i]["intrinsics"]["k3"]
            cameras[prefixes[i]]["intrinsics"]["k4"] = intrinsics[i]["intrinsics"]["k4"]
            cameras[prefixes[i]]["intrinsics"]["cx"] = intrinsics[i]["intrinsics"]["cx"]
            cameras[prefixes[i]]["intrinsics"]["cy"] = intrinsics[i]["intrinsics"]["cy"]
            cameras[prefixes[i]]["intrinsics"]["w"] = resolution[i][0]
            cameras[prefixes[i]]["intrinsics"]["h"] = resolution[i][1]
            x2f = np.eye(4)
            x2f[0:3, 0:3] = Rotation.from_quat(
                [extrinsics[i]["qx"], extrinsics[i]["qy"], extrinsics[i]["qz"], extrinsics[i]["qw"]]
            ).as_matrix()
            x2f[0, 3] = extrinsics[i]["px"]
            x2f[1, 3] = extrinsics[i]["py"]
            x2f[2, 3] = extrinsics[i]["pz"]
            cameras[prefixes[i]]["x2f"] = x2f
    return cameras


def read_images_and_odom(front: Path, odom: Path, c2l: np.ndarray) -> List:
    """Read front images and odometry file, return the poses of images.

    Return list element dict keys format
    - file_name
    - c2w

    Args:
        front: path to front images folder
        odom: path to odometry.csv file
        c2l: calibrated 4x4 matrix

    Return:
        frames: containing valid images name "xxx.jpg" and front camera c2w matrix.
    """
    data = pd.read_csv(odom)
    start_sec = data[".header.stamp.secs"][0]
    sec = np.array(data[".header.stamp.secs"]) - start_sec
    nsec = np.array(data[".header.stamp.nsecs"])
    timestamp = sec + nsec / 1e9
    px = np.array(data[".pose.pose.position.x"])
    py = np.array(data[".pose.pose.position.y"])
    pz = np.array(data[".pose.pose.position.z"])
    qx = np.array(data[".pose.pose.orientation.x"])
    qy = np.array(data[".pose.pose.orientation.y"])
    qz = np.array(data[".pose.pose.orientation.z"])
    qw = np.array(data[".pose.pose.orientation.w"])
    rots = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=1))
    # images idx
    frames = []
    camera_time_l = []
    for f in sorted(front.iterdir()):
        s = f.stem
        camera_time = int(s[0:-9]) - start_sec + int(s[-9:]) / 1e9
        if camera_time > timestamp[0] and camera_time < timestamp[-1]:
            frames.append({"file_name": f.name})
            camera_time_l.append(camera_time)
        else:
            pass
    camera_time_np = np.array(camera_time_l)

    def interpolation(x, y, xnew):
        kind = "quadratic"
        # 'slinear', 'quadratic' and ‘cubic' refer to a spline interpolation of first, second or third order
        f = interpolate.interp1d(x, y, kind=kind)
        ynew = f(xnew)
        return ynew

    slerp = Slerp(timestamp, rots)

    new_px = interpolation(timestamp, px, camera_time_np)
    new_py = interpolation(timestamp, py, camera_time_np)
    new_pz = interpolation(timestamp, pz, camera_time_np)
    new_rots = slerp(camera_time_np)

    for i, frame in enumerate(frames):
        l2w = np.eye(4)
        l2w[0:3, 0:3] = new_rots[i].as_matrix()
        l2w[0:3, 3] = np.array([new_px[i], new_py[i], new_pz[i]])

        frame["c2w"] = l2w @ c2l
    return frames


def undistort_image(src, dst, intrinsics, verbose=False):
    summary_log = []
    img = cv2.imread(str(src))
    K = np.array([[intrinsics["fl_x"], 0, intrinsics["cx"]], [0, intrinsics["fl_y"], intrinsics["cy"]], [0, 0, 1]])
    D = np.array([intrinsics["k1"], intrinsics["k2"], intrinsics["k3"], intrinsics["k4"]])
    u_img = cv2.fisheye.undistortImage(img, K, D, Knew=K, new_size=(4032, 3040))
    cv2.imwrite(str(dst), u_img)
    return summary_log


def delete_dir(path, verbose=False):
    summary_log = []
    if verbose:
        summary_log.append(f"Delete {path}")
    shutil.rmtree(path, ignore_errors=True)
    return summary_log
