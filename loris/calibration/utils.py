from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2


def to_gray(image):
    """rgb to gray scale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extreme_rect(corners, nx):
    """return the (outer) rectable (inner corners of the outer chessboard cases)
    :param corner: corners as detected by cv2.findChessboardCorners
    :param nx: number of inner columns of the chessboard
    :return : rectangle/bounding-box as a dictionnary

    """
    sq_corners = corners.squeeze()
    first_line = sq_corners[:nx]  # extracts first line of corners (with nx)
    last_line = sq_corners[-nx:]  # extracts last line of corners
    return {
        "tl": first_line[0],
        "tr": first_line[-1],
        "bl": last_line[0],
        "br": last_line[-1],
    }


def detect_corners(image, nx=9, ny=6) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Detect corner on calibration image (Chessboard)
    returns the internal coordinates wrapping the cv2.findChessboardCorners and spatial image shape"""
    gray = to_gray(image)
    status, corners = cv2.findChessboardCorners(gray, (nx, ny), None)  # detect corners
    if status == True:
        return corners, gray.shape[::-1]
    return None, gray.shape[::-1]


def calibrate(image_paths: List[str], nx=9, ny=6) -> Dict:
    """Calibrate camera using Chessboard images (path_pattern)"""

    def update_points(
        object_points, image_points, corners: Optional[np.ndarray] = None
    ):
        if corners is not None:
            # Chessboard corners detection is successful
            object_points.append(
                np.concatenate(
                    (np.mgrid[0:nx, 0:ny].T.reshape(-1, 2), np.zeros((nx * ny, 1))),
                    axis=-1,
                ).astype("float32")
            )
            image_points.append(corners)

    object_points = []
    image_points = []
    image_shape = None

    if len(image_paths):
        corners, image_shape = detect_corners(cv2.imread(image_paths[0]), nx=nx, ny=ny)
        update_points(object_points, image_points, corners)
        for image_path in image_paths[1:]:
            corners, shape = detect_corners(cv2.imread(image_path), nx=nx, ny=ny)
            if shape != image_shape:
                raise Exception(
                    f"multiple image sizes detected! ({image_path}) {shape} <> {image_shape}"
                )
            update_points(object_points, image_points, corners)
    output_dict = {}  # calibration output dictionary
    if len(image_points):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            np.array(object_points), np.array(image_points), image_shape, None, None
        )
        output_dict.update(
            {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
        )
    return output_dict


def undistort(image, calibration_output) -> Tuple[bool, np.ndarray]:
    """Return undistorted image using calibration artifacts"""
    mtx = calibration_output.get("mtx", None)
    dist = calibration_output.get("dist", None)
    if all([x is not None for x in [mtx, dist]]):
        return True, cv2.undistort(image, mtx, dist, None, mtx)
    return False, image
