from cv2 import data
import numpy as np
import cv2
from enum import Enum
from typing import Optional

from dataclasses import dataclass


# Rectangular region (based on 2 parallel line)
src = {"tl": [554, 480], "tr": [736, 480], "bl": [214, 720], "br": [1116, 720]}
dest = {"tl": [320, 0], "tr": [960, 0], "bl": [320, 720], "br": [960, 720]}


def rect_dict2nparray(rect: dict) -> np.ndarray:
    """Transform a rectangle/bounding-box dictionary to a numpy array
    :return : (top-left, top-right, bottom-right, bottom-left) as np array"""
    return np.float32([rect["tl"], rect["tr"], rect["br"], rect["bl"]])


def warp_f(transform_matrix, img, output_dim=None):
    """warping helper"""
    if output_dim is None:
        # keep the same dimensions
        output_dim = img.shape[:2][::-1]
    return cv2.warpPerspective(
        img, transform_matrix, output_dim, flags=cv2.INTER_NEAREST
    )


class Unit(Enum):
    METER = "m"
    CENTIMETER = "cm"


class PixelSize:
    """Pixel size (in meter by default) per dimension x / y
    (main usage: to represent the size of a pixel in the warped space real world)"""

    def __init__(self, y_pp: float, x_pp: float, unit: Unit = Unit.METER) -> None:
        """
        :param y_pp: y wise size
        :param x_pp: x sie size
        """
        self.unit = unit
        self.y_pp = y_pp
        self.x_pp = x_pp


class Warper:
    """Perspective transform. helper class"""

    def __init__(
        self, src: np.ndarray, dest: np.ndarray, pixel_size: Optional[PixelSize] = None
    ):
        """Initialize warping transformation.
        :param src: source (4 points )
        :param dest: destination (4 coordinates) numpy array
        """
        self._transform = cv2.getPerspectiveTransform(src, dest)
        self._inv_transform = cv2.getPerspectiveTransform(dest, src)
        self.pixel_size = pixel_size

    def warp(self, img, output_dim=None):
        """Warp the input image (src->dest)"""
        return warp_f(self._transform, img, output_dim=output_dim)

    def inv_warp(self, img, output_dim=None):
        """Inverse warp input image (dest->src)"""
        return warp_f(self._inv_transform, img, output_dim=output_dim)

    def transform(self, src_img_point):
        homogenous = np.array(np.concatenate((src_img_point, [1])))
        homogenous_tr = np.matmul(self._transform, homogenous)
        return (homogenous_tr / homogenous_tr[2])[:2]

    def inv_transform(self, dest_img_point):
        homogenous = np.array(np.concatenate((dest_img_point, [1])))
        homogenous_tr = np.matmul(self._inv_transform, homogenous)
        return (homogenous_tr / homogenous_tr[2])[:2]
