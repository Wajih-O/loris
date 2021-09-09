""" Implement warping helpers and visualization utils"""

import numpy as np
import cv2
from matplotlib.patches import Circle

# TODO move to calibration !
def extreme_rect(corners, nx):
    sq_corners = corners.squeeze()
    first_line = sq_corners[:nx]  # extracts first line of corners (with nx)
    last_line = sq_corners[-nx:]  # extracts last line of corners
    return {
        "tl": first_line[0],
        "tr": first_line[-1],
        "bl": last_line[0],
        "br": last_line[-1],
    }


def rect_dict2nparray(rect: dict) -> np.ndarray:
    return np.float32([rect["tl"], rect["tr"], rect["br"], rect["bl"]])


def draw_polygone(pts, image, isClosed=True, color=(255, 0, 255), thickness=2):
    return cv2.polylines(image, [pts.astype(int)], isClosed, color, thickness)


def highlight_corners(ax, rect: dict, corner_color="red"):
    """Rectangle/bounding box visualization tool."""
    ax.add_patch(Circle(rect.get("tl"), 10, color="red"))
    ax.add_patch(Circle(rect.get("tr"), 10, color="red"))
    ax.add_patch(Circle(rect.get("bl"), 10, color="red"))
    ax.add_patch(Circle(rect.get("br"), 10, color="red"))


def rect_dict2nparray(rect: dict) -> np.ndarray:
    """Transform a rectanle/bouding-box dictionary to a numpy array
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


class Warper:
    """Perspective transform. helper class"""

    def __init__(self, src: np.ndarray, dest: np.ndarray):
        """Initialize warping transformation.
        :param src: source (4 points )
        :param dest: destination (4 coordinates) numpy array
        """
        self._transform = cv2.getPerspectiveTransform(src, dest)
        self._inv_transform = cv2.getPerspectiveTransform(dest, src)


    def warp(self, img, output_dim=None):
        """Warp the input image (src->dest)"""
        return warp_f(self._transform, img, output_dim=output_dim)

    def inv_warp(self, img, output_dim=None):
        """Inverse warp input image (dest->src)"""
        return warp_f(self._inv_transform, img, output_dim=output_dim)

    def transform(self, src_img_point):
        homogenous =  np.array(np.concatenate((src_img_point, [1])))
        homogenous_tr = np.matmul(self._transform, homogenous)
        return (homogenous_tr/homogenous_tr[2])[:2]

    def inv_transform(self, dest_img_point):
        homogenous =  np.array(np.concatenate((dest_img_point, [1])))
        homogenous_tr = np.matmul(self._inv_transform, homogenous)
        return (homogenous_tr/homogenous_tr[2])[:2]