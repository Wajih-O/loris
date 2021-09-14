""" Implement warping helpers and visualization utils"""

import numpy as np
import cv2
from matplotlib.patches import Circle
from dataclasses import dataclass


def draw_polygone(pts, image, isClosed=True, color=(255, 0, 255), thickness=2):
    return cv2.polylines(image, [pts.astype(int)], isClosed, color, thickness)


def draw_line(image, start_point, end_point, color=(255, 0, 0), thickness=2):
    return cv2.line(image, start_point, end_point, color, thickness)


def highlight_corners(ax, rect: dict, corner_color="red"):
    """Rectangle/bounding box visualization tool."""
    ax.add_patch(Circle(rect.get("tl"), 10, color="red"))
    ax.add_patch(Circle(rect.get("tr"), 10, color="red"))
    ax.add_patch(Circle(rect.get("bl"), 10, color="red"))
    ax.add_patch(Circle(rect.get("br"), 10, color="red"))


@dataclass
class Point:
    """Point class to represent a pixel cooordinate"""

    x: int
    y: int

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Quadrilateral:
    """Quadrilateral (arbitrary) box represented with (upper, bottom) X (left, right) corners/points"""

    tl: Point
    tr: Point
    br: Point
    bl: Point

    @property
    def as_array(self) -> np.ndarray:
        """represent the quadrilateral as a numpy array
        :return : (top-left, top-right, bottom-right, bottom-left) as np array"""
        return np.float32([tl.as_array, tr.as_array, br.as_array, bl.as_array])
