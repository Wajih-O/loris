import numpy as np
import cv2


def warp(img, src, dst, output_dimension=None):
    """ Compute and apply perpective transform """
    if output_dimension is None:
        # keep the same dimensions
        output_dimension = img.shape[:2][::-1]
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, output_dimension, flags=cv2.INTER_NEAREST)