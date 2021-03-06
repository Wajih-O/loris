import abc
from collections import deque
from collections.abc import Callable
import logging
from typing import Optional


import numpy as np
import cv2

from loris.perspective import PixelSize, Warper
from loris.calibration.utils import undistort

# from loris.lane_detection.threshold import combined_threshold
from loris.lane_detection.threshold import threshold_pipeline
from loris.lane_detection.advanced import (
    search_around_poly,
    find_lane_pixels,
    search_around_poly,
    non_zero,
    poly_fit,
    apply_poly,
)

# # Rectangular region (based on 2 parallel line)
# src = {"tl": [554, 480], "tr": [736, 480], "bl": [214, 720], "br": [1116, 720]}
# dest = {"tl": [320, 0], "tr": [960, 0], "bl": [320, 720], "br": [960, 720]}

# self.warper.pixel_size.y_pp = 30 / 720  # meters per pixel in y dimension
# self.warper.pixel_size.x_pp = 3.7 / 700  # meters per pixel in x dimension


def gen_curvature_calculator(fit):
    """Generates a curve calculator from a left/right lane side
         (second order polynomial fit)
    :param fit: second order polynomial fit (left lane side)
    """

    def curvature(y_eval):
        """Calculation of R_curve (radius of curvature)"""
        curvature_radius = pow(1 + pow((2 * fit[0] * y_eval + fit[1]), 2), 3 / 2) / (
            2 * np.abs(fit[0])
        )
        return curvature_radius

    return curvature


def convert(pixel_fit, x_meter_per_pixel, y_meter_per_pixel) -> np.array:
    """Convert line poly fit from pixel to meter."""
    b_factor = x_meter_per_pixel / y_meter_per_pixel
    a_factor = b_factor / y_meter_per_pixel
    return pixel_fit * np.array([a_factor, b_factor, x_meter_per_pixel])


class Line:
    """Define a class to receive the characteristics of each line detection"""

    def __init__(self, look_back: int = 2, pixel_size: Optional[PixelSize]=None):

        self._logger = logging.getLogger(__name__)

        self.look_back = look_back
        self.detected = False  # detected/initilized

        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=look_back)

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  # deque(maxlen=look_back)

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in meter (calculator/callable)
        self.meter_curv_radius_func: Callable[[float], float] = None
        self.pixel_curv_radius_func: Callable[[float], float] = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype="float")

        self.pixels_x = None  # detected line pixels x coordinates
        self.pixels_y = None  # detected line pixels y coordinates

        # Pixel size
        self.pixel_size = pixel_size

    def update_curvature_calculator(self):
       
        self.pixel_curv_radius_func = gen_curvature_calculator(self.best_fit)
        if self.pixel_size is not None:
            converted_fit = convert(self.best_fit, self.pixel_size.x_pp, self.pixel_size.y_pp)
            self.meter_curv_radius_func = gen_curvature_calculator(converted_fit)

    def __blend_fit(self, fit, max_y):
        """Integrate/accomodate fit as the current fit and update the line accordingly
        :param fit: the last fit to blend into the Line
        :param max_y: image y size
        """

        self.current_fit = fit
        self.detected = True
        y_range = np.arange(max_y)

        self.recent_xfitted.append(apply_poly(self.current_fit, np.arange(max_y)))
        self.bestx = np.mean(
            np.array(self.recent_xfitted), axis=0
        )  # update best x with the initialization
        self.best_fit = poly_fit(y_range, self.bestx)
        self.update_curvature_calculator()

    def initialize_detection(self, xs, ys, max_y):
        """Initialize line detection with pixels coordinate."""
        self.pixels_x = xs
        self.pixels_y = ys
        try:
            self.__blend_fit(poly_fit(self.pixels_y, self.pixels_x), max_y)
        except Exception as exception:
            self._logger.error(f"initialize detection: {exception}")

    def update_detection(self, nonzerox, nonzeroy, max_y, margin=100):
        """Given an initial second order polynomial fit update/adjust it with the current image"""
        if not self.detected:
            raise Exception("should be first initialized !")

        y_range = np.arange(max_y)
        xs, ys, _ = search_around_poly(nonzerox, nonzeroy, self.best_fit, margin=margin)
        # todo a weighted version (to weight the current vs the best fit) now equaly weighted
        try:
            self.__blend_fit(
                poly_fit(
                    np.concatenate((ys, y_range)),
                    np.concatenate((xs, apply_poly(self.best_fit, y_range))),
                ),
                max_y,
            )

            self.pixels_x = xs
            self.pixels_y = ys
        except Exception as exception:
            self._logger.error(f"update detection: {exception}")
            self.detected = False
            raise exception

    def pixel_curvature(self, y_eval) -> float:
        """R_curve (radius of curvature)"""
        return self.pixel_curv_radius(y_eval)

    def get_x(self, y_eval):
        if self.best_fit is not None:
            return apply_poly(self.best_fit, y_eval)


class LaneDetector:
    def __init__(
        self, calibration_params, warper: Warper, margin: int = 100, look_back: int = 10
    ):
        """LaneDetector (left, right) side of the lane"""

        self.processed_images = 0
        self.left_fitx = None  # second order polynomial fit
        self.right_fitx = None  # second order polynomial fit
        self.search_around_poly_counter = 0

        # Calibration
        self.calibration_params = calibration_params
        self.line_search_margin = margin

        # bird-eye view perspective transform
        self.warper = warper

        # left and right lane lines
        self.right = Line(look_back=look_back, pixel_size=self.warper.pixel_size)
        self.left = Line(look_back=look_back, pixel_size=self.warper.pixel_size)

        

    @staticmethod
    def line_pixel(ploty, poly_fit=None):
        """label line"""
        if poly_fit is not None:
            return poly_fit[0] * ploty ** 2 + poly_fit[1] * ploty + poly_fit[2]
        return None

    def label_detection(self, warped_binary, margin_pixels=5):
        """Label warped detection image then un-warp it (to be weight added to the original image)"""

        # Generate x and y values for plotting
        # TODO: Add edge thickness parameter for lane labeling!

        ploty = np.arange(warped_binary.shape[0])
        left_x = apply_poly(self.left.best_fit, ploty)
        right_x = apply_poly(self.right.best_fit, ploty)

        # Label warped image with (lane pixel annotation)
        out_img = np.dstack((warped_binary,) * 3)

        # plotting the margin detected line/white pixels
        out_img[self.left.pixels_y, self.left.pixels_x] = [
            255,
            0,
            0,
        ]  # left lane side in red

        out_img[self.right.pixels_y, self.right.pixels_x] = [
            0,
            0,
            255,
        ]  # right lane side in blue

        # plotting the fitted lines
        if left_x is not None:
            out_img[
                ploty.astype(int), np.clip(left_x, 0, out_img.shape[1] - 1).astype(int)
            ] = [
                0,
                255,
                255,
            ]  # colors left lane side in blue
        if right_x is not None:
            out_img[
                ploty.astype(int), np.clip(right_x, 0, out_img.shape[1] - 1).astype(int)
            ] = [
                0,
                255,
                255,
            ]  # colors left lane side in blue

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x + margin_pixels, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_x - margin_pixels, ploty])))]
        )
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        green_lane = np.zeros_like(out_img)
        cv2.fillPoly(green_lane, np.int_([pts]), (0, 255, 0))

        return self.warper.inv_warp(
            cv2.addWeighted(out_img, 0.5, green_lane, 0.5, 0, dtype=cv2.CV_32F).astype(
                np.uint8
            )
        )

    def label(self, orig, warped_binary, margin_pixels=5):
        """Label/annotate a road image with detected lane"""
        text_color = (0, 255, 0)
        text_thickness = 2
        max_y = warped_binary.shape[0] - 1

        left_curv_radius = self.left.meter_curv_radius_func(self.warper.pixel_size.y_pp * max_y)
        right_curv_radius = self.right.meter_curv_radius_func(self.warper.pixel_size.y_pp * max_y)

        # avg_curv_radius = (left_curv_radius + right_curv_radius) / 2
        out_img = cv2.addWeighted(
            orig,
            0.6,
            self.label_detection(warped_binary, margin_pixels=margin_pixels),
            0.4,
            0,
            dtype=cv2.CV_32F,
        ).astype(np.uint8)

        cv2.putText(
            out_img,
            f"Curv. radius left: {left_curv_radius:.2f}m, right: {right_curv_radius:.2f}m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            text_thickness,
        )
        # cv2.putText(
        #     out_img,
        #     f"Curv. radius: {avg_curv_radius:.2f}",
        #     (20, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     text_color,
        #     text_thickness,
        # )

        #  Car offset
        center = np.array([warped_binary.shape[1] // 2, max_y])  # image center
        w_center_x, w_center_y = self.warper.transform(center).astype(int).ravel()
        offset = (
            w_center_x
            - (self.left.get_x(w_center_y) + self.right.get_x(w_center_y)) / 2
        ) * self.warper.pixel_size.x_pp

        cv2.putText(
            out_img,
            f"Car offset: {offset:.2f}m",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            text_thickness,
        )
        return out_img

    def __init_lane(self, warped_binary):

        leftx, lefty, rightx, righty = find_lane_pixels(
            255 * warped_binary
        )  # initialize the lane sides
        self.left.initialize_detection(leftx, lefty, warped_binary.shape[0])
        self.right.initialize_detection(rightx, righty, warped_binary.shape[0])

    def process_image(self, image):
        # undistort/correct image
        status, output = undistort(image, self.calibration_params)
        if status:
            # preprocess image apply thresholdeing pipeline then warp
            warped_binary = self.warper.warp(threshold_pipeline(output))

            if not self.left.detected or not self.right.detected:  # initialize lines
                self.__init_lane(warped_binary)

            else:
                try:
                    # raise Exception("to remove")
                    nonzerox, nonzeroy = non_zero(warped_binary)
                    self.left.update_detection(
                        nonzerox,
                        nonzeroy,
                        warped_binary.shape[0],
                        margin=self.line_search_margin,
                    )
                    self.right.update_detection(
                        nonzerox,
                        nonzeroy,
                        warped_binary.shape[0],
                        margin=self.line_search_margin,
                    )

                    self.search_around_poly_counter += 1

                except Exception:
                    self.__init_lane(warped_binary)

            self.processed_images += 1

            return self.label(
                output, warped_binary
            )  # self.__label_detection(255*warped_binary)#

        return output
