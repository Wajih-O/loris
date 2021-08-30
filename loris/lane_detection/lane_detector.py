# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

import abc
from dataclasses import dataclass
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt

from loris.utils import warp
from loris.calibration.utils import calibrate, undistort

# from loris.lane_detection.threshold import combined_threshold
from loris.lane_detection.threshold import threshold_pipeline
from loris.lane_detection.utils import extract_lane
from loris.lane_detection.visu_utils import draw_lane
from loris.lane_detection.advanced import visualize, search_around_poly,find_lane_pixels, fit_poly, visualize, search_around_poly, non_zero, poly_fit, apply_polynomial



# rectangular region (based on 2 parallel line)
src = {'tl': [580 , 500], 'tr': [760, 500], 'bl': [260, 720], 'br': [1050, 720]}
dest =  {'tl': [320 , 0], 'tr': [960, 0], 'bl': [320, 720], 'br': [960, 720]}

def rect_dict2nparray(rect:dict) -> np.ndarray:
    return np.float32([rect["tl"], rect["tr"], rect["br"], rect["bl"]])


@dataclass
class LineDetection:
    fitx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    inds: np.ndarray

    
    def update(margin):
        """ update """
        return poly_fit(y, x) if np.sum(inds) else fitx

class Line():
    """ Define a class to receive the characteristics of each line detection """
    def __init__(self, look_back:int = 5):


        self.look_back = look_back
        self.detected = False # detected/initilized

        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=look_back)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = deque(maxlen=look_back)
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        self.pixels_x = None # detected pixels x coordinates
        self.pixels_y = None # y values for detected line pixels

    def initialize_detection(self, xs, ys):
        """ Initialize line detection with pixels coordinate """
        self.pixels_x = xs
        self.pixels_y = ys
        try:
            self.current_fit = poly_fit(self.pixels_x, self.pixels_y)
            self.detected = True
            self.best_fit.append(self.current_fit)
        except Exception as e:
            assert e
            pass # to log the error and continue !

    def update_detection(self, nonzerox, nonzeroy, max_y, margin=100):
        """ Given an initial second order polynomial fit update/adjust it with the current image """
        if not self.detected:
            raise Exception("should be first initialized !")
        
        self.pixels_x, self.pixels_y, _ = search_around_poly(nonzerox, nonzeroy, self.best_fit, margin=margin)

        ys = np.linspace(0, max_y-1, max_y)

        previous_fitted = apply_polynomial(self.current_fit, ys)

        self.left_fitx = poly_fit(np.concatenate((self.pixels_x, ys)), np.concatenate((self.pixels_y, previous_fitted)))
        

class LaneDetector:
    # todo separate processing for left/right sides
    def __init__(self, calibration_params, margin=100, look_back = 5, prior=None):
        """ """
        self.prior = prior
        self.processed_images = 0
        self.left_fitx = None # second order polynomial fit
        self.right_fitx = None # second order polynomial fit
        self.search_around_poly_counter = 0
        # Calibration
        self.calibration_params = calibration_params
        self.line_search_margin = margin
        self.binary_warped = None

        # left and right lane lines
        self.right = Line(look_back=look_back)
        self.left = Line(look_back=look_back)

    @staticmethod
    def line_pixel(ploty, poly_fit=None):
        """ label line """
        if poly_fit is not None:
            return poly_fit[0]*ploty**2 + poly_fit[1]*ploty + poly_fit[2]
        return None

    def __label_detection(self, detection_image, margin_pixels=5):
        """ Label warped detection image then un-warp it (to be weight added to the original image)"""

        # Generate x and y values for plotting
        # TODO: Add edge thickness parameter for lane labeling!

        ploty = np.linspace(0, detection_image.shape[0]-1, detection_image.shape[0])
        left_x = LaneDetector.line_pixel(ploty, self.left_fitx)
        right_x = LaneDetector.line_pixel(ploty, self.right_fitx)

        out_img = np.copy(detection_image) # np.dstack((detection_image,)*3)

        if left_x is not None:
            out_img[ploty.astype(int), np.clip(left_x, 0, 900).astype(int)] = [255, 0, 0] # colors left lane side in blue
        if right_x is not None:
            out_img[ploty.astype(int), np.clip(right_x, 0, out_img.shape[1]-1).astype(int)] = [0, 0, 255] # colors left lane side in blue


        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x + margin_pixels, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x-margin_pixels, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        green_lane = np.copy(out_img)
        cv2.fillPoly(green_lane, np.int_([pts]), (0,255, 0))

        out_img = cv2.addWeighted(out_img, .5, green_lane, .5, 0, dtype=cv2.CV_32F).astype(np.uint8)

        return warp(out_img, rect_dict2nparray(dest), rect_dict2nparray(src))

    def label(self, orig, detection_image, ax = plt):
        """ Label/annotate an image with """
        # return self.__label_detection(detection_image)
        return cv2.addWeighted(orig, 0.2, self.__label_detection(detection_image), 0.8, 0, dtype=cv2.CV_32F).astype(np.uint8)

    def process_image(self, image, ax=plt):
        status, output = undistort(image, self.calibration_params)
        if status:
            combined = threshold_pipeline(output)
            # color_combined = np.stack((combined,)*3, axis=-1)
            self.binary_warped = warp(combined, rect_dict2nparray(src), rect_dict2nparray(dest))
            if self.left_fitx is None or self.right_fitx is None: # initialize lines
                leftx, lefty, rightx, righty = find_lane_pixels(255*self.binary_warped) # initialize the lane sides
                
                # self.left.initialize_detection(leftx, lefty)
                # self.right.initialize_detection(rightx, righty)

                # left_fitx, right_fitx = fit_poly(leftx, lefty, rightx, righty)
                left_fitx = poly_fit(lefty, leftx)
                right_fitx = poly_fit(righty, rightx)
                
                if self.left_fitx is None:
                    self.left_fitx = left_fitx
                if self.right_fitx is None:
                    self.right_fitx = right_fitx
            else:
                nonzerox, nonzeroy = non_zero(self.binary_warped)

                # TODO REFACTOR here !!!!  a class that takes the nonzero and the fit !!!!then do all the needed to update the fit (or keeping it)
                leftx, lefty, left_inds = search_around_poly(nonzerox, nonzeroy, self.left_fitx, margin=self.line_search_margin)
                rightx, righty, right_inds = search_around_poly(nonzerox, nonzeroy, self.right_fitx, margin=self.line_search_margin)

                # label sides (white pixel search) left in blue and right in red
                out_img = np.dstack((self.binary_warped,)*3)
                out_img[lefty, leftx] = [255, 0, 0] # left lane side in blue
                out_img[righty, rightx] = [0, 0, 255] # right lane side in red


                # Add the previous line side detection as a prior

                ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
                previous_fitted_left = apply_polynomial(self.left_fitx, ploty)
                previous_fitted_right = apply_polynomial(self.right_fitx, ploty)
                self.left_fitx = poly_fit(np.concatenate((lefty, ploty)), np.concatenate((leftx, previous_fitted_left))) # if leftx.shape[0] else left_fit
                self.right_fitx = poly_fit(np.concatenate((righty, ploty)), np.concatenate((rightx, previous_fitted_right))) # if leftx.shape[0] else left_fit

                # # Update detection !
                # self.left.update_detection(nonzerox, nonzeroy, self.binary_warped.shape[0], margin=self.line_search_margin)
                # self.right.update_detection(nonzerox, nonzeroy, self.binary_warped.shape[0], margin=self.line_search_margin)
                
                # print(f"(line pixels) left: {np.sum(left_inds)} right:{np.sum(right_inds)}")
                
                # if np.sum(left_inds):
                #     self.left_fitx = poly_fit(lefty, leftx)
                # if np.sum(right_inds):
                #     self.right_fitx = poly_fit(righty, rightx)

                self.search_around_poly_counter += 1

                self.processed_images += 1
                # Unwarp/project back


                return  self.label(output, out_img) # self.__label_detection(255*binary_warped)#

        return output