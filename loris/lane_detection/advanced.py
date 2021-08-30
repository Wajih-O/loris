from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import cv2

# design the interaction with the line class !!!

def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    Find lane pixels

    :param nwindows: sliding windows (nbr or height dir. buckets)
    :param margin: the width of the windows +/- margin
    :param minpix: minimum number of pixels found to recenter the window
    """
    # a histogram of the bottom half of the image (expecting binary input)
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # initializes the left and right lane side (windows)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows and image shape
    window_height = int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current +  margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2)

        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_poly(leftx, lefty, rightx, righty):
    """ Fit a second order polynomial to each lane side x = f(y) """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def non_zero(binary_image):
    """ return white (1/non-zero) pixel coordinates x, y (nonzerox, nonzeroy)"""
    nonzero = binary_image.nonzero()
    return np.array(nonzero[1]), np.array(nonzero[0]) 


def apply_polynomial(poly, y):
    """ apply polynome P on y -> P(y) """
    return np.sum(np.array([ coef*(np.power(y, exp_))for exp_, coef in zip(reversed(range(len(poly))), poly)]), axis=0).astype(int)


def search_around_poly(nonzerox, nonzeroy, poly2, margin=100):
    """ search around polynomial fitted line/curve
    (Set the area of search based on activated x-values)
    white pixels coordinates within the +/- margin of our polynomial function"""
    inds = ((nonzerox > (poly2[0]*(nonzeroy**2) + poly2[1]*nonzeroy +
                    poly2[2] - margin)) & (nonzerox < (poly2[0]*(nonzeroy**2) +
                    poly2[1]*nonzeroy + poly2[2] + margin)))
    return  nonzerox[inds], nonzeroy[inds], inds


def poly_fit(y, x, degree=2):
    return np.polyfit(y, x, degree)


def visualize(image, left_fit, right_fit, ax = plt):
    """ Find our lane pixels first. """
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img = np.dstack((image,)*3)
    # out_img[lefty, leftx] = [255, 0, 0] # colors left lane side in blue
    # out_img[righty, rightx] = [0, 0, 255] # colors right lane side in red

    # Plots the left and right polynomials on the lane lines
    ax.imshow(out_img)
    ax.plot(left_fitx, ploty, color='red')
    ax.plot(right_fitx, ploty, color='yellow')


def measure_curvature_pixels(left_fit, right_fit):
    """  Generates a right/left curve calculator from a left/right lane side
         (second order polynomial fit)
    :param left_fit: second order polynomial fit (left lane side)
    :param right+fit: second order polynomial fit (right lane side)
    """
    def right_left_curvature(y_eval):
        """ Calculation of R_curve (radius of curvature)"""
        left_curverad = pow(1 + pow((2*left_fit[0]*y_eval + left_fit[1]), 2), 3/2) / (2*np.abs(left_fit[0]))
        right_curverad = pow(1 + pow((2*right_fit[0]*y_eval + right_fit[1]), 2), 3/2) / (2*np.abs(right_fit[0]))

        return left_curverad, right_curverad

    return right_left_curvature