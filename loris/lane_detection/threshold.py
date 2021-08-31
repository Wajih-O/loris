import numpy as np
import cv2

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from loris.lane_detection.utils import grayscale


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray_img = grayscale(img)
    if orient not in {"x", "y"}:
        raise Exception("orient should be 'x' or 'y'")
    sobel = (cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient == 'x'
            else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) # Scale the result to an 8-bit range (0-255)
    # Apply lower and upper thresholds Create binary_output
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray_img = grayscale(img)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    abs_sobel = np.sqrt(sobel_x ** 2 + sobel_y **2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create and return a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ Threshold gradient direction """
    gray_img = grayscale(img)

    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary_output


def combined_threshold(input_image): # todo write a class threshold and
    ksize = 15 # Choose a larg odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(input_image, orient='x', sobel_kernel=ksize, thresh=(150, 200))
    grady = abs_sobel_thresh(input_image, orient='y', sobel_kernel=ksize, thresh=(150, 200))
    mag_binary = mag_thresh(input_image, sobel_kernel=ksize, mag_thresh=(40, 100))
    dir_binary = dir_threshold(input_image, sobel_kernel=ksize, thresh=(0, np.pi/3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_pipeline(img, ls_mean_min=100, sx_thresh=(8, 40)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    ls_mean = np.mean(hls[:, :, 1:], axis=-1)
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    sxbinary = sxbinary * (ls_mean > ls_mean_min)
    return sxbinary*255
