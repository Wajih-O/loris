<style>
img[src$="centerme"] {
  display:block;
  margin: 0 auto;
}
</style>

# Loris (a CV perception lab)

## Advanced Lane Finding [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


<!-- ![Lanes Image](./examples/example_output.jpg) -->

This project, implements a software pipeline to identify the lane boundaries in a video.


## Preprocessing

1. Camera calibration

The pipeline's first step is to undistort the camera image.  This step depends on camera calibration. The calibration helper `loris.calibration.utils.calibrate` generates the calibration artifacts from the chessboard images passed as parameters. For this project, the chessboard images in `data/calibration` directory images are used.

2. Perspective transform preparation: a rectangular region (lane section) is extracted from a straight lane image then warped (into a rectangular region). The `Warper` class part of `loris.utils` implements and holds as an object the needed transformation and inverse transformation for the pipeline. The following parameters are used for current project.

```python
src = {"tl": [554, 480], "tr": [736, 480], "bl": [214, 720], "br": [1116, 720]}
dest = {"tl": [320, 0], "tr": [960, 0], "bl": [320, 720], "br": [960, 720]}
```

![warping transfomation](./output_images/perspective_transform.jpg?centerme) 


3. Thresholding image: Before transforming the image/frame using the previously setup perspective transformation,  a thresholding step is performed to keep only (as much as possible) lane pixels while filtering the non-relevant regions. The proposed threshold pipeline is implemented as the `threshold_pipeline` function, part of  `lane_detection/threshold.py` and worked well for the test video (eliminating the central shadow like line ). Below a demo/visualization of the thresholding component (left image)

```python

from loris.lane_detection.threshold import threshold_pipeline, combined_threshold
from loris.utils import Warper

warper = Warper(rect_dict2nparray(src), rect_dict2nparray(dest))
status, output = undistort(sample_image, calibration_output)

if (status):
    thresholding_output = threshold_pipeline(output)
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(1, 2, 1)
    warped = warper.warp(thresholding_output)
    warped_ax = plt.subplot(1, 2, 2)
    ax.imshow(thresholding_output, cmap="gray")
    highlight_corners(ax, src)
    warped_ax.imshow(warped, cmap="gray")
    highlight_corners(warped_ax, dest)
    plt.tight_layout()

```

![perspective transformed thresholded image](./output_images/perspective_transform_on_thresholded.jpg?centerme) 


4. For each of thresholded image/frame we then apply a perspective transform (transforming a real word rectangular region) "bird-eye view" (right image of previous visualization). Then we do the core lane detection on the transformed (thresholded and warped) image.

## Core-processing

5. A first step of the core line detection then is performed on the binarized then warped image:
    The pipeline implemented in `Lane_detector` keeps track (statefully) of a blended left/right information throughout processing the video.
    The left and right sides of the lane are initialized simultaneously using `find_lane_pixels`. After the initialization, each line is represented as fitted second-order polynomial `x = f(y)`, then for each new frame/image, this polynomial is updated. Lane initialization and update are part of `loris.lane_detection.lane_detector.py` while the `Line` class keeps track of each lane side (a left one and a right one), composing the `LineDetector` processing/labeling object. 

## Initialization


As initializer `find_lane_pixels` is based on a  histogram peak search to define each side (white/thresholded image) pixels.

![Lane initialization histogram](./output_images/perspective_transform_white_pixels_hitogram.jpg?centerme) 



The positions of the white pixels for each side are then used to fit a second-order polynomial `x = f(y)` for each of the pixel collections (below represented in continuous yellow line )
![Lane initialization polynomes](./output_images/initalized_lines_ploynomes.jpg?centerme)


On the first frame of the challenge video, the initialization output looks as below:

![Lane initialization polynomes frame 0](./output_images/initalized_lines_ploynomes_frame_0.jpg?centerme)


## Update

After initialization, the lane sides are kept up-to-date (on the current frame) with a search of lane pixels around previously defined polynomials.  The search is done within a predefined margin in the thresholded and warped current frame. We represent the found pixels below in red and blue, respectively, for the left and right sides of the lane). The previous (n look-back) fitted lines are also used/blended as a prior (see `LaneDetector` and `Line` classes for details)

![Lane updates](./output_images/updates_with_next_frames.jpg?centerme)

## Full pipeline (LaneDetector)


The initialization and update modes are implemented in the `LaneDetector` class. The initialization is invoked with the first call of image processing; then, the update mode will carry over through frames. Note this behavior is abstracted and not exposed by the `LaneDetector` interface `LaneDetector.process_image`.

```python
# For video processing
from glob import glob
from loris.lane_detection.lane_detector import LaneDetector
from loris.utils import Warper
from loris.calibration.utils import calibrate, undistort

calibration_images = [image_path for image_path in sorted(glob("../data/calibration/*.jpg")) if np.array_equal(cv2.imread(image_path).shape[:2],  np.array([720, 1280]))]

# Calibration artifacts extraction 
calibration_output = calibrate(calibration_images)

challenge_video = VideoFileClip('../data/videos/challenge_video.mp4')
output_video = '../data/videos/challenge_video_output.mp4'

# Lane detector (processor)
ld = LaneDetector(calibration_params=calibration_output, warper=Warper( rect_dict2nparray(src),  rect_dict2nparray(dest)), margin=60, look_back=4)

processed_clip = challenge_video.fl_image(ld.process_image)

%time processed_clip.write_videofile(output_video, audio=False)

```

## Curvature of the lane

The radius of curvature.
$ R_{curve} = \frac{[[1+ {\frac{d_x}{d_y}}^2]^\frac{3}{2}]}{\frac{d^2x}{d_y}} $

Lane pixels are located (for each side) as second order polynomial $x= f(y) = A y^2  + B y + C$

The lane cuvature is definded by $R_{curve} = \frac{[1 + [2*A*y+B]^2]^\frac{3}{2}}{2*A} $


See lane `lane_detector.convert` implementing the transofmation to meter (space/unit) polynome before applying the (radius of curvature) using the `lane_detector.gen_curvature_calculator`

## Relative vehicle position to the lane sides

Assuming that the camera is mounted somewhere on the vehicle's central (longitudinal) axis, the lane car offset from the center is then obtained by: transforming the center of the original image to warped space, then computing the distance to the lane center convert it in meter. A positive offset value represents a car leaning to the right side, while a negative offset the vehicle leaning to the left side. 




# Lane detection on video (section)
![](./output_images/video_output.gif)


Pipeline demo/details  in `notebooks/adv_lane_detection.ipynb`