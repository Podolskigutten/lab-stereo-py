import cv2
import numpy as np
import timeit

from common_lab_utils import (Size, add_depth_point, visualize_matches, colours, font)
from kitti_interface import KittiCamera
from sparse_stereo_matcher import (SparseStereoMatcher)
from stereo_calibration import StereoCalibration
from stereo_camera import (StereoCamera, CameraIndex, CaptureMode, LaserMode)


def run_stereo_lab(cam, calibration):
    print(f"camera:\n{cam}\ncalibration:\n{calibration}")

    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.BRISK_create(30, 0)
    stereo_matcher = SparseStereoMatcher(detector, desc_extractor)
    use_grid = False
    laser_on = False
    rectified = True

    print("Press 'q' to quit.")
    print("Press 'g' to toggle feature detection in grid.")
    if isinstance(cam, StereoCamera):
        print("Press 'l' to toggle laser.")
        print("Press 'u' to toggle rectified/unrectified.")
    
    matching_win = "Stereo matching"
    depth_win = "Stereo depth"
    dense_win = "Dense disparity"

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(matching_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(dense_win, cv2.WINDOW_AUTOSIZE)

    while True:
        # Grab raw images
        start = timeit.default_timer()
        stereo_raw = cam.get_stereo_pair()

        # Rectify images.
        stereo_rectified = calibration.rectify(stereo_raw)
        end = timeit.default_timer()
        duration_grabbing = end - start

        # Perform sparse matching.
        start = timeit.default_timer()
        stereo_matcher.match(stereo_rectified, use_grid)
        end = timeit.default_timer()
        duration_matching = end - start

        # Visualize matched point correspondences
        start = timeit.default_timer()
        match_image = visualize_matches(stereo_rectified, stereo_matcher, duration_grabbing, duration_matching)
        cv2.imshow(matching_win, match_image)

        # Visualize depth in meters for each point.
        fu = calibration.f
        bx = calibration.baseline
        vis_depth = cv2.cvtColor(stereo_rectified.left, cv2.COLOR_GRAY2BGR)
        
        if stereo_matcher.point_disparities is not None:
            for pt, d in stereo_matcher.point_disparities:
                depth = fu * bx / d
                add_depth_point(vis_depth, pt, depth)

        end = timeit.default_timer()
        duration_visualisation = end - start
        cv2.putText(vis_depth, f"visualization:  {duration_visualisation:.2f} s", (10, 40), font.face, font.scale, colours.red)

        cv2.imshow(depth_win, vis_depth)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Bye")
            break
        elif key == ord('g'):
            use_grid = not use_grid
            print(f"use grid: {use_grid}")
        elif key == ord('u') and isinstance(cam, StereoCamera):
            rectified = not rectified
            cam.set_capture_mode(CaptureMode.RECTIFIED if rectified else CaptureMode.UNRECTIFIED)
            print(f"Rectified: {rectified}")
        elif key == ord('l') and isinstance(cam, StereoCamera):
            laser_on = not laser_on
            cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)
            print(f"Laser: {laser_on}")


def kitti():
    import sys
    cam = KittiCamera(*sys.argv[1:3])
    calibration = StereoCalibration.from_kitti(cam)
    return cam, calibration

def realsense():
    cam = StereoCamera(CaptureMode.RECTIFIED)
    calibration = StereoCalibration.from_realsense(cam)
    return cam, calibration

if __name__ == "__main__":
    #run_stereo_lab(*kitti())
    run_stereo_lab(*realsense())
