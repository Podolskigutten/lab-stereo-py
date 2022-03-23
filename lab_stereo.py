import cv2
import numpy as np

from common_lab_utils import (Size, add_depth_point, visualize_matches, colours)
from kitti_interface import KittiCamera
from sparse_stereo_matcher import (SparseStereoMatcher)
from stereo_calibration import StereoCalibration
from stereo_camera import (StereoCamera, CameraIndex, CaptureMode, LaserMode)


def run_stereo_lab():
    laser_on = False
    lines = True

    cam = StereoCamera(CaptureMode.RECTIFIED)
    # cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)

    import sys
    cam = KittiCamera(*sys.argv[1:3])
    calibration = StereoCalibration.from_kitti(cam)
    print(f"calibration:\n{calibration}")

    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.BRISK_create(30, 0)
    stereo_matcher = SparseStereoMatcher(detector, desc_extractor)

    print("Press 'l' to toggle laser.")
    print("Press 'g' to toggle lines.")
    print("Press 'u' to toggle rectified/unrectified.")
    print("Press 'q' to quit.")

    matching_win = "Stereo matching"
    depth_win = "Stereo depth"
    dense_win = "Dense disparity"

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(matching_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(dense_win, cv2.WINDOW_AUTOSIZE)

    while True:
        # Grab raw images
        stereo_raw = cam.get_stereo_pair()

        # Rectify images.
        stereo_rectified = calibration.rectify(stereo_raw)

        # Perform sparse matching.
        stereo_matcher.match(stereo_rectified)

        # Visualize matched point correspondences
        match_image = visualize_matches(stereo_rectified, stereo_matcher)
        cv2.imshow(matching_win, match_image)

        # Visualize depth in meters for each point.
        fu = calibration.f
        bx = calibration.baseline
        vis_depth = cv2.cvtColor(stereo_rectified.left, cv2.COLOR_GRAY2BGR)
        
        if stereo_matcher.point_disparities is not None:
            for pt, d in stereo_matcher.point_disparities:
                depth = fu * bx / d
                add_depth_point(vis_depth, pt, depth)

        cv2.imshow(depth_win, vis_depth)
        

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Bye")
            break
        elif key == ord('l'):
            laser_on = not laser_on
            cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)
            print(f"Laser: {laser_on}")
        elif key == ord('g'):
            lines = not lines
            print(f"Lines: {lines}")
        elif key == ord('u'):
            rectified = not rectified
            cam.set_capture_mode(CaptureMode.RECTIFIED if rectified else CaptureMode.UNRECTIFIED)
            print(f"Rectified: {rectified}")


if __name__ == "__main__":
    run_stereo_lab()
