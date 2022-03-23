import cv2
import numpy as np

from common_lab_utils import (Size, add_depth_point, visualize_matches, colours)
from kitti_interface import KittiCamera
from sparse_stereo_matcher import (SparseStereoMatcher)
from stereo_calibration import StereoCalibration
from stereo_camera import (StereoCamera, CameraIndex, CaptureMode, LaserMode)


def run_stereo_lab(cam, calibration):
    print(f"camera:\n{cam}\ncalibration:\n{calibration}")

    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.BRISK_create(30, 0)
    stereo_matcher = SparseStereoMatcher(detector, desc_extractor)

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
