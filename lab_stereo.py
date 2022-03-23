import cv2
import numpy as np

from common_lab_utils import (Size, visualize_matches)
from sparse_stereo_matcher import (SparseStereoMatcher)
from stereo_calibration import StereoCalibration
from stereo_camera import (StereoCamera, CameraIndex, CaptureMode, LaserMode)


def run_stereo_lab():
    laser_on = False
    rectified = False
    lines = True

    cam = StereoCamera(CaptureMode.RECTIFIED)
    calibration = StereoCalibration.from_camera(cam)

    cam.set_capture_mode(CaptureMode.RECTIFIED if rectified else CaptureMode.UNRECTIFIED)
    cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)

    print("Connected to RealSense camera:")
    cam.get_info()
    print(f" resolution: {cam.get_resolution(CameraIndex.LEFT)}")
    print(f" framerate: {cam.get_framerate(CameraIndex.LEFT)}")
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
        match_image = visualize_matches(stereo_rectified, stereo_matcher);
        cv2.imshow(matching_win, match_image)

        # Visualise
        pair_raw = np.hstack((stereo_raw.left, stereo_raw.right))
        pair_rectified = np.hstack((stereo_rectified.left, stereo_rectified.right))

        zeros = np.zeros(pair_raw.shape[:2], dtype=pair_raw.dtype)
        viz = cv2.merge([pair_raw, pair_rectified, zeros])

        # Draw lines along the image rows.
        # For rectified pair, these should coincide with the epipolar lines.
        if lines:
            for i in np.arange(50, viz.shape[0], 50):
                cv2.line(viz, (0, i), (viz.shape[1], i), (0, 0, 65535))

        cv2.imshow('RealSense', viz)
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
