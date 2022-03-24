import cv2
import numpy as np
import timeit

from cv_stereo_matcher_wrap import (CvStereoMatcherWrap)
from common_lab_utils import (add_depth_point, visualize_matches, colours, font)
from kitti_interface import KittiCamera
from sparse_stereo_matcher import (SparseStereoMatcher)
from stereo_calibration import StereoCalibration
from real_sense_stereo_camera import (RealSenseStereoCamera, LaserMode)


def run_stereo_lab(cam, calibration):
    print(f"camera:\n{cam}\ncalibration:\n{calibration}")

    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.ORB_create(nlevels=1)
    stereo_matcher = SparseStereoMatcher(detector, desc_extractor)
    use_anms = False
    laser_on = False
    dense = False

    print("Press 'q' to quit.")
    print("Press 'g' to toggle adaptive non-maximal suppression.")
    print("Press 'd' to toggle dense processing.")
    if isinstance(cam, RealSenseStereoCamera):
        print("Press 'l' to toggle laser.")
    
    matching_win = "Stereo matching"
    depth_win = "Stereo depth"
    dense_win = "Dense disparity"

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(matching_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_win, cv2.WINDOW_AUTOSIZE)

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
        match_result = stereo_matcher.match(stereo_rectified, use_anms)
        end = timeit.default_timer()
        duration_matching = (end - start) * 1000

        # Visualize matched point correspondences
        match_image = visualize_matches(stereo_rectified, match_result, duration_grabbing, duration_matching)
        cv2.imshow(matching_win, match_image)

        # Visualize depth in meters for each point.
        fu = calibration.f
        bx = calibration.baseline
        vis_depth = cv2.cvtColor(stereo_rectified.left, cv2.COLOR_GRAY2BGR)
        
        if match_result.point_disparities is not None:
            for pt, d in match_result.point_disparities:
                depth = fu * bx / d
                add_depth_point(vis_depth, pt, depth)

        cv2.imshow(depth_win, vis_depth)

        # Dense stereo matching using OpenCV
        if dense:
            start = timeit.default_timer()
            dense_matcher = CvStereoMatcherWrap(cv2.StereoSGBM_create(0, 192, 5))
            dense_disparity = dense_matcher.compute(stereo_rectified)

            dense_depth = (calibration.f * calibration.baseline) / dense_disparity
            max_depth = 1.0
            dense_depth[(dense_disparity < 0) | (dense_depth > max_depth)] = 0
            dense_depth /= max_depth

            # Colormap
            dense_depth = cv2.cvtColor(dense_depth * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            dense_depth = cv2.applyColorMap(dense_depth, cv2.COLORMAP_JET)
            
            end = timeit.default_timer()
            duration_dense = end - start
            cv2.putText(dense_depth, f"dense:  {duration_dense:.2f} s", (10, 20), font.face, font.scale, colours.white)

            cv2.imshow(dense_win, dense_depth)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Bye")
            break
        elif key == ord('g'):
            use_anms = not use_anms
            print(f"Anms: {use_anms}")
        elif key == ord('d'):
            dense = not dense
            print(f"Dense: {dense}")
            if dense:
                cv2.namedWindow(dense_win, cv2.WINDOW_AUTOSIZE)
        elif key == ord('l') and isinstance(cam, RealSenseStereoCamera):
            laser_on = not laser_on
            cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)
            print(f"Laser: {laser_on}")


def kitti():
    import sys
    cam = KittiCamera(*sys.argv[1:3])
    calibration = StereoCalibration.from_kitti(cam)
    return cam, calibration


def realsense():
    cam = RealSenseStereoCamera()
    calibration = StereoCalibration.from_realsense(cam)
    return cam, calibration


if __name__ == "__main__":
    # run_stereo_lab(*kitti())
    run_stereo_lab(*realsense())
