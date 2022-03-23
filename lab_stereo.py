import cv2
import numpy as np

from common_lab_utils import (Size)
from stereo_camera import (StereoCamera, CameraIndex, CaptureMode, LaserMode)

def run_stereo_lab():
    laser_on = False
    rectified = False
    lines = True

    cam = StereoCamera(CaptureMode.RECTIFIED if rectified else CaptureMode.UNRECTIFIED)
    cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)

    print("Connected to RealSense camera:")
    cam.get_info()
    print(f" resolution: {cam.get_resolution(CameraIndex.LEFT)}")
    print(f" framerate: {cam.get_framerate(CameraIndex.LEFT)}")
    print("Press 'l' to toggle laser.")
    print("Press 'g' to toggle lines.")
    print("Press 'u' to toggle rectified/unrectified.")
    print("Press 'q' to quit.")
    
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    
    while True:
        frame_1, frame_2, ts = cam.get_stamped_stereo_pair()
        pair = np.hstack((frame_1, frame_2))
        pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)

        # Draw lines along the image rows.
        # For rectified pair, these should coincide with the epipolar lines.
        if lines:
            for i in np.arange(50, pair.shape[0], 50):
                cv2.line(pair, (0, i), (pair.shape[1], i), (0,0,65535))
        
        cv2.imshow('RealSense', pair)
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