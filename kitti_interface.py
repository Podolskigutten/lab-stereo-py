import numpy as np
import cv2
import os


class KittiCamera:
    camera_id = {'GrayLeft': '00', 'GrayRight': '01', 'ColourLeft': '02', 'ColourRight': '03'}
    camera_from_id = {val: key for key, val in camera_id.items()}

    def __init__(self, dataset_path: str, calibration_path: str, get_grayscale_images=True):
        """Construct a camera interface to a Kitti stereo dataset.

        :param dataset_path: The path to the dataset.
        :param calibration_path: The path to the calibration.
        :param get_grayscale_images: Get grayscale instead of colour images.
        """

        # Create capture object for left camera.
        self._left_cap = self._create_capture_for_kitti_sequence(
            dataset_path, self.camera_id['GrayLeft'] if get_grayscale_images else self.camera_id['ColourLeft'])
        if not self._left_cap.isOpened():
            raise RuntimeError("Could not open left camera sequence")

        # Create capture object for right camera.
        self._right_cap = self._create_capture_for_kitti_sequence(
            dataset_path, self.camera_id['GrayRight'] if get_grayscale_images else self.camera_id['ColourRight'])
        if not self._right_cap.isOpened():
            raise RuntimeError("Could not open right camera sequence")

        # Set path to calibration file.
        self._calibration = self._read_kitti_calibration_file(calibration_path)

        # Initialise frame count.
        self._frame_count = 0

    def get_stereo_pair(self):
        success_left, left_frame = self._left_cap.read()
        success_right, right_frame = self._right_cap.read()

        if not success_left:
            print("End of left camera sequence")
        if not success_right:
            print("End of right camera sequence")

        self._frame_count += 1

        return left_frame, right_frame

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def calibration(self):
        return self._calibration

    @staticmethod
    def _create_capture_for_kitti_sequence(dataset_path, cam_num):
        image_sequence = os.path.join(dataset_path, f'image_{cam_num}', 'data', '0%09d.png')
        return cv2.VideoCapture(image_sequence, cv2.CAP_IMAGES)

    @staticmethod
    def _read_kitti_calibration_file(calibration_path):
        cam_to_cam_file_path = os.path.join(calibration_path, 'calib_cam_to_cam.txt')

        with open(cam_to_cam_file_path, mode='r') as file:
            raw_calibration = dict(line.strip().split(": ", 1) for line in file.readlines())

        calibration_for_camera = {}
        for camera_id in KittiCamera.camera_id.values():
            calibration_for_camera[KittiCamera.camera_from_id[camera_id]] = {
                'size': np.fromstring(raw_calibration[f'S_{camera_id}'], sep=' '),
                'calibration': np.fromstring(raw_calibration[f'K_{camera_id}'], sep=' ').reshape(3, 3),
                'distortion': np.fromstring(raw_calibration[f'D_{camera_id}'], sep=' '),
                'rotation': np.fromstring(raw_calibration[f'R_{camera_id}'], sep=' ').reshape(3, 3),
                'translation': np.fromstring(raw_calibration[f'T_{camera_id}'], sep=' '),
                'rectified_size': np.fromstring(raw_calibration[f'S_rect_{camera_id}'], sep=' '),
                'rectified_rotation': np.fromstring(raw_calibration[f'R_rect_{camera_id}'], sep=' ').reshape(3, 3),
                'rectified_projection': np.fromstring(raw_calibration[f'P_rect_{camera_id}'], sep=' ').reshape(3, 4)
            }

        return calibration_for_camera


if __name__ == "__main__":
    dataset_path = ''
    calibration_path = ''
    kitti_cam = KittiCamera(dataset_path, calibration_path)

    print("Calibration matrix for left camera:")
    print(kitti_cam.calibration['GrayLeft']['calibration'])

    while True:
        left_frame, right_frame = kitti_cam.get_stereo_pair()
        if left_frame is None or right_frame is None:
            break

        img = np.concatenate([left_frame, right_frame], axis=1)
        cv2.imshow('Test', img)

        key = cv2.waitKey(100)
        if key == ord('q'):
            print("Quit")
            break
