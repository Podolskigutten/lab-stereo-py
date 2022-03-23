import pyrealsense2 as rs2
import numpy as np
from enum import Enum, IntEnum, unique, auto
from common_lab_utils import (Size, StereoPair)


@unique
class CameraIndex(IntEnum):
    LEFT = 1
    RIGHT = 2


@unique
class CaptureMode(Enum):
    UNRECTIFIED = auto()
    RECTIFIED = auto()


@unique
class LaserMode(Enum):
    ON = auto()
    OFF = auto()


class StereoCamera:

    def __init__(self, capture_mode: CaptureMode):
        ir_size = Size(width=1280, height=800)

        self._pipe = rs2.pipeline()
        self.set_capture_mode(capture_mode)

    def __del__(self):
        self._pipe.stop()

    def get_info(self):
        device = self._pipe.get_active_profile().get_device()
        serial_number = device.get_info(rs2.camera_info.serial_number)
        device_product_line = str(device.get_info(rs2.camera_info.product_line))

        print(f" product line: {device_product_line}")
        print(f" serial: {serial_number}")

    def get_stereo_pair(self) -> StereoPair:
        data = self._pipe.wait_for_frames()
        frame_1 = np.asanyarray(data.get_infrared_frame(int(CameraIndex.LEFT)).get_data())
        frame_2 = np.asanyarray(data.get_infrared_frame(int(CameraIndex.RIGHT)).get_data())
        return StereoPair(frame_1, frame_2)

    def get_stamped_stereo_pair(self):
        data = self._pipe.wait_for_frames()
        frame_1 = np.asanyarray(data.get_infrared_frame(int(CameraIndex.LEFT)).get_data())
        frame_2 = np.asanyarray(data.get_infrared_frame(int(CameraIndex.RIGHT)).get_data())
        usec_timestamp = data.get_frame_metadata(rs2.frame_metadata_value.frame_timestamp)
        return StereoPair(frame_1, frame_2), usec_timestamp

    def get_framerate(self, camera: CameraIndex):
        profile = self.get_video_stream_profile(camera)
        return profile.fps()

    def get_resolution(self, camera: CameraIndex) -> Size:
        profile = self.get_video_stream_profile(camera)
        return Size(width=profile.width(), height=profile.height())

    def get_k_matrix(self, camera: CameraIndex):
        """bare for rs2.format.y8"""
        i = self.get_video_stream_profile(camera).get_intrinsics()
        return np.array([
            [i.fx, 9, i.ppx],
            [0, i.fy, i.ppy],
            [0, 0, 1]
        ])

    def get_distortion(self, camera: CameraIndex):
        """bare for rs2.format.y8"""
        d = self.get_video_stream_profile(camera).get_intrinsics().coeffs
        return np.array([d])

    def get_pose(self):
        selection = self._pipe.get_active_profile()
        left_stream = selection.get_stream(rs2.stream.infrared, int(CameraIndex.LEFT))
        right_stream = selection.get_stream(rs2.stream.infrared, int(CameraIndex.RIGHT))
        e = left_stream.get_extrinsics_to(right_stream)
        R = np.asarray(e.rotation).reshape((3, 3))
        t = np.asarray(e.translation)

        return R, t

    def set_capture_mode(self, capture_mode: CaptureMode):
        if capture_mode is CaptureMode.RECTIFIED:
            mode = rs2.format.y8
        elif capture_mode is CaptureMode.UNRECTIFIED:
            mode = rs2.format.y16

        ir_size = Size(width=1280, height=800)

        try:
            self._pipe.stop()
        except:
            pass
        cfg = rs2.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs2.stream.infrared, int(CameraIndex.LEFT), ir_size.width, ir_size.height, mode)
        cfg.enable_stream(rs2.stream.infrared, int(CameraIndex.RIGHT), ir_size.width, ir_size.height, mode)
        self._pipe.start(cfg)

    def get_video_stream_profile(self, camera: CameraIndex):
        return self._pipe.get_active_profile().get_stream(rs2.stream.infrared, int(camera)).as_video_stream_profile()

    def set_laser_mode(self, laser_mode: LaserMode):
        """'True' or anyting"""
        depth_sensor = self._pipe.get_active_profile().get_device().first_depth_sensor()

        if not depth_sensor.supports(rs2.option.emitter_enabled):
            return

        if laser_mode is LaserMode.ON:
            depth_sensor.set_option(rs2.option.emitter_enabled, 1)
        else:
            depth_sensor.set_option(rs2.option.emitter_enabled, 0)
