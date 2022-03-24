import numpy as np
from dataclasses import dataclass
import cv2
from pylie import SE3


class Size:
    """Represents image size"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    def __str__(self):
        return f"w: {self._width}, h: {self._height}"

    @classmethod
    def from_numpy_shape(cls, shape):
        return cls(*shape[1::-1])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def as_cv_size(self):
        return np.array((self._width, self._height), dtype=int)


@dataclass
class StereoPair:
    left: np.ndarray
    right: np.ndarray
    
    def __iter__(self):
        print("iter")
        return iter((self.left, self.right))


class StereoMatchingResult:
    def __init__(self, keypoints_left=(), keypoints_right=(), matches=()):
        self._keypoints_left = keypoints_left
        self._keypoints_right = keypoints_right
        self._matches = matches

    @property
    def keypoints_left(self):
        return self._keypoints_left

    @property
    def keypoints_right(self):
        return self._keypoints_right

    @property
    def matches(self):
        return self._matches


class CvStereoMatcherWrap:
    def __init__(self, matcher):
        self._matcher = matcher

    def compute(self, stereo_rectified: StereoPair):
        num_disparities = self._matcher.getNumDisparities()
        padded_l = cv2.copyMakeBorder(stereo_rectified.left, 0, 0, num_disparities, 0, cv2.BORDER_CONSTANT,
                                      value=(0, 0, 0))
        padded_r = cv2.copyMakeBorder(stereo_rectified.right, 0, 0, num_disparities, 0, cv2.BORDER_CONSTANT,
                                      value=(0, 0, 0))
        disparity_padded = self._matcher.compute(padded_l, padded_r)

        # Unpad.
        pixels_to_unpad_left = num_disparities
        pixels_to_unpad_right = padded_l.shape[1]
        disparity_16bit_fixed = disparity_padded[:, pixels_to_unpad_left:pixels_to_unpad_right]

        # Convert from 16 bit fixed point.
        ratio_1_over_16 = 1.0 / 16.0
        disparity = (disparity_16bit_fixed * ratio_1_over_16).astype(np.float32)

        return disparity


def homogeneous(x):
    """Transforms Cartesian column vectors to homogeneous column vectors"""
    return np.r_[x, [np.ones(x.shape[1])]]


def hnormalized(x):
    """Transforms homogeneous column vector to Cartesian column vectors"""
    return x[:-1] / x[-1]


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self,
                 calibration_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray,
                 pose_world_camera: SE3):
        """Constructs the camera model.
        :param calibration_matrix: The intrinsic calibration matrix.
        :param distortion_coeffs: Distortion coefficients on the form [k1, k2, p1, p2, k3].
        :param pose_world_camera: The pose of the camera in the world coordinate system.
        """
        self._calibration_matrix = calibration_matrix
        self._calibration_matrix_inv = np.linalg.inv(calibration_matrix)
        self._distortion_coeffs = distortion_coeffs
        self._pose_world_camera = pose_world_camera
        self._camera_projection_matrix = self._compute_camera_projection_matrix()

    def _compute_camera_projection_matrix(self):
        return self._calibration_matrix @ self._pose_world_camera.inverse().to_matrix()[:3, :]

    def project_world_point(self, point_world):
        """Projects a world point into pixel coordinates.
        :param point_world: A 3D point in world coordinates.
        """

        if point_world.ndim == 1:
            # Convert to column vector.
            point_world = point_world[:, np.newaxis]

        return hnormalized(self._camera_projection_matrix @ homogeneous(point_world))

    def undistort_image(self, distorted_image):
        """Undistorts an image corresponding to the camera model.
        :param distorted_image: The original, distorted image.
        :returns: The undistorted image.
        """

        return cv2.undistort(distorted_image, self._calibration_matrix, self._distortion_coeffs)

    def pixel_to_normalised(self, point_pixel):
        """Transform a pixel coordinate to normalised coordinates
        :param point_pixel: The 2D point in the image given in pixels.
        """

        if point_pixel.ndim == 1:
            # Convert to column vector.
            point_pixel = point_pixel[:, np.newaxis]

        return self._calibration_matrix_inv @ homogeneous(point_pixel)

    @property
    def pose_world_camera(self):
        """The pose of the camera in world coordinates."""
        return self._pose_world_camera

    @property
    def calibration_matrix(self):
        """The intrinsic calibration matrix K."""
        return self._calibration_matrix

    @property
    def calibration_matrix_inv(self):
        """The inverse calibration matrix K^{-1}."""
        return self._calibration_matrix_inv

    @property
    def distortion_coeffs(self):
        """The distortion coefficients on the form [k1, k2, p1, p2, k3]."""
        return self._distortion_coeffs

    @property
    def projection_matrix(self):
        """The projection matrix P."""
        return self._camera_projection_matrix