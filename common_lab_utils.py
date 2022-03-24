import numpy as np
from dataclasses import dataclass
import cv2


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
