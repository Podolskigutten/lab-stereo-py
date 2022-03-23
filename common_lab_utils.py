import cv2.cv2
import numpy as np
from dataclasses import dataclass


class Size:
    """Represents image size"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

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


class DotDict(dict):
    """
    dot.notation access to dictionary attributes.

    https://stackoverflow.com/a/23689767/14325545
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


colours = DotDict({
    'green': (0, 255, 0),
    'red': (0, 0, 255)
})

font = DotDict({
    'face': cv2.FONT_HERSHEY_PLAIN,
    'scale': 1.0
})


def retain_best(keypoints, num_to_keep):
    """Retains the given number of keypoints with highest response"""
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best


def visualize_matches(stereo_pair, stereo_matcher):
    """
    This function will create an image that shows corresponding keypoints in two images.

    :param stereo_pair: The two images
    :param stereo_matcher: The matcher that has extracted the keypoints
    :return: an image with visualization of keypoint matches
    """
    cv2.putText(stereo_pair.left, f"LEFT", (10, 20), font.face, font.scale, colours.green)
    cv2.putText(stereo_pair.right, f"RIGHT", (10, 20), font.face, font.scale, colours.green)
    vis_img = cv2.drawMatches(
        stereo_pair.left, stereo_matcher.keypoints_left,
        stereo_pair.right, stereo_matcher.keypoints_right,
        stereo_matcher.matches, None, flags=2)
    return vis_img
