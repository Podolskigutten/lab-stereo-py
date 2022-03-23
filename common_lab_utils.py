import cv2.cv2
import numpy as np
from dataclasses import dataclass


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


@dataclass
class colours:
    green = (0, 255, 0)
    red = (0, 0, 255)


@dataclass
class font:
    face = cv2.FONT_HERSHEY_PLAIN
    scale = 1.0


def retain_best(keypoints, num_to_keep):
    """Retains the given number of keypoints with highest response"""
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return np.asanyarray(keypoints)[best]


def visualize_matches(stereo_pair, stereo_matcher):
    """
    This function will create an image that shows corresponding keypoints in two images.

    :param stereo_pair: The two images
    :param stereo_matcher: The matcher that has extracted the keypoints
    :return: an image with visualization of keypoint matches
    """
    if stereo_matcher.matches is None:
        return np.hstack((stereo_pair.left, stereo_pair.right))

    cv2.putText(stereo_pair.left, f"LEFT", (10, 20), font.face, font.scale, colours.green)
    cv2.putText(stereo_pair.right, f"RIGHT", (10, 20), font.face, font.scale, colours.green)
    vis_img = cv2.drawMatches(
        stereo_pair.left, stereo_matcher.keypoints_left,
        stereo_pair.right, stereo_matcher.keypoints_right,
        stereo_matcher.matches, None, flags=2)
    return vis_img

def add_depth_point(img, px, depth):
    """
    In an image, draw a cross at a pixel coordinate with a depth value printed next to it.

    :param img: The image to draw on.
    :param px:  The pixel position of the depth measurement.
    :param depth: The depth value
    """
    marker_size = 5
    cv2.drawMarker(img, px, colours.green, cv2.MARKER_CROSS, marker_size)
    cv2.putText(img, f"{depth:.2f}", px, font.face, font.scale, colours.green)
