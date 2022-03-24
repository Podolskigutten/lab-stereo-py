import cv2
import numpy as np
from common_lab_utils import (StereoPair)

class CvStereoMatcherWrap:
    def __init__(self, matcher):
        self._matcher = matcher
    
    def compute(self, stereo_rectified : StereoPair):
        num_disparities = self._matcher.getNumDisparities()
        padded_l = cv2.copyMakeBorder(stereo_rectified.left,  0, 0, num_disparities, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        padded_r = cv2.copyMakeBorder(stereo_rectified.right, 0, 0, num_disparities, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        disparity_padded = self._matcher.compute(padded_l, padded_r)

        # Unpad.
        pixels_to_unpad_left = num_disparities
        pixels_to_unpad_right = padded_l.shape[1]
        disparity_16bit_fixed = disparity_padded[:, pixels_to_unpad_left:pixels_to_unpad_right]

        # Convert from 16 bit fixed point.
        ratio_1_over_16 = 1.0 / 16.0
        disparity = (disparity_16bit_fixed * ratio_1_over_16).astype(np.float32)

        return disparity