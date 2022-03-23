import cv2
from common_lab_utils import (Size, StereoPair)

class SparseStereoMatcher:
    def __init__(self, detector, desc_extractor):
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._matcher = cv2.BFMatcher_create(self._desc_extractor.defaultNorm())

    def match(self, stereo_pair:StereoPair):
        """Detect and match keypoints in both images."""
        n_grid_cols = stereo_pair.left.shape[1] // 16
        n_grid_rows = stereo_pair.left.shape[0] // 16
        grid_size = Size(n_grid_cols, n_grid_rows)
        patch_width = 32

        # Detect and describe features in the left image.
        del self._keypoints_left
        self._keypoints_left = self._detect_in_grid(stereo_pair.left, grid_size, 1, patch_width)
        keypoints, descriptors = self._desc_extractor.compute(stereo_pair.left, self._keypoints_left)
        # fixme

    @property
    def keypoints_left(self):
        """keypoints detected in the left image"""
        return self._keypoints_left
    
    @property
    def keypoints_right(self):
        """keypoints detected in the right image"""
        return self._keypoints_right
    
    @property
    def matches(self):
        """matched keypoints (keypoint pairs in left and right images)"""
        return self._good_matches

    @property
    def point_disparities(self):
        """computed disparity values for each keypoint pair in matches"""
        return self._point_disparities
    
    def _compute_disparities(self):
        pass  # fixme

    def _extract_good_matches(self, matches):
        pass  # fixme

    def _detect_in_grid(self, image, grid_size, max_in_cell, patch_width):
        pass  # fixme