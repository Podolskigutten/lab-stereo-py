import cv2
import numpy as np
from common_lab_utils import (Size, StereoPair, retain_best)


class SparseStereoMatcher:
    def __init__(self, detector, desc_extractor):
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._matcher = cv2.BFMatcher_create(self._desc_extractor.defaultNorm())

    def match(self, stereo_pair: StereoPair):
        """Detect and match keypoints in both images."""
        n_grid_cols = stereo_pair.left.shape[1] // 16
        n_grid_rows = stereo_pair.left.shape[0] // 16
        grid_size = Size(n_grid_cols, n_grid_rows)
        patch_width = 32

        # Detect and describe features in the left image.
        self._keypoints_left = self._detect_in_grid(stereo_pair.left, grid_size, 1, patch_width)
        self._keypoints_left, query_descriptors = self._desc_extractor.compute(stereo_pair.left, self._keypoints_left)

        # Detect and describe features in the right image.
        self._keypoints_right = self._detect_in_grid(stereo_pair.right, grid_size, 1, patch_width)
        self._keypoints_right, train_descriptors = self._desc_extractor.compute(stereo_pair.right,
                                                                                self._keypoints_right)

        # Match features
        matches = self._matcher.match(query_descriptors, train_descriptors)
        self._extract_good_matches(matches)
        self._compute_disparities()

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

    def _detect_in_grid(self, image, grid_size: Size, max_in_cell, patch_width):
        image_size = Size.from_numpy_shape(image.shape)
        height = image_size.height // grid_size.height
        width = image_size.width // grid_size.width

        patch_rad = patch_width // 2

        all_keypoints = ()
        for x in range(grid_size.width):
            for y in range(grid_size.height):
                row_range = slice(max(y * height - patch_rad, 0), min((y + 1) * height + patch_rad, image_size.height))
                col_range = slice(max(x * width - patch_rad, 0), min((x + 1) * width + patch_rad, image_size.width))
                sub_img = image[row_range, col_range]

                grid_keypoints = self._detector.detect(sub_img)
                if not grid_keypoints:
                    continue
                best = retain_best(grid_keypoints, max_in_cell)
                grid_keypoints = np.asarray(grid_keypoints)[best]

                for keypoint in grid_keypoints:
                    pt = list(keypoint.pt)
                    pt[0] += col_range.start
                    pt[1] += row_range.start
                    keypoint.pt = tuple(pt)

                all_keypoints = all_keypoints + grid_keypoints

        return all_keypoints

    def _extract_good_matches(self, matches):
        self._good_matches = []
        for match in matches:  # fixme enklere måte?
            epipolar_is_ok = np.abs(
                self._keypoints_left[match.queryIdx].pt[0] - self._keypoints_right[match.trainIdx].pt[0]) < 1.0
            disparity_is_positive = self._keypoints_left[match.queryIdx].pt[1] > \
                                    self._keypoints_right[match.trainIdx].pt[1]
            if epipolar_is_ok and disparity_is_positive:
                self._good_matches.append(match)

    def _compute_disparities(self):
        self._point_disparities = []
        for match in self._good_matches:  # fixme enklere måte?
            left_point = self._keypoints_left[match.queryIdx].pt
            right_point = self._keypoints_right[match.trainIdx].pt
            disparity = left_point[1] - right_point[1]
            self._point_disparities.append((left_point, disparity))
