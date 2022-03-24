import cv2
import numpy as np
from common_lab_utils import (Size, StereoPair, retain_best, colours)


class SparseStereoMatcher:
    def __init__(self, detector, desc_extractor):
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._matcher = cv2.BFMatcher_create(self._desc_extractor.defaultNorm())

    def match(self, stereo_pair: StereoPair, use_grid:bool=True):
        """Detect and match keypoints in both images."""
        
        n_grid_cols = stereo_pair.left.shape[1] // 16
        n_grid_rows = stereo_pair.left.shape[0] // 16
        grid_size = Size(n_grid_cols, n_grid_rows)
        patch_width = 32

        self._keypoints_left = None
        self._keypoints_right = None
        self._matches = None
        self._point_disparities = None

        # Detect and describe features in the left image.
        self._keypoints_left = self._detect_in_grid(stereo_pair.left, grid_size, 1, patch_width) if use_grid else self._detector.detect(stereo_pair.left)
        self._keypoints_left, query_descriptors = self._desc_extractor.compute(stereo_pair.left, self._keypoints_left)

        # Detect and describe features in the right image.
        self._keypoints_right = self._detect_in_grid(stereo_pair.right, grid_size, 1, patch_width) if use_grid else self._detector.detect(stereo_pair.right)
        self._keypoints_right, train_descriptors = self._desc_extractor.compute(stereo_pair.right, self._keypoints_right)

        if self._keypoints_left and self._keypoints_right:
            # Match features
            matches = self._matcher.match(query_descriptors, train_descriptors)
            self._good_matches = self._extract_good_matches(self._keypoints_left, self._keypoints_right, matches, epipolar_limit=1.0)
            if len(self._good_matches):
                self._compute_disparities(self._keypoints_left, self._keypoints_right, self._good_matches)

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

                grid_keypoints = self._detector.detect(image[row_range, col_range])
                if not grid_keypoints:
                    continue

                grid_keypoints = retain_best(grid_keypoints, max_in_cell)

                for i in range(len(grid_keypoints)):
                    grid_keypoints[i].pt = (grid_keypoints[i].pt[0] + col_range.start, grid_keypoints[i].pt[1] + row_range.start)

                all_keypoints = all_keypoints + tuple(grid_keypoints)

        return all_keypoints

    def _extract_good_matches(self, keypoints_l, keypoints_r, matches, epipolar_limit):
        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]
        
        pts_l = np.array([k.pt for k in np.asarray(keypoints_l)[query_idx]])
        pts_r = np.array([k.pt for k in np.asarray(keypoints_r)[train_idx]])
        
        epipolar_is_ok = abs(pts_l[:,1] - pts_r[:,1]) < epipolar_limit
        disparity_is_positive = pts_l[:,0] > pts_r[:,0]
        good_matches = epipolar_is_ok & disparity_is_positive

        return np.asarray(matches)[good_matches]


    def _compute_disparities(self, keypoints_l, keypoints_r, matches):
        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]
        
        pts_l = np.array([k.pt for k in np.asarray(keypoints_l)[query_idx]])
        pts_r = np.array([k.pt for k in np.asarray(keypoints_r)[train_idx]])

        disparity = pts_l[:,0] - pts_r[:,0]
        self._point_disparities = list(zip(pts_l.astype(int), disparity))
