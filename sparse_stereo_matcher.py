import cv2
import numpy as np
from common_lab_utils import (Size, StereoPair)
from anms import anms


class SparseStereoMatcher:
    def __init__(self, detector, desc_extractor):
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._matcher = cv2.BFMatcher_create(self._desc_extractor.defaultNorm())

    def match(self, stereo_pair: StereoPair, use_anms=False):
        """Detect and match keypoints in both images."""

        # Detect and describe features in the left image.
        keypoints_left = self._detector.detect(stereo_pair.left)
        if use_anms:
            img_size = Size.from_numpy_shape(stereo_pair.left.shape)
            keypoints_left = self._adaptive_non_maximal_suppression(keypoints_left, img_size)
        keypoints_left, query_descriptors = self._desc_extractor.compute(stereo_pair.left, keypoints_left)

        # Detect and describe features in the right image.
        keypoints_right = self._detector.detect(stereo_pair.right)
        if use_anms:
            img_size = Size.from_numpy_shape(stereo_pair.right.shape)
            keypoints_right = self._adaptive_non_maximal_suppression(keypoints_right, img_size)
        keypoints_right, train_descriptors = self._desc_extractor.compute(stereo_pair.right, keypoints_right)

        if not keypoints_left or not keypoints_right:
            return StereoMatchingResult()

        # Match features
        matches = self._matcher.match(query_descriptors, train_descriptors)

        # Extract good matches.
        good_matches = self._extract_good_matches(keypoints_left, keypoints_right, matches, epipolar_limit=1.5)

        # Compute disparites.
        disparities = self._compute_disparities(keypoints_left, keypoints_right, good_matches)

        return StereoMatchingResult(keypoints_left, keypoints_right, good_matches, disparities)

    @staticmethod
    def _adaptive_non_maximal_suppression(keypoints, img_size: Size, max_num=1000, max_ratio=0.7, tolerance=0.1):
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)

        num_to_retain = min(max_num, round(max_ratio * len(keypoints)))
        return anms.ssc(keypoints, num_to_retain, tolerance, img_size.width, img_size.height)

    @staticmethod
    def _extract_good_matches(keypoints_l, keypoints_r, matches, epipolar_limit):
        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]
        
        pts_l = np.array([k.pt for k in np.asarray(keypoints_l)[query_idx]])
        pts_r = np.array([k.pt for k in np.asarray(keypoints_r)[train_idx]])
        
        epipolar_is_ok = abs(pts_l[:, 1] - pts_r[:, 1]) < epipolar_limit
        disparity_is_positive = pts_l[:, 0] > pts_r[:, 0]
        good_matches = epipolar_is_ok & disparity_is_positive

        return np.asarray(matches)[good_matches]

    @staticmethod
    def _compute_disparities(keypoints_l, keypoints_r, matches):
        if len(matches) <= 0:
            return []

        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]
        
        pts_l = np.array([k.pt for k in np.asarray(keypoints_l)[query_idx]])
        pts_r = np.array([k.pt for k in np.asarray(keypoints_r)[train_idx]])

        disparity = pts_l[:, 0] - pts_r[:, 0]
        return list(zip(pts_l.astype(int), disparity))


class StereoMatchingResult:
    def __init__(self, keypoints_left=(), keypoints_right=(), matches=(), point_disparities=()):
        self._keypoints_left = keypoints_left
        self._keypoints_right = keypoints_right
        self._matches = matches
        self._point_disparities = point_disparities

    @property
    def keypoints_left(self):
        return self._keypoints_left

    @property
    def keypoints_right(self):
        return self._keypoints_right

    @property
    def matches(self):
        return self._matches

    @property
    def point_disparities(self):
        return self._point_disparities
