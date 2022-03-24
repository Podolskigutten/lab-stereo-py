import numpy as np
from dataclasses import dataclass
import cv2


@dataclass
class Colours:
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


@dataclass
class Font:
    face = cv2.FONT_HERSHEY_PLAIN
    scale = 1.0


def visualize_matches(stereo_pair, stereo_matcher, duration_rectification, duration_matching):
    """ This function will create an image that shows corresponding keypoints in two images."""

    if stereo_matcher.matches is None or not stereo_matcher.keypoints_left or not stereo_matcher.keypoints_right:
        return np.hstack((stereo_pair.left, stereo_pair.right))

    cv2.putText(stereo_pair.left, "LEFT", (10, 20), Font.face, Font.scale, Colours.black)
    cv2.putText(stereo_pair.right, "RIGHT", (10, 20), Font.face, Font.scale, Colours.black)
    vis_img = cv2.drawMatches(
        stereo_pair.left, stereo_matcher.keypoints_left,
        stereo_pair.right, stereo_matcher.keypoints_right,
        stereo_matcher.matches, None, flags=2)
    cv2.putText(vis_img, f"Rectification:  {round(duration_rectification)} ms", (10, 40), Font.face, Font.scale, Colours.red)
    cv2.putText(vis_img, f"Matching:  {round(duration_matching)} ms", (10, 60), Font.face, Font.scale, Colours.red)
    cv2.putText(vis_img, f"Number of matches:  {len(stereo_matcher.matches)}", (10, 80), Font.face, Font.scale, Colours.red)
    return vis_img


def visualize_depths(stereo_pair, pts_left, depths):
    """"""

    vis_depth = cv2.cvtColor(stereo_pair.left, cv2.COLOR_GRAY2BGR)

    marker_size = 5
    for pt, depth in zip(pts_left, depths):
        px = pt.astype(int)
        cv2.drawMarker(vis_depth, px, Colours.green, cv2.MARKER_CROSS, marker_size)
        cv2.putText(vis_depth, f"{depth:.2f}", px, Font.face, Font.scale, Colours.green)

    return vis_depth


def visualise_dense(dense_depth: np.ndarray, min_depth, max_depth, duration_dense):
    vis_dense_depth = dense_depth / max_depth
    vis_dense_depth = cv2.cvtColor(vis_dense_depth * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    vis_dense_depth = cv2.applyColorMap(vis_dense_depth, cv2.COLORMAP_JET)

    cv2.putText(vis_dense_depth, f"Dense:  {round(duration_dense)} ms", (10, 20), Font.face, Font.scale, Colours.white)
    cv2.putText(vis_dense_depth, f"Depth range:  [{min_depth:.2f}, {max_depth:.2f}]", (10, 40), Font.face, Font.scale, Colours.white)

    return vis_dense_depth
