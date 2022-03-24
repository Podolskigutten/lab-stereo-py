# Step 4: Dense stereo processing

## 6. Dense stereo processing
Go to `DenseStereoMatcher` in [lab_stereo.py](../lab_stereo.py).
Your task is again to compute the correct `depths` in meters:

1. In the constructor, compute the minimum and maximum depths the matcher can compute from the given minimum and maximum disparities.
2. In `DenseStereoMatcher.match()`, compute the correct depth given the disparity results.

## Further experiments
- Play around with the parameters for the dense stereo matcher.
  See the [cv::StereoSGBM Class Reference](https://docs.opencv.org/4.5.5/d2/d85/classcv_1_1StereoSGBM.html#adb7a50ef5f200ad9559e9b0e976cfa59).
- Activate the projector. 
  How does it influence the stereo processing?
- Compute 3D points for the dense result and show them instead of the sparse point cloud!
