# motion_farneback.py (Optimized for full-frame pipeline)
import cv2
import numpy as np

def compute_farneback_motion(prev_gray, curr_gray):
    """
    Computes mean optical flow magnitude between two grayscale frames.
    prev_gray, curr_gray: uint8 GRAY images, same size.

    Returns:
        float motion_value
    """

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,   # pyr_scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0      # flags
    )

    # magnitude only (direction not needed for your report)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # return average movement
    return float(np.mean(mag))
