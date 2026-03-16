import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def process_depth(depth_map, gaussian_radius=2):
    """
    Post-process a raw depth map into an 8-bit grayscale image for relief carving.

    Steps:
        1. Normalize depth values to 0-255
        2. Invert depth direction (white = high point, black = low point)
        3. Gaussian smooth to reduce noise
        4. Return as 8-bit grayscale PIL Image
    """
    depth = depth_map.astype(np.float64)

    # Normalize to 0-255
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        normalized = (depth - d_min) / (d_max - d_min) * 255.0
    else:
        normalized = np.zeros_like(depth)

    # Invert: closer objects (small depth) become white (high point)
    inverted = 255.0 - normalized

    # Gaussian smooth
    smoothed = gaussian_filter(inverted, sigma=gaussian_radius)

    # Clip and convert to uint8
    result = np.clip(smoothed, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode="L")
