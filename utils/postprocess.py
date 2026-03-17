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

def fuse_layers(depth_map, normal_height, lora_relief=None, weights=(0.4, 0.45, 0.15), gaussian_radius=2):
    """
    Combines layers from the triple-fusion pipeline.
    
    Args:
        depth_map: base depth from Marigold (numpy array or PIL Image)
        normal_height: height map from Poisson integration of normals (numpy array)
        lora_relief: micro-texture map from SD (PIL Image)
        weights: tuple of (alpha, beta, gamma)
        
    Returns:
        PIL Image: 8-bit grayscale BMP
    """
    # 1. Normalize and structure Depth Base
    if isinstance(depth_map, Image.Image):
        depth_arr = np.array(depth_map).astype(np.float64)
    else:
        depth_arr = depth_map.astype(np.float64)
        
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_arr)
        
    # Invert baseline: normally closer objects (high depth) are white (1.0)
    # The existing process_depth function inverted things so big depth value = white
    # Let's ensure everything is consistent: 1.0 = Highest Point (White)
    
    # 2. Normalize and structure Normal Height
    normal_norm = normal_height.astype(np.float64)
    n_min, n_max = normal_norm.min(), normal_norm.max()
    if n_max > n_min:
        normal_norm = (normal_norm - n_min) / (n_max - n_min)
    else:
        normal_norm = np.zeros_like(normal_norm)
        
    # 3. Handle LoRA Relief if it exists
    if lora_relief is not None:
        lora_arr = np.array(lora_relief).astype(np.float64)
        # Note: LoRA usually outputs actual images, so lighter = higher
        lora_norm = lora_arr / 255.0
        
        w_d, w_n, w_l = weights
        total_w = w_d + w_n + w_l
        w_d, w_n, w_l = w_d/total_w, w_n/total_w, w_l/total_w
        
        fused = (depth_norm * w_d) + (normal_norm * w_n) + (lora_norm * w_l)
    else:
        w_d, w_n = weights[0], weights[1]
        total_w = w_d + w_n
        w_d, w_n = w_d/total_w, w_n/total_w
        
        fused = (depth_norm * w_d) + (normal_norm * w_n)

    # 4. Scale to 0-255 
    # Since we want white = high point, and our normalized inputs already follow
    # higher value = closer, we just scale.
    fused_255 = fused * 255.0
    
    # 5. Gaussian smooth
    smoothed = gaussian_filter(fused_255, sigma=gaussian_radius)
    
    result = np.clip(smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")
