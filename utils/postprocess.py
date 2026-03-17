import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter

def guided_filter(guide, src, radius, epsilon):
    """
    Pure NumPy implementation of Guided Filter.
    guide: 2D numpy array [0,1]
    src: 2D numpy array [0,1]
    radius: int
    epsilon: float
    """
    from scipy.ndimage import uniform_filter
    
    I = guide.astype(np.float64)
    p = src.astype(np.float64)
    
    size = int(2 * radius + 1)
    
    N = uniform_filter(np.ones_like(I), size=size)
    
    mean_I = uniform_filter(I, size=size) / N
    mean_p = uniform_filter(p, size=size) / N
    mean_Ip = uniform_filter(I * p, size=size) / N
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = uniform_filter(I * I, size=size) / N
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=size) / N
    mean_b = uniform_filter(b, size=size) / N
    
    q = mean_a * I + mean_b
    return q


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

def fuse_layers(depth_map, original_image, lora_relief=None, detail_strength=1.0, relief_strength=0.5, gaussian_radius=2, guided_radius=None, guided_eps=1e-3):
    """
    Frequency-Separation Fusion for CNC Relief Carving (V4 Hybrid).
    
    Layer 1 (Depth base): Marigold depth map provides global 3D structure.
    Layer 2 (Surface detail): Guided Filter extracts crisp micro-texture directly from Original Image,
                              rejecting lighting artifacts inconsistent with depth.
    Layer 3 (Micro-texture): Optional SD+LoRA overlay.
    
    Args:
        depth_map: raw depth from Marigold (numpy array, float)
        original_image: original input PIL Image (RGB)
        lora_relief: optional micro-texture map from SD (PIL Image, grayscale)
        detail_strength: multiplier for high-pass detail injection (0~3.0)
        relief_strength: multiplier for SD LoRA texture injection (0~3.0)
        gaussian_radius: final smoothing radius
    """
    # --- 1. Prepare Depth Base ---
    if isinstance(depth_map, Image.Image):
        depth_arr = np.array(depth_map).astype(np.float64)
    else:
        depth_arr = depth_map.astype(np.float64)
    
    H, W = depth_arr.shape[:2]
    
    # Normalize depth to [0, 1]
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_arr)
    
    # Invert: Marigold small value = near. We want white(1.0) = near = high point.
    depth_norm = 1.0 - depth_norm
    
    # --- 2. Extract surface detail from ORIGINAL IMAGE ---
    gray = original_image.convert("L")
    if gray.size != (W, H):
        gray = gray.resize((W, H), Image.Resampling.LANCZOS)
    
    gray_arr = np.array(gray).astype(np.float64) / 255.0
    
    # Depth-weighted masking
    detail_mask = depth_norm  # 1.0 on raised surfaces, 0.0 on background
    
    # V4: Use Guided Filter instead of naive Gaussian highpass
    # The depth map acts as the guide. The guided filter preserves the structure of depth_norm
    # while fitting the intensities of gray_arr.
    # The residual reveals pure textures that don't belong to the gross geometry.
    if guided_radius is None:
        guided_radius = max(8, int(max(H, W) * 0.015)) # ~1.5% of max dimension
        
    gf_base = guided_filter(guide=depth_norm, src=gray_arr, radius=guided_radius, epsilon=guided_eps)
    detail = gray_arr - gf_base
    
    combined_detail = detail * detail_mask
    
    # --- 3. SD LoRA micro-texture (if available) ---
    lora_detail = np.zeros((H, W))
    if lora_relief is not None:
        lora_pil = lora_relief
        if lora_pil.size != (W, H):
            lora_pil = lora_pil.resize((W, H), Image.Resampling.LANCZOS)
        lora_arr = np.array(lora_pil).astype(np.float64) / 255.0
        
        lora_lowfreq = gaussian_filter(lora_arr, sigma=max(H, W) * 0.012)
        lora_detail = (lora_arr - lora_lowfreq) * relief_strength * detail_mask
    
    # --- 4. ADDITIVE FUSION ---
    # Depth provides the correct 3D shape. Details are overlaid as surface texture only.
    fused = depth_norm + (combined_detail * detail_strength) + lora_detail
    
    # --- 5. Auto-stretch to full 0~255 range ---
    f_min, f_max = fused.min(), fused.max()
    if f_max > f_min:
        fused = (fused - f_min) / (f_max - f_min)
    
    fused_255 = fused * 255.0
    
    # --- 6. Minimal smoothing (just kill single-pixel noise, preserve ALL detail) ---
    smoothed = gaussian_filter(fused_255, sigma=max(0.3, gaussian_radius * 0.25))
    
    result = np.clip(smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")
