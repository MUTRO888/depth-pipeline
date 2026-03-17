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

def fuse_layers(depth_map, original_image, lora_relief=None, detail_strength=1.0, relief_strength=0.5, gaussian_radius=2):
    """
    Frequency-Separation Fusion for CNC Relief Carving.
    
    Layer 1 (Depth base): Marigold depth map provides global 3D structure.
    Layer 2 (Surface detail): Original image grayscale high-pass filter extracts
                              wrinkles, folds, facial features, clothing textures
                              directly from the photograph's luminance.
    Layer 3 (Micro-texture): Optional SD+LoRA overlay for additional surface grain.
    
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
    # Convert to grayscale and resize to match depth map dimensions
    gray = original_image.convert("L")
    if gray.size != (W, H):
        gray = gray.resize((W, H), Image.Resampling.LANCZOS)
    
    gray_arr = np.array(gray).astype(np.float64) / 255.0
    
    # Multi-scale high-pass extraction (3 octaves for maximum detail coverage)
    # Small sigma: captures fine details (hair strands, tiny wrinkles, pores)
    # Medium sigma: captures mid details (clothing folds, facial features)
    # Large sigma: captures broad surface undulations
    sigma_fine = max(H, W) * 0.005    # ~0.5% of image size
    sigma_medium = max(H, W) * 0.02   # ~2% of image size
    sigma_broad = max(H, W) * 0.05    # ~5% of image size
    
    highpass_fine = gray_arr - gaussian_filter(gray_arr, sigma=sigma_fine)
    highpass_medium = gray_arr - gaussian_filter(gray_arr, sigma=sigma_medium)
    highpass_broad = gray_arr - gaussian_filter(gray_arr, sigma=sigma_broad)
    
    # Combine all octaves with decreasing weight (fine detail is most important for relief)
    combined_detail = (highpass_fine * 0.5) + (highpass_medium * 0.35) + (highpass_broad * 0.15)
    
    # --- 3. SD LoRA micro-texture (if available) ---
    lora_detail = np.zeros((H, W))
    if lora_relief is not None:
        lora_pil = lora_relief
        if lora_pil.size != (W, H):
            lora_pil = lora_pil.resize((W, H), Image.Resampling.LANCZOS)
        lora_arr = np.array(lora_pil).astype(np.float64) / 255.0
        
        # High-pass the LoRA output: only keep its unique micro-textures
        lora_lowfreq = gaussian_filter(lora_arr, sigma=sigma_medium)
        lora_detail = (lora_arr - lora_lowfreq) * relief_strength
    
    # --- 4. ADDITIVE FUSION ---
    # Depth provides the 3D shape base.
    # High-pass detail from original image is ADDED on top (not averaged).
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
