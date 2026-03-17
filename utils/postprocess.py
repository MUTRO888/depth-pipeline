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
    Frequency-Separation Fusion for CNC Relief Carving.
    
    Strategy (fundamentally different from weighted averaging):
      - Depth layer = LOW-FREQUENCY BASE (overall 3D shape)
      - Normal layer = HIGH-FREQUENCY DETAIL EXTRACTION (wrinkles, folds, facial features)
      - SD LoRA layer = MICRO-TEXTURE OVERLAY (hair strands, fabric weave)
    
    The Normal and SD layers are high-pass filtered to extract ONLY the details
    that the Depth layer is missing, then additively layered on top.
    """
    # --- 1. Prepare Depth Base ---
    if isinstance(depth_map, Image.Image):
        depth_arr = np.array(depth_map).astype(np.float64)
    else:
        depth_arr = depth_map.astype(np.float64)
    
    H_d, W_d = depth_arr.shape[:2]
        
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_arr)
    
    # Invert: Marigold depth has small value = near. We want white(1.0) = near = high point.
    depth_norm = 1.0 - depth_norm
    
    # --- 2. Prepare Normal Height and extract its UNIQUE details ---
    normal_arr = normal_height.astype(np.float64)
    
    # Resize normal to match depth if dimensions differ
    if normal_arr.shape[:2] != (H_d, W_d):
        from PIL import Image as PILImage
        normal_pil = PILImage.fromarray(
            ((normal_arr - normal_arr.min()) / (normal_arr.max() - normal_arr.min() + 1e-8) * 255).astype(np.uint8), 
            mode="L"
        )
        normal_pil = normal_pil.resize((W_d, H_d), PILImage.Resampling.LANCZOS)
        normal_arr = np.array(normal_pil).astype(np.float64) / 255.0
    else:
        n_min, n_max = normal_arr.min(), normal_arr.max()
        if n_max > n_min:
            normal_arr = (normal_arr - n_min) / (n_max - n_min)
        else:
            normal_arr = np.zeros_like(normal_arr)
    
    # HIGH-PASS FILTER: Extract details that exist in Normal but NOT in Depth.
    # Large sigma = only the broadest shapes survive the blur.
    # Subtracting the blur from the original gives us pure surface detail.
    normal_blur_sigma = max(H_d, W_d) * 0.02  # ~2% of image size, adaptive
    normal_lowfreq = gaussian_filter(normal_arr, sigma=normal_blur_sigma)
    normal_detail = normal_arr - normal_lowfreq  # Pure high-frequency: wrinkles, edges, folds
    
    # The detail_strength controls how aggressively we carve surface details.
    # weights[1] (normal_weight) now controls the INTENSITY of detail injection.
    # Scale from 0~1 slider to a meaningful multiplier (0 = no detail, 1.0 = 3x amplification)
    detail_strength = weights[1] * 3.0
    
    # --- 3. SD LoRA micro-texture (if available) ---
    lora_detail = np.zeros_like(depth_norm)
    if lora_relief is not None:
        lora_arr = np.array(lora_relief).astype(np.float64)
        if lora_arr.shape[:2] != (H_d, W_d):
            lora_pil = Image.fromarray(lora_arr.astype(np.uint8), mode="L")
            lora_pil = lora_pil.resize((W_d, H_d), Image.Resampling.LANCZOS)
            lora_arr = np.array(lora_pil).astype(np.float64)
        lora_norm = lora_arr / 255.0
        
        # High-pass the LoRA output too: we only want its unique micro-textures
        lora_lowfreq = gaussian_filter(lora_norm, sigma=normal_blur_sigma * 0.5)
        lora_detail = lora_norm - lora_lowfreq
        
        lora_strength = weights[2] * 3.0
        lora_detail = lora_detail * lora_strength
    
    # --- 4. ADDITIVE FUSION ---
    # Base shape comes 100% from Depth (scaled by its weight for overall relief height)
    # Details are ADDED on top, not averaged in.
    base = depth_norm * weights[0] * 2.5  # Scale base to use more of 0~1 range
    
    fused = base + (normal_detail * detail_strength) + lora_detail
    
    # --- 5. Auto-stretch to full 0~255 range ---
    f_min, f_max = fused.min(), fused.max()
    if f_max > f_min:
        fused = (fused - f_min) / (f_max - f_min)
    
    fused_255 = fused * 255.0
    
    # --- 6. Multi-scale detail boost (Unsharp Mask at two scales) ---
    # Fine scale: hair, tiny wrinkles
    blur_fine = gaussian_filter(fused_255, sigma=1.0)
    detail_fine = fused_255 - blur_fine
    
    # Medium scale: clothing folds, facial contours
    blur_medium = gaussian_filter(fused_255, sigma=5.0)
    detail_medium = fused_255 - blur_medium
    
    # Boost both scales
    boosted = fused_255 + (detail_fine * 1.0) + (detail_medium * 0.5)
    
    # --- 7. Minimal smoothing (just kill single-pixel noise) ---
    smoothed = gaussian_filter(boosted, sigma=0.5)
    
    result = np.clip(smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")
