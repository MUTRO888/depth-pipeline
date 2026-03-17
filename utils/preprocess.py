import cv2
import numpy as np

def remove_specular_highlights(image: np.ndarray, v_threshold=220, s_threshold=50) -> np.ndarray:
    """
    Remove specular highlights using HSV thresholding and inpainting.
    The image should be a BGR numpy array (OpenCV format) or RGB.
    Assuming input is RGB (as used in Diffusers/PIL).
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # Check if image is RGB, convert to BGR for OpenCV
    # We'll assume the input is RGB because PIL outputs RGB
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Create mask for high brightness and low saturation (specular reflection characteristics)
    mask = cv2.inRange(hsv_img, np.array([0, 0, v_threshold]), np.array([179, s_threshold, 255]))
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Inpaint
    result_bgr = cv2.inpaint(bgr_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

def single_scale_retinex(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Single Scale Retinex (SSR)
    """
    # Convert to float to avoid overflow
    img_float = np.float64(image) + 1.0 # Add 1 to avoid log(0)
    
    # Gaussian blur to estimate illumination
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    
    # Retinex formula: log(R) = log(I) - log(L)
    retinex = np.log10(img_float) - np.log10(blurred + 1.0)
    return retinex

def multi_scale_retinex(image: np.ndarray, sigma_list=[15, 80, 250]) -> np.ndarray:
    """
    Multi Scale Retinex (MSR)
    Applies SSR with multiple sigmas and averages them.
    Expects and returns RGB numpy arrays.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    retinex_total = np.zeros_like(image, dtype=np.float64)
    for sigma in sigma_list:
        retinex_total += single_scale_retinex(image, sigma)
        
    retinex_total = retinex_total / len(sigma_list)
    return retinex_total

def msrcr(image: np.ndarray, sigma_list=[15, 80, 250], alpha=125, beta=46, G=192, b=-30) -> np.ndarray:
    """
    Multi Scale Retinex with Color Restoration (MSRCR)
    Keeps colors natural while normalizing illumination.
    """
    img_float = np.float64(image) + 1.0
    
    # MSR part
    msr = multi_scale_retinex(image, sigma_list)
    
    # Color Restoration part
    sum_dist = np.sum(img_float, axis=2, keepdims=True)
    color_restoration_func = np.log10(alpha * img_float / sum_dist)
    
    msrcr_val = msr * color_restoration_func
    
    # Linear stretch to 0-255
    msrcr_val = G * msrcr_val + b
    
    # Clip and convert to uint8
    msrcr_val = np.clip(msrcr_val, 0, 255).astype(np.uint8)
    return msrcr_val

def preprocess_for_depth(image_rgb: np.ndarray, config: dict) -> np.ndarray:
    """
    Full preprocessing pipeline to be fed into Marigold.
    Returns a processed RGB numpy array.
    """
    processed = image_rgb.copy()
    
    cfg = config.get('preprocess', {})
    
    # 1. Specular Highlight Removal
    if cfg.get('remove_specular', True):
        v_thresh = cfg.get('specular_v_threshold', 220)
        s_thresh = cfg.get('specular_s_threshold', 50)
        processed = remove_specular_highlights(processed, v_threshold=v_thresh, s_threshold=s_thresh)
        
    # 2. Illumination Normalization (MSRCR)
    if cfg.get('normalize_illumination', True):
        sigmas = cfg.get('retinex_sigmas', [15, 80, 250])
        processed = msrcr(processed, sigma_list=sigmas)
        
    return processed
