import torch
import numpy as np
from diffusers import MarigoldNormalsPipeline
from utils.poisson import normal_to_height

class NormalEstimator:
    """Marigold-based monocular normal estimation with OOM fallback and Poisson Integration."""

    def __init__(self, model_name, torch_dtype="float16", device="cuda"):
        self.device = device
        dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        
        # Load the Marigold Normals Pipeline
        self.pipe = MarigoldNormalsPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    def estimate(self, image, ensemble_size=5, denoising_steps=10, status_callback=None):
        """
        Run normal estimation and convert to height map via Poisson Integration.
        Returns a 2D numpy array of height values [0, 1].
        """
        try:
            # Output is a PIL Image or numpy array representing normals (RGB = XYZ)
            output = self.pipe(
                image,
                num_inference_steps=denoising_steps,
                ensemble_size=ensemble_size,
            )
            
            # Predict returns a normalized standard Normal space in [-1, 1] mapped to [0, 1] image space
            # We need to extract the raw numpy array to accurately integrate
            predictions = output.prediction  # typically shape (1, 3, H, W) or (B, H, W, 3) 
            
            # Diffusers output format varying:
            if torch.is_tensor(predictions):
                p_np = predictions.cpu().numpy()
            else:
                p_np = np.array(predictions)
                
            # Check shape to adapt correctly
            if p_np.ndim == 4 and p_np.shape[1] == 3:
                # (1, 3, H, W)
                normals_map = p_np.squeeze(0).transpose(1, 2, 0)
            elif p_np.ndim == 4 and p_np.shape[-1] == 3:
                # (1, H, W, 3)
                normals_map = p_np.squeeze(0)
            elif p_np.ndim == 3 and p_np.shape[0] == 3:
                # (3, H, W)
                normals_map = p_np.transpose(1, 2, 0)
            elif p_np.ndim == 3 and p_np.shape[-1] == 3:
                # (H, W, 3)
                normals_map = p_np
            else:
                # Fallback forced reshape if possible
                normals_map = p_np.squeeze()
                if normals_map.ndim == 3 and normals_map.shape[0] == 3:
                    normals_map = normals_map.transpose(1, 2, 0)
            
            # Print for debug
            print(f"DEBUG: Processed normals_map shape: {normals_map.shape}")
            
            # Extract height via Poisson Equation
            if status_callback:
                status_callback("正在通过泊松方程重建微细节高度...")
                
            height_map = normal_to_height(normals_map)
            return height_map

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                e, torch.cuda.OutOfMemoryError
            ):
                raise

            torch.cuda.empty_cache()
            if status_callback:
                status_callback("法线估计显存不足，自动降低精度重试...")

            output = self.pipe(
                image,
                num_inference_steps=denoising_steps,
                ensemble_size=1,
            )
            
            predictions = output.prediction
            if torch.is_tensor(predictions):
                p_np = predictions.cpu().numpy()
            else:
                p_np = np.array(predictions)
                
            if p_np.ndim == 4 and p_np.shape[1] == 3:
                normals_map = p_np.squeeze(0).transpose(1, 2, 0)
            elif p_np.ndim == 4 and p_np.shape[-1] == 3:
                normals_map = p_np.squeeze(0)
            elif p_np.ndim == 3 and p_np.shape[0] == 3:
                normals_map = p_np.transpose(1, 2, 0)
            elif p_np.ndim == 3 and p_np.shape[-1] == 3:
                normals_map = p_np
            else:
                normals_map = p_np.squeeze()
                if normals_map.ndim == 3 and normals_map.shape[0] == 3:
                    normals_map = normals_map.transpose(1, 2, 0)
                    
            print(f"DEBUG fallback: Processed normals_map shape: {normals_map.shape}")
            
            if status_callback:
                status_callback("正在通过泊松方程重建微细节高度...")
                
            height_map = normal_to_height(normals_map)
            return height_map
            
    def unload(self):
        """Releases the model from VRAM."""
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
