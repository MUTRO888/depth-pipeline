import torch
from diffusers import MarigoldDepthPipeline


class DepthEstimator:
    """Marigold-based monocular depth estimation with OOM fallback."""

    def __init__(self, model_name, torch_dtype="float16", device="cuda"):
        self.device = device
        dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        self.pipe = MarigoldDepthPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    def estimate(self, image, ensemble_size=5, denoising_steps=10, status_callback=None):
        """
        Run depth estimation on a PIL Image.
        On OOM, automatically retries with ensemble_size=1.
        Returns a 2D numpy array of depth values.
        """
        try:
            output = self.pipe(
                image,
                num_inference_steps=denoising_steps,
                ensemble_size=ensemble_size,
            )
            return output.prediction.squeeze()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                e, torch.cuda.OutOfMemoryError
            ):
                raise

            torch.cuda.empty_cache()
            if status_callback:
                status_callback("显存不足，已自动降低处理精度重试")

            output = self.pipe(
                image,
                num_inference_steps=denoising_steps,
                ensemble_size=1,
            )
            return output.prediction.squeeze()
            
    def unload(self):
        """Releases the model from VRAM."""
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
