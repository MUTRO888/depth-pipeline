import os

import torch
from diffusers import MarigoldDepthPipeline

from utils.model_source import resolve_model_source


class DepthEstimator:
    """Marigold-based monocular depth estimation with OOM fallback."""

    def __init__(
        self,
        model_config,
        project_dir,
        torch_dtype="float16",
        device="cuda",
        offline=False,
    ):
        self.device = device
        dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        model_source = resolve_model_source(
            model_config,
            project_dir=project_dir,
            path_keys=("local_path", "path"),
            repo_keys=("name", "repo_id"),
            fallback_keys=("name",),
            offline=offline,
            label="深度模型",
        )
        local_files_only = os.path.isdir(model_source)
        load_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
        }
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
        self.pipe = MarigoldDepthPipeline.from_pretrained(
            model_source,
            **load_kwargs,
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
