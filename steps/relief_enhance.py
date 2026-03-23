import os

import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

from utils.model_source import resolve_model_source, resolve_path
from utils.device import resolve_device, resolve_dtype, empty_cache, is_oom_error, generator_device

class ReliefEnhancer:
    """Uses SD 1.5 with ControlNet Depth and Relief LoRA to add surface micro-textures."""

    def __init__(self, config, project_dir, device="auto", offline=False):
        self.config = config
        self.project_dir = project_dir
        self.device = resolve_device(device)
        self.dtype = resolve_dtype("auto", self.device)
        self.offline = offline
        self.pipe = None

    def _load_model(self):
        """Lazy load to save memory when not in use."""
        if self.pipe is not None:
            return

        dtype = self.dtype
        controlnet_source = resolve_model_source(
            self.config,
            project_dir=self.project_dir,
            path_keys=("controlnet_local_path", "controlnet_path"),
            repo_keys=("controlnet_model_id", "controlnet"),
            fallback_keys=("controlnet_model_id",),
            offline=self.offline,
            label="ControlNet 模型",
        )
        base_model_source = resolve_model_source(
            self.config,
            project_dir=self.project_dir,
            path_keys=("base_model_local_path", "base_model_path"),
            repo_keys=("base_model", "model_id"),
            fallback_keys=("model_id",),
            offline=self.offline,
            label="SD 基础模型",
        )
        local_controlnet = os.path.isdir(controlnet_source)
        local_base_model = os.path.isdir(base_model_source)
        shared_load_kwargs = {}
        if dtype == torch.float16:
            shared_load_kwargs["variant"] = "fp16"
            shared_load_kwargs["use_safetensors"] = True

        # 1. Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_source,
            torch_dtype=dtype,
            local_files_only=local_controlnet,
            **shared_load_kwargs,
        )

        # 2. Load Pipeline
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_source,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=local_base_model,
            **shared_load_kwargs,
        ).to(self.device)

        # Use UniPC for faster inference
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # 3. Load LoRA if provided
        lora_path = self.config.get("lora_path")
        if lora_path and lora_path != "auto":
            try:
                lora_path = resolve_path(self.project_dir, lora_path) or lora_path
                # We expect a safetensors file
                self.pipe.load_lora_weights(lora_path)
            except Exception as e:
                print(f"Warning: Failed to load LoRA {lora_path}: {e}")

    def enhance(self, original_image, depth_map_pil, status_callback=None):
        """
        Runs the enhancement process.
        Args:
            original_image (PIL.Image): The original RGB image.
            depth_map_pil (PIL.Image): The depth map (grayscale) produced by Marigold.
        Returns:
            PIL.Image (L): The enhanced structure as a grayscale map.
        """
        self._load_model()

        if status_callback:
            status_callback("正在加载纹理生成模型...")

        # SD 1.5 prefers 512x512 multiples. We should resize keeping aspect ratio roughly.
        w, h = original_image.size
        # A simple resizing strategy ensuring multiples of 8, max dimension ~768 to save VRAM
        max_dim = 768
        scale = min(max_dim / w, max_dim / h)
        if scale < 1.0:
            new_w = int((w * scale) // 8 * 8)
            new_h = int((h * scale) // 8 * 8)
            proc_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            proc_depth = depth_map_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w = int(w // 8 * 8)
            new_h = int(h // 8 * 8)
            proc_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            proc_depth = depth_map_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # We need an initial image for Img2Img. To strongly enforce depth,
        # we can pass the depth map as the initial image, or the original image
        # converted to grayscale. Passing original image lets SD see some contrast details.
        init_img = proc_image.convert("RGB")
        control_img = proc_depth.convert("RGB")

        # Read parameters
        prompt = self.config.get("prompt", "BAS-RELIEF, grayscale 3D relief, desaturated")
        n_prompt = self.config.get("negative_prompt", "color, colorful")
        strength = self.config.get("strength", 0.45)
        guidance = self.config.get("guidance_scale", 7.5)
        steps = self.config.get("num_inference_steps", 30)

        if status_callback:
            status_callback("正在渲染表面微纹理 (LoRA)...")

        # MPS does not support on-device Generator; use CPU generator instead
        gen_dev = generator_device(self.device)
        generator = torch.Generator(device=gen_dev).manual_seed(42)

        try:
            output = self.pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                image=init_img,
                control_image=control_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator,
                cross_attention_kwargs={
                    "scale": self.config.get("lora_scale", self.config.get("lora_weight", 0.8))
                },
            )

            result_rgb = output.images[0]
            # Convert to grayscale
            result_l = result_rgb.convert("L")

            # Resize back to true original dimensions
            if result_l.size != (w, h):
                result_l = result_l.resize((w, h), Image.Resampling.LANCZOS)

            return result_l

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not is_oom_error(e):
                raise
            if status_callback:
                status_callback("执行 LoRA 增强时内存不足，已跳过此步骤")
            empty_cache(self.device)
            return None

    def unload(self):
        """Releases the model from GPU/accelerator memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        empty_cache(self.device)
