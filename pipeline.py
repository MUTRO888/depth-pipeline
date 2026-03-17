import yaml
from PIL import Image
import numpy as np

from steps.depth_estimation import DepthEstimator
from steps.relief_enhance import ReliefEnhancer
from utils.image_io import load_image
from utils.postprocess import fuse_layers
from utils.civitai_downloader import download_civitai_lora
from utils.preprocess import preprocess_for_depth
import torch


class DepthPipeline:
    """Orchestrates the full depth estimation pipeline."""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.depth_estimator = None
        self.relief_enhancer = None

    def process(self, image_path, status_callback=None):
        """
        Run the full pipeline on an image file.

        Args:
            image_path: Path to the input JPG/PNG image.
            status_callback: Optional callable(str) for status updates.

        Returns:
            (original_image, depth_image) - both PIL Images.
        """

        def update(msg):
            if status_callback:
                status_callback(msg)

        # Download LoRA if needed
        lora_cfg = self.config.get("sd_relief", {})
        if lora_cfg.get("lora_path") == "auto":
            lora_path = download_civitai_lora(status_callback=status_callback)
            if lora_path:
                self.config["sd_relief"]["lora_path"] = lora_path
            else:
                self.config["sd_relief"]["lora_path"] = ""
                
        # Step 0: Load image
        image = load_image(image_path)
        original_size = image.size

        # --- V4 Preprocessing (CPU) ---
        update("Stage 1a/2: 正在进行图像几何纠偏预处理(去高光+光照均衡)...")
        if "preprocess" not in self.config:
            self.config["preprocess"] = {"remove_specular": True, "normalize_illumination": True}
        
        image_np = np.array(image.convert("RGB"))
        preprocessed_np = preprocess_for_depth(image_np, self.config)
        preprocessed_image = Image.fromarray(preprocessed_np)

        # --- Stage 1: Depth Estimation (GPU) ---
        update("Stage 1b/2: 正在分析全局深度...")
        cfg = self.config["model"]
        self.depth_estimator = DepthEstimator(
            model_name=cfg["name"],
            torch_dtype=cfg["torch_dtype"],
            device=cfg["device"],
        )
        
        inf_cfg = self.config["inference"]
        # Feeding PREPROCESSED image to Marigold
        depth_arr = self.depth_estimator.estimate(
            preprocessed_image,
            ensemble_size=inf_cfg["ensemble_size"],
            denoising_steps=inf_cfg["denoising_steps"],
            status_callback=status_callback,
        )
        
        # Free VRAM immediately
        self.depth_estimator.unload()
        self.depth_estimator = None

        # --- Stage 2: SD + LoRA Micro-Texture (Optional, GPU) ---
        lora_relief = None
        if self.config.get("sd_relief", {}).get("lora_path"):
            update("正在加载纹理增强模型 (SD+LoRA)...")
            self.relief_enhancer = ReliefEnhancer(self.config["sd_relief"])
            
            # Depth map as image for ControlNet conditioning
            d_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)
            d_img = Image.fromarray((d_norm * 255).astype("uint8"), mode="L")
            
            update("Stage 2/2: 正在渲染表面微纹理...")
            lora_relief = self.relief_enhancer.enhance(image, d_img, status_callback=status_callback)
            
            # Free VRAM
            self.relief_enhancer.unload()
            self.relief_enhancer = None

        # --- Fusion & Postprocess (CPU, zero GPU) ---
        update("正在融合原图细节到深度图...")
        f_cfg = self.config.get("fusion", {})
        detail_strength = f_cfg.get("detail_strength", 1.5)
        relief_strength = f_cfg.get("relief_strength", 0.5)
        radius = self.config["postprocess"]["gaussian_radius"]
        
        result = fuse_layers(
            depth_map=depth_arr,
            original_image=image,
            lora_relief=lora_relief,
            detail_strength=detail_strength,
            relief_strength=relief_strength,
            gaussian_radius=radius,
        )

        # Ensure output matches input resolution
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        update("处理完成")
        return image, result

