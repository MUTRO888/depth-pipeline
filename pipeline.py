import yaml
from PIL import Image

from steps.depth_estimation import DepthEstimator
from steps.normal_estimation import NormalEstimator
from steps.relief_enhance import ReliefEnhancer
from utils.image_io import load_image
from utils.postprocess import process_depth, fuse_layers
from utils.civitai_downloader import download_civitai_lora
import torch


class DepthPipeline:
    """Orchestrates the full depth estimation pipeline."""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.depth_estimator = None
        self.normal_estimator = None
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
                self.config["sd_relief"]["lora_path"] = "" # Fallback to dual-fusion
                
        # Step 0: Load image
        image = load_image(image_path)
        original_size = image.size

        # --- Stage 1: Depth Estimation ---
        update("正在加载深度模型...")
        cfg = self.config["model"]
        self.depth_estimator = DepthEstimator(
            model_name=cfg["name"],
            torch_dtype=cfg["torch_dtype"],
            device=cfg["device"],
        )
        
        update("Stage 1: 正在分析全局深度...")
        inf_cfg = self.config["inference"]
        depth_arr = self.depth_estimator.estimate(
            image,
            ensemble_size=inf_cfg["ensemble_size"],
            denoising_steps=inf_cfg["denoising_steps"],
            status_callback=status_callback,
        )
        
        # Free VRAM immediately
        self.depth_estimator.unload()
        self.depth_estimator = None
        
        # --- Stage 2: Normal Estimation + Poisson ---
        update("正在加载法线模型...")
        n_cfg = self.config["normals_model"]
        self.normal_estimator = NormalEstimator(
            model_name=n_cfg["name"],
            torch_dtype=n_cfg["torch_dtype"],
            device=n_cfg["device"],
        )
        
        update("Stage 2: 正在提取法线表面细节...")
        normal_height_arr = self.normal_estimator.estimate(
            image,
            ensemble_size=inf_cfg["ensemble_size"],
            denoising_steps=inf_cfg["denoising_steps"],
            status_callback=status_callback,
        )
        
        # Free VRAM
        self.normal_estimator.unload()
        self.normal_estimator = None

        # --- Stage 3: SD + LoRA Micro-Texture (Optional) ---
        lora_relief = None
        if self.config.get("sd_relief", {}).get("lora_path"):
            update("正在加载纹理增强模型 (SD+LoRA)...")
            self.relief_enhancer = ReliefEnhancer(self.config["sd_relief"])
            
            # Need depth map as an image for ControlNet
            # First normalize depth linearly inside [0, 255] just for ControlNet
            d_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)
            d_img = Image.fromarray((d_norm * 255).astype("uint8"), mode="L")
            
            update("Stage 3: 正在渲染表面微纹理...")
            lora_relief = self.relief_enhancer.enhance(image, d_img, status_callback=status_callback)
            
            # Free VRAM
            self.relief_enhancer.unload()
            self.relief_enhancer = None

        # --- Fusion & Postprocess ---
        update("正在融合多层细节生成灰度图...")
        f_cfg = self.config.get("fusion", {})
        weights = (
            f_cfg.get("depth_weight", 0.4),
            f_cfg.get("normal_weight", 0.45),
            f_cfg.get("relief_weight", 0.15) if lora_relief else 0.0
        )
        
        radius = self.config["postprocess"]["gaussian_radius"]
        result = fuse_layers(
            depth_map=depth_arr,
            normal_height=normal_height_arr,
            lora_relief=lora_relief,
            weights=weights,
            gaussian_radius=radius
        )

        # Ensure output matches input resolution
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        update("处理完成")
        return image, result
