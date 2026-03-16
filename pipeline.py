import yaml
from PIL import Image

from steps.depth_estimation import DepthEstimator
from utils.image_io import load_image
from utils.postprocess import process_depth


class DepthPipeline:
    """Orchestrates the full depth estimation pipeline."""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.estimator = None

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

        # Step 1: Load model (lazy, first run only)
        if self.estimator is None:
            update("正在加载模型...")
            cfg = self.config["model"]
            self.estimator = DepthEstimator(
                model_name=cfg["name"],
                torch_dtype=cfg["torch_dtype"],
                device=cfg["device"],
            )
        else:
            update("模型已就绪")

        # Step 2: Load image
        image = load_image(image_path)
        original_size = image.size

        # Step 3: Depth estimation
        update("正在分析图片深度，请稍候")
        inf_cfg = self.config["inference"]
        depth_map = self.estimator.estimate(
            image,
            ensemble_size=inf_cfg["ensemble_size"],
            denoising_steps=inf_cfg["denoising_steps"],
            status_callback=status_callback,
        )

        # Step 4: Post-process
        update("正在生成灰度图...")
        radius = self.config["postprocess"]["gaussian_radius"]
        result = process_depth(depth_map, gaussian_radius=radius)

        # Ensure output matches input resolution
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        update("处理完成")
        return image, result
