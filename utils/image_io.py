import os
from PIL import Image


def load_image(path):
    """Load and validate an image file. Returns RGB PIL Image."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise ValueError("请上传 JPG 或 PNG 格式的图片")

    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        raise ValueError("文件读取失败，请重新选择")


def save_bmp(image, path):
    """Save image as 8-bit grayscale BMP."""
    if image.mode != "L":
        image = image.convert("L")

    dir_path = os.path.dirname(path) or "."
    if not os.access(dir_path, os.W_OK):
        raise PermissionError("无法写入该目录，请选择其他位置")

    image.save(path, format="BMP")
