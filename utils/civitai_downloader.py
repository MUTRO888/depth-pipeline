import os
import urllib.request
import traceback

def download_civitai_lora(model_version_id="449471", save_dir="models/lora", status_callback=None):
    """
    Downloads a LoRA from Civitai given its version ID.
    Default ID is "449471" which corresponds to "Depth map Lora - SD1.5 - v1.0"
    (A highly rated LoRA for grayscale depth map generation).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Civitai download API endpoint
    download_url = f"https://civitai.com/api/download/models/{model_version_id}"
    save_path = os.path.join(save_dir, f"relief_lora_{model_version_id}.safetensors")
    
    if os.path.exists(save_path):
        if status_callback:
            status_callback("已找到浮雕 LoRA 模型，跳过下载")
        return save_path
        
    if status_callback:
        status_callback("首次运行：正在自动下载最佳浮雕 LoRA 模型...（约 144MB）")
    
    try:
        # We don't have tqdm in requirements list currently. We can either add it, 
        # or just use a simple reporthook. Doing simple reporthook to minimize dependencies.
        def reporthook(count, block_size, total_size):
            if total_size > 0 and status_callback:
                percent = int(count * block_size * 100 / total_size)
                # print updates every 10% to avoid flooding the UI
                if percent % 10 == 0:
                    status_callback(f"正在下载浮雕 LoRA 模型... {percent}%")
                    
        req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlretrieve(download_url, filename=save_path, reporthook=reporthook)
        
        if status_callback:
            status_callback("浮雕 LoRA 下载完成")
        return save_path
    except Exception as e:
        # Fallback if download fails
        if status_callback:
            status_callback(f"LoRA 下载失败，自动回退到基础模式: {str(e)}")
            print(traceback.format_exc())
        return None
