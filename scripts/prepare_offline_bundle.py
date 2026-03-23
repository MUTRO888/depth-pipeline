#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import venv
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_DIR = ROOT_DIR / "dist" / "depth-pipeline-offline-win"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
DEFAULT_HF_HELPER_DIR = ROOT_DIR / ".offline-build-tools" / "hf-helper-venv"
DEFAULT_CIVITAI_MODEL_VERSION_ID = "438287"

MARIGOLD_ALLOW_PATTERNS = [
    "README.md",
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

SD15_ALLOW_PATTERNS = [
    "README.md",
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/model.fp16.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

CONTROLNET_ALLOW_PATTERNS = [
    "README.md",
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Windows wheels and models into a portable offline bundle."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Portable bundle directory.",
    )
    parser.add_argument(
        "--platform",
        default="win_amd64",
        help="Target pip platform tag. Default: win_amd64",
    )
    parser.add_argument(
        "--python-version",
        default="310",
        help="Target Python version for wheels. Default: 310",
    )
    parser.add_argument(
        "--implementation",
        default="cp",
        help="Target Python implementation. Default: cp",
    )
    parser.add_argument(
        "--abi",
        default=None,
        help="Target ABI tag. Default: derived from --python-version",
    )
    parser.add_argument(
        "--torch-index-url",
        default=DEFAULT_TORCH_INDEX_URL,
        help="PyTorch wheel index URL.",
    )
    parser.add_argument(
        "--include-sd-relief",
        action="store_true",
        help="Also download the optional SD relief models and LoRA.",
    )
    parser.add_argument(
        "--hf-helper-dir",
        type=Path,
        default=DEFAULT_HF_HELPER_DIR,
        help="Hidden helper virtualenv used to download Hugging Face models.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def normalize_python_version_tag(version: str) -> str:
    return version.replace(".", "").strip()


def resolve_abi(args: argparse.Namespace) -> str:
    return args.abi or f"{args.implementation}{normalize_python_version_tag(args.python_version)}"


def normalize_requirement_name(line: str) -> str:
    line = line.split("#", 1)[0].strip()
    if not line:
        return ""
    for separator in ("[", "=", "<", ">", "!", "~", " "):
        if separator in line:
            line = line.split(separator, 1)[0]
            break
    return line.strip().lower().replace("_", "-")


def write_runtime_requirements_file(temp_path: Path) -> None:
    requirements_path = ROOT_DIR / "requirements.txt"
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    filtered_lines = []
    for raw_line in lines:
        name = normalize_requirement_name(raw_line)
        if name in {"torch", "torchvision"}:
            continue
        filtered_lines.append(raw_line)
    temp_path.write_text("\n".join(filtered_lines).strip() + "\n", encoding="utf-8")


def pip_download_common_args(args: argparse.Namespace, destination: Path) -> list[str]:
    return [
        "-m",
        "pip",
        "download",
        "--dest",
        str(destination),
        "--platform",
        args.platform,
        "--python-version",
        normalize_python_version_tag(args.python_version),
        "--implementation",
        args.implementation,
        "--abi",
        resolve_abi(args),
        "--only-binary",
        ":all:",
        "--prefer-binary",
    ]


def download_runtime_wheels(args: argparse.Namespace, wheel_dir: Path) -> None:
    temp_requirements = wheel_dir / ".requirements.runtime.txt"
    write_runtime_requirements_file(temp_requirements)
    try:
        run(
            [
                sys.executable,
                *pip_download_common_args(args, wheel_dir),
                "-r",
                str(temp_requirements),
            ]
        )
    finally:
        temp_requirements.unlink(missing_ok=True)


def download_torch_wheels(args: argparse.Namespace, wheel_dir: Path) -> None:
    run(
        [
            sys.executable,
            *pip_download_common_args(args, wheel_dir),
            "--index-url",
            args.torch_index_url,
            "--extra-index-url",
            "https://pypi.org/simple",
            "torch",
            "torchvision",
        ]
    )


def helper_python_path(helper_dir: Path) -> Path:
    if os.name == "nt":
        return helper_dir / "Scripts" / "python.exe"
    return helper_dir / "bin" / "python"


def ensure_hf_helper(helper_dir: Path) -> Path:
    helper_python = helper_python_path(helper_dir)
    if helper_python.exists():
        try:
            run(
                [
                    str(helper_python),
                    "-c",
                    "import huggingface_hub",
                ]
            )
            return helper_python
        except subprocess.CalledProcessError:
            pass

    if helper_dir.exists():
        shutil.rmtree(helper_dir)

    print(f"[info] Creating helper environment: {helper_dir}")
    helper_dir.parent.mkdir(parents=True, exist_ok=True)
    venv.create(helper_dir, with_pip=True, clear=True)
    helper_python = helper_python_path(helper_dir)
    run([str(helper_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(helper_python), "-m", "pip", "install", "huggingface_hub>=0.30"])
    return helper_python


def download_huggingface_snapshot(
    helper_python: Path,
    repo_id: str,
    target_dir: Path,
    allow_patterns: list[str] | None = None,
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    helper_code = textwrap.dedent(
        """
        import json
        import sys
        from huggingface_hub import snapshot_download

        repo_id = sys.argv[1]
        target_dir = sys.argv[2]
        allow_patterns = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
        """
    ).strip()
    run(
        [
            str(helper_python),
            "-c",
            helper_code,
            repo_id,
            str(target_dir),
            json.dumps(allow_patterns or []),
        ]
    )


def download_civitai_lora(target_path: Path, model_version_id: str = DEFAULT_CIVITAI_MODEL_VERSION_ID) -> None:
    if target_path.exists():
        print(f"[info] Reusing existing LoRA: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://civitai.com/api/download/models/{model_version_id}"
    print(f"[info] Downloading Civitai LoRA -> {target_path}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, target_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def load_base_config() -> dict:
    config_path = ROOT_DIR / "config.yaml"
    try:
        import yaml  # type: ignore
    except ImportError:
        return {
            "model": {
                "name": "prs-eth/marigold-depth-v1-1",
                "local_path": "models/marigold-depth-v1-1",
                "torch_dtype": "auto",
                "device": "auto",
            },
            "runtime": {"offline": False},
            "sd_relief": {
                "use_sd_relief": False,
                "model_id": "runwayml/stable-diffusion-v1-5",
                "base_model_local_path": "models/sd-relief/stable-diffusion-v1-5",
                "controlnet_model_id": "lllyasviel/control_v11f1p_sd15_depth",
                "controlnet_local_path": "models/sd-relief/control_v11f1p_sd15_depth",
                "lora_path": "auto",
                "lora_scale": 0.8,
                "prompt": "intricate micro details, subtle surface texture, high resolution, relief carving",
                "negative_prompt": "smooth, blurry, deep shadows, noisy, high contrast",
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
            },
            "preprocess": {
                "remove_specular": True,
                "specular_v_threshold": 220,
                "specular_s_threshold": 50,
                "normalize_illumination": True,
                "retinex_sigmas": [15, 80, 250],
            },
            "inference": {
                "ensemble_size": 5,
                "denoising_steps": 10,
            },
            "postprocess": {"gaussian_radius": 2},
            "fusion": {
                "detail_strength": 0.8,
                "relief_strength": 0.5,
                "guided_radius": None,
                "guided_eps": 0.001,
            },
        }

    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml_lines(value, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines

    return [f"{prefix}{yaml_scalar(value)}"]


def write_yaml(path: Path, payload: dict) -> None:
    lines = dump_yaml_lines(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_offline_config(target_dir: Path, include_sd_relief: bool) -> None:
    config = load_base_config()
    config.setdefault("runtime", {})["offline"] = True
    config.setdefault("model", {})["local_path"] = "models/marigold-depth-v1-1"

    sd_relief = config.setdefault("sd_relief", {})
    sd_relief.setdefault("base_model_local_path", "models/sd-relief/stable-diffusion-v1-5")
    sd_relief.setdefault(
        "controlnet_local_path",
        "models/sd-relief/control_v11f1p_sd15_depth",
    )
    if include_sd_relief:
        sd_relief["use_sd_relief"] = True
        sd_relief["lora_path"] = "models/sd-relief/lora/relief_lora_438287.safetensors"
    else:
        sd_relief["use_sd_relief"] = False
        if sd_relief.get("lora_path") == "auto":
            sd_relief["lora_path"] = ""

    write_yaml(target_dir / "config.offline.yaml", config)


def write_offline_requirements(target_dir: Path) -> None:
    content = textwrap.dedent(
        """
        -r requirements.txt
        torch
        torchvision
        """
    ).strip()
    (target_dir / "requirements.offline.txt").write_text(content + "\n", encoding="utf-8")


def relative_to_target(path: Path, target_dir: Path) -> str:
    return str(path.relative_to(target_dir)).replace("\\", "/")


def write_manifest(
    target_dir: Path,
    wheel_dir: Path,
    model_paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_platform": args.platform,
        "target_python_version": args.python_version,
        "target_implementation": args.implementation,
        "target_abi": resolve_abi(args),
        "torch_index_url": args.torch_index_url,
        "include_sd_relief": args.include_sd_relief,
        "downloaded_paths": {
            key: relative_to_target(path, target_dir)
            for key, path in model_paths.items()
        },
        "wheel_dir": relative_to_target(wheel_dir, target_dir),
    }
    manifest_path = target_dir / "offline_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.resolve()
    wheel_dir = target_dir / "wheels"
    model_dir = target_dir / "models"

    wheel_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Preparing offline bundle in: {target_dir}")
    download_runtime_wheels(args, wheel_dir)
    download_torch_wheels(args, wheel_dir)

    helper_python = ensure_hf_helper(args.hf_helper_dir.resolve())

    downloaded_paths = {
        "depth_model": model_dir / "marigold-depth-v1-1",
    }
    download_huggingface_snapshot(
        helper_python,
        "prs-eth/marigold-depth-v1-1",
        downloaded_paths["depth_model"],
        allow_patterns=MARIGOLD_ALLOW_PATTERNS,
    )

    if args.include_sd_relief:
        downloaded_paths["sd_base_model"] = model_dir / "sd-relief" / "stable-diffusion-v1-5"
        downloaded_paths["sd_controlnet"] = model_dir / "sd-relief" / "control_v11f1p_sd15_depth"
        downloaded_paths["sd_lora"] = model_dir / "sd-relief" / "lora" / "relief_lora_438287.safetensors"
        download_huggingface_snapshot(
            helper_python,
            "runwayml/stable-diffusion-v1-5",
            downloaded_paths["sd_base_model"],
            allow_patterns=SD15_ALLOW_PATTERNS,
        )
        download_huggingface_snapshot(
            helper_python,
            "lllyasviel/control_v11f1p_sd15_depth",
            downloaded_paths["sd_controlnet"],
            allow_patterns=CONTROLNET_ALLOW_PATTERNS,
        )
        download_civitai_lora(downloaded_paths["sd_lora"])

    write_offline_config(target_dir, include_sd_relief=args.include_sd_relief)
    write_offline_requirements(target_dir)
    write_manifest(target_dir, wheel_dir, downloaded_paths, args)

    print("[done] Offline assets are ready.")


if __name__ == "__main__":
    main()
