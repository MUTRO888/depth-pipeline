#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from prepare_offline_bundle import (
    DEFAULT_CIVITAI_MODEL_VERSION_ID,
    download_civitai_lora,
    download_runtime_wheels,
    download_torch_wheels,
    write_offline_requirements,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dist" / "depth-pipeline-cn-full"
SOURCE_FILES = [
    "app.py",
    "pipeline.py",
    "config.yaml",
    "config.full.yaml",
    "requirements.txt",
    "README.md",
]
SOURCE_DIRS = [
    "steps",
    "utils",
    "scripts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Windows support bundle for the full-quality China-network workflow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Final bundle path.",
    )
    parser.add_argument(
        "--platform",
        default="win_amd64",
        help="Target pip platform tag. Default: win_amd64",
    )
    parser.add_argument(
        "--python-version",
        default="312",
        help="Target Python version for wheel download. Default: 312",
    )
    parser.add_argument(
        "--implementation",
        default="cp",
        help="Target Python implementation. Default: cp",
    )
    parser.add_argument(
        "--abi",
        default="cp312",
        help="Target ABI tag.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu121",
        help="PyTorch wheel index URL.",
    )
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
        dirs_exist_ok=True,
    )


def write_manifest(output_dir: Path, args: argparse.Namespace, lora_path: Path) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_type": "cn_full_quality",
        "target_platform": args.platform,
        "target_python_version": args.python_version,
        "target_implementation": args.implementation,
        "target_abi": args.abi,
        "torch_index_url": args.torch_index_url,
        "hf_endpoint": "https://hf-mirror.com",
        "downloaded_paths": {
            "sd_lora": str(lora_path.relative_to(output_dir)).replace("\\", "/"),
            "wheel_dir": "wheels",
        },
    }
    (output_dir / "cn_full_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in SOURCE_FILES:
        shutil.copy2(ROOT_DIR / filename, output_dir / filename)

    for dirname in SOURCE_DIRS:
        copy_tree(ROOT_DIR / dirname, output_dir / dirname)

    wheel_dir = output_dir / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    download_runtime_wheels(args, wheel_dir)
    download_torch_wheels(args, wheel_dir)
    write_offline_requirements(output_dir)

    lora_path = output_dir / "models" / "sd-relief" / "lora" / f"relief_lora_{DEFAULT_CIVITAI_MODEL_VERSION_ID}.safetensors"
    download_civitai_lora(lora_path)

    shutil.copy2(
        ROOT_DIR / "scripts" / "windows" / "install_and_run_cn_full.bat",
        output_dir / "install_and_run_cn_full.bat",
    )

    write_manifest(output_dir, args, lora_path)
    print(f"[done] China-network full-quality bundle created at: {output_dir}")


if __name__ == "__main__":
    main()
