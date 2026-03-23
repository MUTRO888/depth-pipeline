#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from prepare_offline_bundle import (
    CONTROLNET_ALLOW_PATTERNS,
    MARIGOLD_ALLOW_PATTERNS,
    SD15_ALLOW_PATTERNS,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the full-quality Hugging Face models required by this project."
    )
    parser.add_argument(
        "--config",
        default="config.full.yaml",
        help="Config file that defines the model repo IDs and local paths.",
    )
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def download_snapshot(repo_id: str, target_dir: Path, allow_patterns: list[str]) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
        resume_download=True,
    )


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_cfg = config["model"]
    sd_cfg = config["sd_relief"]

    download_snapshot(
        repo_id=model_cfg["name"],
        target_dir=project_path(model_cfg["local_path"]),
        allow_patterns=MARIGOLD_ALLOW_PATTERNS,
    )
    download_snapshot(
        repo_id=sd_cfg["model_id"],
        target_dir=project_path(sd_cfg["base_model_local_path"]),
        allow_patterns=SD15_ALLOW_PATTERNS,
    )
    download_snapshot(
        repo_id=sd_cfg["controlnet_model_id"],
        target_dir=project_path(sd_cfg["controlnet_local_path"]),
        allow_patterns=CONTROLNET_ALLOW_PATTERNS,
    )
    print("[done] Full-quality Hugging Face models are ready.")


if __name__ == "__main__":
    main()
