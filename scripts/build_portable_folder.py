#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dist" / "depth-pipeline-offline-win"
SOURCE_FILES = [
    "app.py",
    "pipeline.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
]
SOURCE_DIRS = [
    "steps",
    "utils",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable Windows offline folder from this project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Final portable folder path.",
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
        default=None,
        help="Target ABI tag. Default: derived from --python-version",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu121",
        help="PyTorch wheel index URL.",
    )
    parser.add_argument(
        "--include-sd-relief",
        action="store_true",
        help="Also bundle optional SD relief models and LoRA.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the existing output directory first.",
    )
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
        dirs_exist_ok=True,
    )


def normalize_python_version_tag(version_tag: str) -> str:
    return version_tag.replace(".", "").strip()


def python_version_for_windows(version_tag: str) -> str:
    version_tag = normalize_python_version_tag(version_tag)
    if len(version_tag) == 2:
        return f"{version_tag[0]}.{version_tag[1]}"
    if len(version_tag) == 3:
        return f"{version_tag[0]}.{version_tag[1:]}"
    raise ValueError(f"Unsupported python version tag: {version_tag}")


def write_windows_launcher(output_dir: Path, python_version: str) -> None:
    template_path = ROOT_DIR / "scripts" / "windows" / "install_and_run_offline.bat"
    content = template_path.read_text(encoding="utf-8")
    content = content.replace("__TARGET_PYTHON__", python_version)
    (output_dir / "install_and_run_offline.bat").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in SOURCE_FILES:
        shutil.copy2(ROOT_DIR / filename, output_dir / filename)

    for dirname in SOURCE_DIRS:
        copy_tree(ROOT_DIR / dirname, output_dir / dirname)

    write_windows_launcher(output_dir, python_version_for_windows(args.python_version))

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "prepare_offline_bundle.py"),
            "--target-dir",
            str(output_dir),
            "--platform",
            args.platform,
            "--python-version",
            normalize_python_version_tag(args.python_version),
            "--implementation",
            args.implementation,
            "--abi",
            args.abi or f"{args.implementation}{normalize_python_version_tag(args.python_version)}",
            "--torch-index-url",
            args.torch_index_url,
            *(["--include-sd-relief"] if args.include_sd_relief else []),
        ],
        check=True,
    )

    print(f"[done] Portable folder created at: {output_dir}")


if __name__ == "__main__":
    main()
