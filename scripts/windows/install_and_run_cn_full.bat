@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..\..") do set "PROJECT_DIR=%%~fI"
cd /d "%PROJECT_DIR%"

set "TARGET_PYTHON=3.10"
set "VENV_DIR=%CD%\.venv"
set "CONFIG_FILE=%CD%\config.full.yaml"
set "LORA_FILE=%CD%\models\sd-relief\lora\relief_lora_438287.safetensors"
set "PYTHON_EXE="
set "PYTHON_VERSION="
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

for /f "delims=" %%I in ('py -%TARGET_PYTHON% -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%I"
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%I"
)

if not defined PYTHON_EXE (
  echo [ERROR] No Python interpreter found. Install Python %TARGET_PYTHON% x64 first. 1>&2
  exit /b 1
)

for /f "delims=" %%I in ('"%PYTHON_EXE%" -c "import sys; print(""{}.{}"".format(sys.version_info.major, sys.version_info.minor))" 2^>nul') do set "PYTHON_VERSION=%%I"
if not "%PYTHON_VERSION%"=="%TARGET_PYTHON%" (
  echo [ERROR] This bundle was built for Python %TARGET_PYTHON%, but found Python %PYTHON_VERSION%. 1>&2
  exit /b 1
)

if not exist "%CONFIG_FILE%" (
  echo [ERROR] Missing full-quality config: %CONFIG_FILE% 1>&2
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  "%PYTHON_EXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install -U pip -i "%PIP_INDEX_URL%"
if errorlevel 1 exit /b 1

python -m pip install -r "%CD%\requirements.txt" -i "%PIP_INDEX_URL%"
if errorlevel 1 exit /b 1

set "HF_ENDPOINT=https://hf-mirror.com"
set "DEPTH_PIPELINE_CONFIG=%CONFIG_FILE%"

if exist "%CD%\.git" (
  for /f "delims=" %%I in ('git lfs version 2^>nul') do set "HAS_GIT_LFS=1"
  if defined HAS_GIT_LFS (
    git lfs pull
  )
)

if not exist "%LORA_FILE%" (
  echo [ERROR] Missing full-quality LoRA file: %LORA_FILE% 1>&2
  echo [ERROR] This file should be stored in the GitHub repo via Git LFS. 1>&2
  exit /b 1
)

python -c "from pathlib import Path; import sys; p=Path(r'%LORA_FILE%'); text=p.read_text(encoding='utf-8', errors='ignore') if p.exists() and p.stat().st_size < 4096 else ''; sys.exit(1 if 'git-lfs.github.com/spec/v1' in text else 0)"
if errorlevel 1 (
  echo [ERROR] The LoRA file is still a Git LFS pointer. Install Git LFS and run `git lfs pull`. 1>&2
  exit /b 1
)

python scripts\download_full_models.py --config "%CONFIG_FILE%"
if errorlevel 1 exit /b 1

python app.py
exit /b %errorlevel%
