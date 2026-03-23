@echo off
setlocal

cd /d "%~dp0"

set "TARGET_PYTHON=__TARGET_PYTHON__"
set "VENV_DIR=%CD%\.venv"
set "WHEEL_DIR=%CD%\wheels"
set "CONFIG_FILE=%CD%\config.offline.yaml"
set "PYTHON_EXE="
set "PYTHON_VERSION="

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
  echo [ERROR] This offline bundle was built for Python %TARGET_PYTHON%, but found Python %PYTHON_VERSION%. 1>&2
  exit /b 1
)

if not exist "%WHEEL_DIR%" (
  echo [ERROR] Missing wheel directory: %WHEEL_DIR% 1>&2
  exit /b 1
)

if not exist "%CONFIG_FILE%" (
  echo [ERROR] Missing offline config: %CONFIG_FILE% 1>&2
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  "%PYTHON_EXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --no-index --find-links "%WHEEL_DIR%" -r "%CD%\requirements.offline.txt"
if errorlevel 1 exit /b 1

set "DEPTH_PIPELINE_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DEPTH_PIPELINE_CONFIG=%CONFIG_FILE%"

python app.py
exit /b %errorlevel%
