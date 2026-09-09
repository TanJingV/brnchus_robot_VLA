@echo off
setlocal
set "TDCR_PYTHON=D:\anaconda3\envs\Bronchoscope\python.exe"
set "YOLO_CONFIG_DIR=%~dp0.ultralytics"
set "XDG_CONFIG_HOME=%~dp0.ultralytics"
set "HF_HOME=%~dp0.hf"
set "KMP_DUPLICATE_LIB_OK=TRUE"
cd /d "%~dp0"
if exist "%TDCR_PYTHON%" (
  "%TDCR_PYTHON%" -m Visual_information.seven_marker_3d_fusion.ui
) else (
  python -m Visual_information.seven_marker_3d_fusion.ui
)
endlocal
