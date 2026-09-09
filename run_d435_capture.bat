@echo off
setlocal
set "TDCR_PYTHON=D:\anaconda3\envs\Bronchoscope\python.exe"
cd /d "%~dp0"
if exist "%TDCR_PYTHON%" (
  "%TDCR_PYTHON%" -m Visual_information.d435_tdcr_capture %*
) else (
  python -m Visual_information.d435_tdcr_capture %*
)
endlocal
