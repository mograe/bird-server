@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM --- выбрать Python 3.11 через launcher, если есть ---
where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3.11"
) else (
  set "PY=python"
)

REM --- venv ---
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)

call ".venv\Scripts\activate.bat"

python -m pip install -U pip setuptools wheel

REM --- Windows: выкидываем nvidia-* (они в requirements для Linux и на Windows часто не ставятся) ---
findstr /V /R /C:"^nvidia-" /C:"^torch==" /C:"^torchvision==" /C:"^torchaudio==" /C:"^sympy==" /C:"^triton==" /C:"^uvloop==" requirements.txt > requirements.windows.txt

REM --- Torch GPU (CUDA 12.1) ---
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.windows.txt

REM --- ffmpeg check ---
where ffmpeg >nul 2>&1
if not %errorlevel%==0 (
  echo [WARN] ffmpeg не найден в PATH. Установи ffmpeg, иначе /predict будет падать.
)

if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

python -m uvicorn server:app --host %HOST% --port %PORT%
endlocal

if errorlevel 1 (
  echo.
  echo [ERROR] Скрипт завершился с ошибкой. Код: %errorlevel%
  pause
)