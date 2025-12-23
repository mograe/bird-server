#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Какой питон использовать (можно переопределить: PYTHON=python3.12 ./run.sh)
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python3"
fi

# venv
if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel

# dependencies
python -m pip install -r requirements.txt

# ffmpeg check (нужен для декодирования)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg не найден в PATH. Установи ffmpeg, иначе /predict будет падать."
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Если хочешь autoreload: RELOAD=1 ./run.sh
RELOAD_FLAG=""
if [ "${RELOAD:-0}" = "1" ]; then
  RELOAD_FLAG="--reload"
fi

exec python -m uvicorn server:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
