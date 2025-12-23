# bird-server (BirdLover Inference API)

Небольшой FastAPI-сервер для инференса модели распознавания птицы по аудио.

- **API**: `GET /health`, `POST /predict`
- **Вход**: аудиофайл (любой формат, который умеет ffmpeg)
- **Выход**: top-K классов с вероятностями
- При старте в консоль печатаются адреса для открытия:
  - `LAN: http://<локальный_IP>:<порт>`
  - `Local: http://127.0.0.1:<порт>`

---

## Структура проекта

- `server.py` — FastAPI приложение
- `model/`
  - `sipuha_v2.pth` — веса
  - `classes.json` — список классов
- `requirements.txt` — зависимости (может быть «linux/cuda-ориентированным»)
- Скрипты запуска (рекомендуется держать рядом с `server.py`):
  - `run_cpu.bat`, `run_gpu.bat` — Windows
  - `run_cpu.sh`, `run_gpu.sh` — Linux

---

## Требования

- Python **3.11+**
- `ffmpeg` в `PATH` или в одной папке с exe или скриптом (иначе `/predict` будет падать на декодировании)
- (Опционально) NVIDIA GPU + CUDA (если хочешь инференс на GPU)

---

## Быстрый старт через скрипты

> Скрипты обычно:
> 1) создают `.venv`
> 2) ставят `pip/setuptools/wheel`
> 3) ставят **PyTorch отдельно** (CPU или CUDA-сборку)
> 4) ставят зависимости проекта (без CUDA-linux-пакетов на Windows)
> 5) запускают `uvicorn server:app`

### Windows (CPU)

```powershell
.\run_cpu.bat
````

### Windows (GPU)

```powershell
.\run_gpu.bat
```

---

### Linux (CPU)

```bash
chmod +x ./run.sh
./run.sh
```

---

## Настройки через переменные окружения

По умолчанию (если не задано) сервер стартует на `0.0.0.0:8000`.

Можно переопределить:

* `HOST` — хост (например `127.0.0.1` или `0.0.0.0`)
* `PORT` — порт (например `8000`)

Примеры:

### Windows

```powershell
$env:HOST="0.0.0.0"
$env:PORT="8000"
.\run_cpu.bat
```

### Linux

```bash
HOST=0.0.0.0 PORT=8000 ./run_cpu.sh
```

---

## Запуск без скриптов (вручную)

```bash
python -m venv .venv
# Windows:
#   .\.venv\Scripts\activate
# Linux:
#   source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> ⚠️ На Windows `requirements.txt` может содержать linux/cuda-пакеты (типа `nvidia-*`, `triton`, pinned `torch==...+cu...`),
> которые **не ставятся**. Для Windows обычно делают «очищенный» requirements (см. ниже в разделе про `.bat`) и ставят torch отдельно.

---

## Проверка работы

### Healthcheck

Открой в браузере:

* `http://127.0.0.1:8000/health`
* или `http://<твой_LAN_IP>:8000/health`

Ожидаемый ответ примерно:

```json
{"ok": true, "device": "cuda|cpu", "num_classes": 123}
```

### Predict (curl)

```bash
curl -X POST "http://127.0.0.1:8000/predict?tta=5&top_k=3" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.wav"
```

Ответ:

```json
{
  "top_k": [
    {"class": "...", "prob": 0.91},
    {"class": "...", "prob": 0.05},
    {"class": "...", "prob": 0.02}
  ],
  "tta": 5,
  "top_k_n": 3
}
```

---

# Запуск EXE (Windows)


## Как запустить

1. Убедись, что рядом с `.exe` есть папка `model/` с:

* `model/sipuha_v2.pth`
* `model/classes.json`

2. Запусти `.exe`

3. В консоли увидишь адреса (LAN и Local) — открывай `/health` или дергай `/predict`.

> ⚠️ Если `ffmpeg` не встроен в сборку — поставь `ffmpeg` и добавь в `PATH`.

---

# Сборка EXE через spec-файлы (ветка `exe`)

В ветке `exe` лежат готовые `.spec` файлы для PyInstaller:

* `bird-server-cpu.spec`
* `bird-server-gpu.spec`
* `requirements.base.txt`
* `run_server.py` (вспомогательный entrypoint для упаковки)

## 1) Переключись на ветку exe

```bash
git fetch
git checkout exe
```

## 2) Создай окружение и поставь зависимости

### Вариант CPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.base.txt

# PyTorch CPU (вариант)
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m pip install pyinstaller
pyinstaller bird-server-cpu.spec
```

### Вариант GPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.base.txt

# PyTorch CUDA (подбери под свою CUDA)
# пример (если используешь cu121):
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m pip install pyinstaller
pyinstaller bird-server-gpu.spec
```

## 3) Где будет результат

Обычно PyInstaller складывает результат в:

* `dist/...` (готовая папка с `.exe`)
* `build/...` (временные файлы)

Запускай `.exe` из `dist/`.

---


