import json
import math
import tempfile
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision.models import efficientnet_b0
import torchaudio
import socket

from contextlib import asynccontextmanager
import os

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "sipuha_v2.pth"
CLASSES_PATH = MODEL_DIR / "classes.json"

TARGET_SR = 32000
CLIP_SECONDS = 5.0
TARGET_LEN = int(TARGET_SR * CLIP_SECONDS)

N_FFT = 1024
HOP = 320
N_MELS = 128
FMIN = 0.0
FMAX = TARGET_SR / 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app):
    port = int(os.getenv("PORT", "8000"))  # если у тебя порт в env
    print(f"LAN:   http://{get_lan_ip()}:{port}")
    print(f"Local: http://127.0.0.1:{port}")
    yield


app = FastAPI(title="BirdLover Inference API", lifespan=lifespan)


def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def load_classes() -> List[str]:
    if not CLASSES_PATH.exists():
        raise RuntimeError(f"Classes file not found: {CLASSES_PATH}")
    return json.load(open(CLASSES_PATH, "r", encoding="utf-8"))

def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model

def decode_to_wav(input_path: Path, out_wav_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "wav",
        str(out_wav_path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode('utf-8', errors='ignore')}")
    
def crop_or_pad(x: np.ndarray, target_len: int, start: int | None = None) -> np.ndarray:
    T = x.shape[0]
    if T >= target_len:
        if start is None:
            start = (T - target_len) // 2
        start = max(0, min(start, T - target_len))
        return x[start:start+target_len]
    return np.pad(x, (0, target_len - T), mode="constant")


def make_starts(T: int, target_len: int, n_crops: int) -> List[int]:
    if T <= target_len:
        return [0]
    if n_crops <= 1:
        return [(T - target_len) // 2]
    return np.linspace(0, T - target_len, n_crops, dtype=int).tolist()

class BirdPredictor:
    def __init__(self):
        if not MODEL_DIR.exists():
            raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
        
        self.classes = load_classes()
        self.num_classes = len(self.classes)

        self.model = build_model(self.num_classes)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(DEVICE)

        self.window = torch.hann_window(N_FFT).to(DEVICE)
        self.fbanks = None

    def _ensure_fbanks(self, n_freqs: int):
        if self.fbanks is not None and self.fbanks.shape[0] == n_freqs:
            return
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=FMIN,
            f_max=FMAX,
            n_mels=N_MELS,
            sample_rate=TARGET_SR,
            norm=None,
            mel_scale="htk",
        )
        self.fbanks = fb.to(DEVICE)

    @torch.no_grad()
    def logmel(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav).to(DEVICE)
        stft = torch.stft(
            x,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=self.window,
            center=True,
            return_complex=True,
        )
        spec = (stft.abs() ** 2)

        n_freqs = spec.shape[0]
        self._ensure_fbanks(n_freqs)

        mel = spec.transpose(0, 1) @ self.fbanks  # [frames, n_mels]
        mel = mel.transpose(0, 1)            # [n_mels, frames]
        logmel = torch.log(mel + 1e-10)   
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        
        X = logmel.unsqueeze(0).repeat(3, 1, 1) # [3, n_mels, frames]
        return X.unsqueeze(0) # [1, 3, n_mels, frames]
    
    @torch.no_grad()
    def predict(self, wav: np.ndarray, tta: int = 5, top_k: int = 3):
        T = wav.shape[0]
        starts = make_starts(T, TARGET_LEN, tta)

        logits_sum = None
        for s in starts:
            clip = crop_or_pad(wav, TARGET_LEN, start=s)
            X = self.logmel(clip)
            logits = self.model(X)
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        logits = logits_sum / len(starts)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        
        idx = np.argsort(-probs)[:top_k]
        return [
            {"class": self.classes[i], "prob": float(probs[i])} for i in idx
        ]
    
PRED = BirdPredictor()



@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "num_classes": PRED.num_classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...), tta: int = 5, top_k: int = 3):
    if tta < 1 or tta > 20:
        raise HTTPException(status_code=400, detail="tta must be in range [1, 20]")
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be in range [1, 10]")
    
    suffix = Path(file.filename).suffix.lower() if file.filename else ".bin"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / f"input{suffix}"
        out_wav = td / "decoded.wav"

        data = await file.read()
        in_path.write_bytes(data)

        try:
            decode_to_wav(in_path, out_wav)
            wav, sr = sf.read(str(out_wav), always_2d=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        wav = wav.mean(axis=1).astype(np.float32)
        if sr != TARGET_SR:
            raise HTTPException(
                status_code=400,
                detail=f"sample rate must be {TARGET_SR}, got {sr}",
            )

        result = PRED.predict(wav, tta=tta, top_k=top_k)

        return {"top_k": result, "tta": tta, "top_k_n": top_k}
    