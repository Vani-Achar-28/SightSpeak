"""SightSpeak - Phase 1: offline phrases in 7 languages + built-in offline voices."""
import hashlib
import io
import json
import re
import threading
import wave
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from deep_translator import GoogleTranslator, MyMemoryTranslator
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer, VitsModel
from ultralytics import YOLO

from phrases import DIRECTIONS, DISTANCES, OBJECTS, OFFLINE_LANGS, TEMPLATES, UI

BASE_DIR = Path(__file__).parent
app = FastAPI(title="SightSpeak API", version="2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the detection model once when the server starts
model = YOLO(str(BASE_DIR / "yolov8s.pt"))

# ---------------- Built-in voices (Meta MMS-TTS, free, runs offline after download) ----------------
MMS_MODELS = {
    "hi": "facebook/mms-tts-hin",
    "kn": "facebook/mms-tts-kan",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "mr": "facebook/mms-tts-mar",
    "bn": "facebook/mms-tts-ben",
}
TTS_DIR = BASE_DIR / "tts_cache"
TTS_DIR.mkdir(exist_ok=True)
_tts_models = {}
_tts_lock = Lock()


def load_tts(lang: str):
    with _tts_lock:
        if lang not in _tts_models:
            name = MMS_MODELS[lang]
            print(f"Loading voice model {name} (first time downloads about 150 MB)...")
            tok = AutoTokenizer.from_pretrained(name)
            mdl = VitsModel.from_pretrained(name)
            mdl.eval()
            _tts_models[lang] = (tok, mdl)
            print("Voice ready:", lang)
        return _tts_models[lang]


def synthesize(text: str, lang: str) -> bytes:
    key = hashlib.sha1(f"{lang}|{text}".encode("utf-8")).hexdigest()
    path = TTS_DIR / f"{lang}_{key}.wav"
    if path.exists():
        return path.read_bytes()

    tok, mdl = load_tts(lang)
    sr = mdl.config.sampling_rate
    gap = np.zeros(int(sr * 0.25), dtype=np.float32)
    pieces = [p.strip() for p in re.split(r"(?<=[।.!?])\s+", text) if p.strip()]

    chunks = []
    for p in pieces:
        inputs = tok(p, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            continue
        with torch.no_grad():
            wav = mdl(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        chunks += [wav, gap]
    if not chunks:
        raise ValueError("nothing to speak")

    audio = np.concatenate(chunks)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    data = buf.getvalue()
    path.write_bytes(data)
    return data


def _warm(langs):
    for lang in langs:
        try:
            synthesize(UI[lang]["changed"], lang)
            print("Warm-up done:", lang)
        except Exception as e:
            print("WARM-UP ERROR", lang, repr(e))


# ---------------- Online translation (backup only) ----------------
CACHE_FILE = BASE_DIR / "translations_cache.json"
_cache_lock = Lock()
try:
    CACHE = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
except Exception:
    CACHE = {}

MYMEMORY_CODES = {
    "hi": "hi-IN", "kn": "kn-IN", "ta": "ta-IN",
    "te": "te-IN", "mr": "mr-IN", "bn": "bn-IN",
}


def save_cache():
    try:
        CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print("CACHE SAVE ERROR:", repr(e))


def translate_text(text: str, lang: str):
    """Returns (text, provider). Provider is 'failed' if no translator worked."""
    if lang == "en":
        return text, "none"

    key = f"{lang}|{text}"
    with _cache_lock:
        if key in CACHE:
            return CACHE[key], "cache"

    try:
        out = GoogleTranslator(source="en", target=lang).translate(text)
        if out:
            with _cache_lock:
                CACHE[key] = out
                save_cache()
            return out, "google"
    except Exception as e:
        print("GOOGLE TRANSLATE ERROR:", repr(e))

    try:
        out = MyMemoryTranslator(source="en-GB", target=MYMEMORY_CODES.get(lang, "hi-IN")).translate(text)
        if out and "MYMEMORY WARNING" not in out.upper():
            with _cache_lock:
                CACHE[key] = out
                save_cache()
            return out, "mymemory"
    except Exception as e:
        print("MYMEMORY ERROR:", repr(e))

    return text, "failed"


# ---------------- Sentence building ----------------
IRREGULAR_PLURALS = {
    "person": "people", "knife": "knives", "mouse": "mice",
    "sheep": "sheep", "scissors": "scissors", "skis": "skis",
}


def get_direction(x_center: float, width: int) -> str:
    ratio = x_center / width
    if ratio < 0.33:
        return "on your left"
    if ratio > 0.66:
        return "on your right"
    return "in front of you"


def get_distance(area_ratio: float) -> str:
    if area_ratio > 0.40:
        return "very close"
    if area_ratio > 0.15:
        return "close"
    if area_ratio > 0.04:
        return "a few steps away"
    return "far away"


def pluralize(name: str) -> str:
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def phrase(name: str, count: int) -> str:
    if count == 1:
        article = "an" if name[0].lower() in "aeiou" else "a"
        return f"{article} {name}"
    return f"{count} {pluralize(name)}"


def build_sentence(groups: list) -> str:
    """English sentence."""
    if not groups:
        return "I could not recognise anything. Please move the camera and try again."
    parts = []
    for i, g in enumerate(groups):
        text = f"{phrase(g['name'], g['count'])} {g['direction']}, {g['distance']}"
        parts.append("I can see " + text if i == 0 else text[0].upper() + text[1:])
    return ". ".join(parts) + "."


def build_offline(groups: list, lang: str) -> str:
    """Sentence built directly in the chosen language. No internet needed."""
    objs, dirs, dists, tpl = OBJECTS[lang], DIRECTIONS[lang], DISTANCES[lang], TEMPLATES[lang]
    if not groups:
        return tpl["none"]
    parts = [tpl["intro"]]
    for g in groups:
        name = objs.get(g["name"], g["name"])
        count = f"{g['count']} " if g["count"] > 1 else ""
        direction = dirs.get(g["direction"], g["direction"])
        distance = dists.get(g["distance"], g["distance"])
        parts.append(f"{direction} {count}{name}, {distance}")
    return ". ".join(parts) + "."


def speak_groups(groups: list, lang: str):
    """Returns (spoken_text, provider)."""
    if lang == "en":
        return build_sentence(groups), "none"
    if lang in OFFLINE_LANGS:
        return build_offline(groups, lang), "offline"
    return translate_text(build_sentence(groups), lang)


# ---------------- API ----------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/phrases")
async def get_phrases(lang: str = "en"):
    """Fixed interface phrases (welcome, help, ...) in the chosen language."""
    if lang in UI:
        return UI[lang]
    return UI["en"]


@app.post("/api/tts")
async def tts(text: str = Form(...), lang: str = Form("hi")):
    """Built-in voice. Returns a WAV file."""
    if lang not in MMS_MODELS:
        raise HTTPException(status_code=400, detail="No built-in voice for this language")
    try:
        data = await run_in_threadpool(synthesize, text[:1500], lang)
    except Exception as e:
        print("TTS ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=repr(e))
    return Response(content=data, media_type="audio/wav")


@app.get("/api/warm")
async def warm(lang: str = "kn"):
    """Download the voice models in the background. Use lang=all for every language."""
    langs = list(MMS_MODELS) if lang == "all" else [lang]
    langs = [l for l in langs if l in MMS_MODELS]
    threading.Thread(target=_warm, args=(langs,), daemon=True).start()
    return {"started": langs, "note": "Watch the uvicorn terminal for 'Voice ready'."}


@app.post("/api/describe")
async def describe(groups: str = Form(...), lang: str = Form("en")):
    """Say the last scan again in another language."""
    data = json.loads(groups)
    spoken, provider = await run_in_threadpool(speak_groups, data, lang)
    return {"spoken": spoken, "translated": provider != "failed", "provider": provider}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    conf: float = Form(0.25),
):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    img_area = width * height

    results = model(np.array(image), conf=conf, verbose=False)

    # Group detections by (object name, direction) so duplicates are merged
    grouped = {}
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = model.names[int(box.cls[0])]
            direction = get_direction((x1 + x2) / 2, width)
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            key = (name, direction)
            if key not in grouped:
                grouped[key] = {"name": name, "direction": direction, "count": 0, "ratio": 0.0}
            grouped[key]["count"] += 1
            grouped[key]["ratio"] = max(grouped[key]["ratio"], area_ratio)

    groups = sorted(grouped.values(), key=lambda g: g["ratio"], reverse=True)[:6]
    for g in groups:
        g["distance"] = get_distance(g["ratio"])
        g["ratio"] = round(g["ratio"], 3)

    spoken, provider = await run_in_threadpool(speak_groups, groups, lang)
    return {
        "groups": groups,
        "text": build_sentence(groups),
        "spoken": spoken,
        "translated": provider != "failed",
        "provider": provider,
    }


# Serve the frontend
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")