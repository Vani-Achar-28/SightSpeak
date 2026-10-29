"""SightSpeak - Step 1: FastAPI backend with object detection."""
import io
from collections import Counter
from pathlib import Path

import numpy as np
from deep_translator import GoogleTranslator
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
app = FastAPI(title="SightSpeak API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the server starts
model = YOLO(str(BASE_DIR / "yolov8s.pt"))


def get_direction(x_center: float, width: int) -> str:
    ratio = x_center / width
    if ratio < 0.33:
        return "on your left"
    if ratio > 0.66:
        return "on your right"
    return "in front of you"


def get_distance(box_area: float, img_area: float) -> str:
    ratio = box_area / img_area
    if ratio > 0.40:
        return "very close"
    if ratio > 0.15:
        return "close"
    if ratio > 0.04:
        return "a few steps away"
    return "far away"


@app.get("/api/health")
def health():
    return {"status": "ok"}


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

    objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append(
                {
                    "name": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 2),
                    "direction": get_direction((x1 + x2) / 2, width),
                    "distance": get_distance((x2 - x1) * (y2 - y1), img_area),
                }
            )

    objects.sort(key=lambda o: o["confidence"], reverse=True)

    if objects:
        top = objects[:5]
        sentence = "I can see: " + ". ".join(
            f"{o['name']} {o['direction']}, {o['distance']}" for o in top
        ) + "."
    else:
        sentence = "I could not recognise any objects. Please try again."

    spoken = sentence
    if lang != "en":
        try:
            spoken = GoogleTranslator(source="en", target=lang).translate(sentence)
        except Exception:
            spoken = sentence  # fall back to English if translation fails

    counts = dict(Counter(o["name"] for o in objects))
    return {"objects": objects, "counts": counts, "text": sentence, "spoken": spoken}


# Serve the frontend
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")