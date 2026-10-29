# SightSpeak 

**A multilingual AI vision assistant for visually impaired users.**
Point a camera at a scene and SightSpeak tells you, out loud, what is around you, where it is, and how far away.

## Features
- Live camera interface with one large, high-contrast SCAN button (built for accessibility)
- Object detection with YOLO
- Direction guidance: *on your left / in front of you / on your right*
- Distance estimate: *very close / close / a few steps away / far away*
- Voice output in 7 languages: English, Hindi, Kannada, Tamil, Telugu, Marathi, Bengali
- REST API with automatic documentation (FastAPI `/docs`)

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Computer Vision | Ultralytics YOLO, Pillow, NumPy |
| Translation | deep-translator |
| Frontend | HTML, CSS, JavaScript, MediaDevices API, Web Speech API |

## How to Run
```bash
git clone https://github.com/Vani-Achar-28/SightSpeak.git
cd SightSpeak
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** in Chrome and allow camera access.

The model file `yolov8s.pt` is not stored in this repo. It should download on first run; if it does not, place `yolov8s.pt` in the project folder.

## API
`POST /api/detect` takes an image (`file`), a language code (`lang`) and a confidence value (`conf`), and returns the detected objects with direction and distance, plus a sentence to speak.

## Screenshots
*(add a screenshot of the app here)*

## Roadmap
- [ ] Read text aloud (OCR) for signboards, labels and medicine strips
- [ ] Scene description with a vision-language model
- [ ] Voice commands ("scan", "read this")
- [ ] Scan history with a database
- [ ] Custom-trained model for currency notes
- [ ] Docker and cloud deployment
- [ ] Accuracy evaluation on a test set

## Limitations
- Uses a pretrained model (80 common object classes), so unusual objects may be missed
- Distance and direction are estimated from the position and size of the detected box, not measured
- Needs an internet connection for translation
- Not a replacement for a cane or other mobility aids

## Author
Vani Achar, MCA
