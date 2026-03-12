

# IdentiFruit 🍓🍌🍎🍊

IdentiFruit is a web app that uses a computer vision model to classify **which fruit** is in an image and whether it is **fresh or rotten**.

- No user accounts
- Two-page UI: **Home** → **App** (camera or upload)
- One server: **Python backend serves the pages** + exposes an API endpoint for predictions
- Classifies **4 fruits** (apple, banana, orange, strawberry) as fresh or rotten, or rejects unknown images

---

## Tech Stack

- Backend: **FastAPI (Python)** — serves pages + `/predict` API
- Model: **PyTorch** — DINOv2 ViT-B/14 (self-supervised pretraining, fine-tuned)
- Frontend: **HTML templates (Jinja)** + minimal JavaScript
- Styling: **Tailwind CSS** → compiled into `src/app/static/styles.css`

---

# Setup

## 1) Python
- **Python 3.11** recommended
```bash
pip install -r requirements.txt
```

## 2) Node.js / npm
Required for Tailwind CSS. Install **Node.js (LTS)**, then:
```bash
npm install
npm run build:css
```

---

# Quick Commands

## Build Tailwind CSS (after you edit styles)
```bash
npm run build:css
```

## Train the model
```bash
python -m src.training.train
```

Reads images from `data/Train/`. Outputs:
- `models/fruit_model.pt` — best checkpoint (by val loss), includes class names

Training uses two phases: frozen backbone (5 epochs) then full fine-tune (20 epochs).
Supports resume — if interrupted, re-run the same command and enter `y` to continue.

## Evaluate against test set
```bash
python -m src.inference.eval_test
```

Runs every image in `data/Test/` through the model and prints a per-folder accuracy table.

## Run the web app (development)
```bash
python -m uvicorn src.app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/app`