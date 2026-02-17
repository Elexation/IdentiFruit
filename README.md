

# IdentiFruit 🍓🍌🍎🥭🍊

IdentiFruit is a simple web app that uses a computer vision model to classify **which fruit** is in an image and whether it is **fresh or rotten**.

- No user accounts  
- Two-page UI: **Home** → **App** (camera or upload)  
- One server: **Python backend serves the pages** + exposes an API endpoint for predictions  

---

## Tech Stack

- Backend: **FastAPI (Python)** — serves pages + `/predict` API  
- Model: **PyTorch** — training + inference  
- Frontend: **HTML templates (Jinja)** + minimal JavaScript  
- Styling: **Tailwind CSS** → compiled into `src/app/static/styles.css`  

---

## Project Structure

> `node_modules/` is generated locally by npm and is not committed.

```txt
IdentiFruit/
├─ package.json                 npm deps + scripts (Tailwind build)
├─ package-lock.json            locked npm versions (team consistency)
├─ tailwind.config.js           Tailwind config (content scan)
├─ .gitignore                   ignores node_modules, data, model artifacts, etc.
├─ requirements.txt             Python deps (training + API)
├─ README.md                    project overview + commands
│
├─ data/                        dataset images (kept local; not committed)
├─ models/                      trained outputs (generated after training)
│  ├─ fruit_model.pt            model weights
│  └─ classes.txt               class order / label map
│
└─ src/                         application source code
   ├─ training/
   │  ├─ dataset.py             loads images + labels from `data/`
   │  └─ train.py               trains + saves into `models/`
   │
   ├─ inference/
   │  ├─ model.py               loads weights + preprocessing
   │  └─ predict.py             returns {fruit, freshness, confidence}
   │
   └─ app/
      ├─ main.py                FastAPI app instance (uvicorn entrypoint)
      ├─ routes.py              endpoints: /, /app, /predict
      ├─ templates/             HTML pages (Jinja)
      │  ├─ home.html           landing page
      │  └─ app.html            upload/camera page
      ├─ static/                browser assets
      │  ├─ app.js              upload + fetch(/predict)
      │  └─ styles.css          compiled Tailwind output
      └─ tailwind/
         └─ input.css           Tailwind entry CSS (compiled → styles.css)
```

---

# Requirements / Installs (Required)

## 1) Python
- **Python 3.11** recommended

Install Python packages (global install):
```bash
pip install -r requirements.txt
```

## 2) Node.js / npm
We use Tailwind for styling, so **Node.js + npm are required** for setup (and for rebuilding CSS when styles change).

- Install **Node.js (LTS)** (includes npm)

Install node packages:
```bash
npm install
```

---

# Setup Instructions

## A) Install Python dependencies
```bash
pip install -r requirements.txt
```

## B) Tailwind setup

1) Install node deps:
```bash
npm install
```

2) Ensure Tailwind input exists at:
`src/app/tailwind/input.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

3) Build CSS:
```bash
npm run build:css
```

Your `package.json` should include:
```json
"scripts": {
  "build:css": "tailwindcss -i ./src/app/tailwind/input.css -o ./src/app/static/styles.css --minify"
}
```

---

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

Outputs:
- `models/fruit_model.pt`
- `models/classes.txt`

## Run the web app (development)
```bash
python -m uvicorn src.app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/app`

## How it Works (simple)

### Training phase (run occasionally)
- Train + evaluate using `src/training/`
- Export a model into `models/` (ex: `models/fruit_model.pt`)

### App phase (normal use / demo)
- No retraining needed
- Start FastAPI, it loads the saved model once
- UI sends an image to `POST /predict`
- API returns `{ fruit, freshness, confidence }` and UI displays it
