
# IdentiFruit 🍓🍌🍎

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

## Project Structure (with descriptions)

> Note: `node_modules/` is not included in the repo (generated locally by npm).

```txt
IdentiFruit/
├─ package.json                 # npm deps + scripts (Tailwind build)
├─ package-lock.json            # exact npm dependency versions (team consistency)
├─ tailwind.config.js           # tells Tailwind what files to scan for class names
├─ .gitignore                   # ignores node_modules, etc.
├─ requirements.txt             # Python dependencies for training + API
├─ README.md                    # project overview + setup instructions
│
├─ data/                        # dataset images (usually not committed)
│  ├─ Apple/                    # apple images
│  ├─ Banana/                   # banana images
│  └─ Strawberry/               # strawberry images
│
├─ models/                      # exported trained model files (ex: fruit_model.pt)
├─ notebooks/                   # optional experiments / quick checks
│
├─ tests/
│  └─ test_predict.py           # sanity test for inference predict()
│
└─ src/
   ├─ training/                 # scripts used to train/evaluate/export the model
   │  ├─ dataset.py             # dataset loader (images → tensors + labels)
   │  ├─ train.py               # training loop (learn model weights)
   │  ├─ eval.py                # evaluation/metrics (accuracy, confusion matrix)
   │  └─ export.py              # saves trained model into /models
   │
   ├─ inference/                # code used during runtime (no training)
   │  ├─ model.py               # loads saved model + preprocessing transforms
   │  └─ predict.py             # predict(image) -> {fruit, freshness, confidence}
   │
   └─ app/                      # FastAPI server + frontend files
      ├─ main.py                # FastAPI app creation + startup
      ├─ routes.py              # routes: /, /app, /predict
      ├─ tailwind/
      │  └─ input.css           # Tailwind input file (@tailwind base/components/utilities)
      ├─ static/
      │  ├─ app.js              # browser JS (camera/upload + fetch(/predict))
      │  └─ styles.css          # compiled Tailwind output (committed)
      └─ templates/
         ├─ home.html           # landing page
         └─ app.html            # camera/upload page
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
