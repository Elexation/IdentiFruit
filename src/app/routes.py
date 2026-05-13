import io
import logging
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from src.inference.predict import predict

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Define router and templates
router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")

# Home page
@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"request": request},
    )

# App page
@router.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="app.html",
        context={"request": request},
    )

# API endpoint for image prediction
@router.post("/predict")
def predict_route(file: UploadFile = File(...)):
    contents = file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict(img)
        return result

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not found.")

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
