import io
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from src.inference.predict import predict

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
async def predict_route(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict(img)
        return result

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(result)