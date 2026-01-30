from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.inference.predict import predict

router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})


@router.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    # read uploaded bytes -> PIL image
    contents = await file.read()
    img = Image.open(__import__("io").BytesIO(contents))

    result = predict(img)
    return JSONResponse(result)
