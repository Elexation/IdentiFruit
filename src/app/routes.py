import io
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from src.inference.predict import predict

# Define router and templates
router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")

# Home page
@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# App page
@router.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

# API endpoint for image prediction
@router.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()

    # Convert bytes data to a PIL Image
    img = Image.open(io.BytesIO(contents))

    result = predict(img) # Get prediction result
    return JSONResponse(result) # Return result as JSON response
