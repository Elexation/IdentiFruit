from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router

app = FastAPI()

# Serve static files (app.js + styles.css)
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

app.include_router(router)
