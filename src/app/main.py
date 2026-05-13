from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router
from src.inference.predict import _load


@asynccontextmanager
async def lifespan(app):
	_load()
	yield


app = FastAPI(lifespan=lifespan)

# Serve static files (app.js + styles.css)
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

app.include_router(router)
