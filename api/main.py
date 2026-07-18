from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.invoice import router as invoice_router
from db.session import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="VisionOCR — Invoice Intelligence Platform",
    description="Extract structured information from Vietnamese invoices using Qwen2.5-VL + LoRA",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(invoice_router)


@app.get("/")
async def root():
    return {
        "service": "VisionOCR API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
