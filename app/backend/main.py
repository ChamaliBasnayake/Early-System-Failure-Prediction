"""FastAPI application. Run from the project root:  uvicorn app.backend.main:app --reload --port 8000"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.routes import router, services
from app.backend.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(); services()        # load Drain state, texts and checkpoints once
    yield


app = FastAPI(lifespan=lifespan, title="BGL early failure prediction", version="0.1", description="Predicts kernel-level node failures on a BlueGene/L midplane from the last hour of logs, with an enforced lead time.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
