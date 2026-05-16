"""
FastAPI entrypoint for the Temporal Intelligence backend.
"""
from fastapi import FastAPI

from api_router import router

app = FastAPI(
    title="Palestine SME Temporal Intelligence API",
    version="1.0",
    description="API layer for business momentum, anomaly detection, and forecasting.",
)

app.include_router(router)
