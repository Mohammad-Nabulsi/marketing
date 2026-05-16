"""
Pydantic schema definitions for the Temporal Intelligence backend.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


class BusinessMomentumResponse(BaseModel):
    business_name: str
    sector: str
    trend: str
    comparison_label: str
    latest_business_engagement_rate: Optional[float]
    latest_sector_engagement_rate: Optional[float]
    difference_from_sector: float
    message: str
    chart_url: str
    csv_outputs: List[str]


class AnomalyResponse(BaseModel):
    business_name: str
    sector: str
    message: str
    top_positive_anomalies: List[Dict[str, Any]]
    top_negative_anomalies: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    chart_url: Optional[str]
    csv_outputs: List[str]


class ForecastResponse(BaseModel):
    business_name: str
    best_model: str
    forecast_horizon_weeks: int
    best_MAE: float
    best_RMSE: float
    message: str
    future_forecast: List[Dict[str, Any]]
    chart_url: str
    csv_outputs: List[str]
