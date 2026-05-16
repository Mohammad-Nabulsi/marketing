"""
API router for the Temporal Intelligence backend.
"""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from schemas import (
    AnomalyResponse,
    BusinessMomentumResponse,
    ForecastResponse,
    UploadedDatasetRequest,
)
from temporal_analysis_adapter import (
    build_frontend_response,
    get_sector_momentum,
    get_sector_anomaly_summary,
    run_anomaly_pipeline,
    run_business_momentum_pipeline,
    run_forecasting_pipeline,
)

router = APIRouter(prefix="/api")


def _handle_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{type(error).__name__}: {str(error)}",
    )


@router.post(
    "/business-momentum/compare-with-sector",
    response_model=BusinessMomentumResponse,
    status_code=status.HTTP_200_OK,
)
def compare_with_sector(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return build_frontend_response(
            run_business_momentum_pipeline(request.uploaded_file_path)
        )
    except Exception as exc:
        raise _handle_error(exc)





@router.get("/sectors/momentum", status_code=status.HTTP_200_OK)
def get_sectors_momentum() -> List[Dict[str, Any]]:
    try:
        return get_sector_momentum()
    except Exception as exc:
        raise _handle_error(exc)


@router.post(
    "/anomalies/analyze",
    response_model=AnomalyResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_anomalies(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return build_frontend_response(
            run_anomaly_pipeline(request.uploaded_file_path)
        )
    except Exception as exc:
        raise _handle_error(exc)



@router.get("/anomalies/sector-summary", status_code=status.HTTP_200_OK)
def anomaly_sector_summary(sector: str) -> List[Dict[str, Any]]:
    try:
        return get_sector_anomaly_summary(sector)
    except Exception as exc:
        raise _handle_error(exc)


@router.post(
    "/forecast/analyze",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_forecast(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return build_frontend_response(
            run_forecasting_pipeline(request.uploaded_file_path)
        )
    except Exception as exc:
        raise _handle_error(exc)



