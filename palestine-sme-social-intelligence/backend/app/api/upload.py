from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.schemas import UploadResponse
from app.services.validation import validate_dataframe
from app.utils.file_utils import ensure_dir, safe_read_csv, write_json


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    dataset_id = str(uuid.uuid4())
    storage = settings.storage_path()

    raw_dir = ensure_dir(storage / "raw" / dataset_id)
    raw_path = raw_dir / "raw.csv"

    try:
        content = await file.read()
        raw_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        df = safe_read_csv(raw_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    report = validate_dataframe(df)

    reports_dir = ensure_dir(storage / "reports" / dataset_id)
    write_json(reports_dir / "validation_report.json", report.model_dump())

    return UploadResponse(dataset_id=dataset_id, validation_report=report)

