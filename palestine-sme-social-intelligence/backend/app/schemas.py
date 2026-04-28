from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    type: str
    message: str
    column: Optional[str] = None
    count: Optional[int] = None
    examples: Optional[List[Any]] = None


class ValidationReport(BaseModel):
    ok: bool
    dataset_rows: int = 0
    dataset_columns: int = 0
    missing_required_columns: List[str] = Field(default_factory=list)
    issues: List[ValidationIssue] = Field(default_factory=list)


class UploadResponse(BaseModel):
    dataset_id: str
    validation_report: ValidationReport


class PipelineStepStatus(BaseModel):
    step: str
    ok: bool
    message: str
    output_files: List[str] = Field(default_factory=list)


class PipelineSummary(BaseModel):
    dataset_id: str
    ok: bool
    message: str
    steps: List[PipelineStepStatus] = Field(default_factory=list)
    outputs_dir: str


class DashboardResponse(BaseModel):
    dataset_id: str
    data: Dict[str, Any]

