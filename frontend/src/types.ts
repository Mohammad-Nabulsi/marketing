export type ValidationIssue = {
  type: string;
  message: string;
  column?: string | null;
  count?: number | null;
  examples?: unknown[] | null;
};

export type ValidationReport = {
  ok: boolean;
  dataset_rows: number;
  dataset_columns: number;
  missing_required_columns: string[];
  issues: ValidationIssue[];
};

export type UploadResponse = {
  dataset_id: string;
  validation_report: ValidationReport;
};

export type PipelineStepStatus = {
  step: string;
  ok: boolean;
  message: string;
  output_files: string[];
};

export type PipelineSummary = {
  dataset_id: string;
  ok: boolean;
  message: string;
  steps: PipelineStepStatus[];
  outputs_dir: string;
};

export type DashboardResponse = {
  dataset_id: string;
  data: Record<string, any>;
};

