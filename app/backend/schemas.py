from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    log_text: str = Field(..., description="Raw BGL log lines (the last ~60 minutes of the system log)")
    unit: Optional[str] = Field(None, description="Midplane id, e.g. R02-M1. If omitted, every midplane present in the input is scored.")
    lead_time: int = Field(5, description="Lead time N in minutes (1, 5, 15, 30)")
    mc_samples: Optional[int] = Field(None, description="Monte-Carlo dropout passes (default from settings; 0 = deterministic)")


class LineOut(BaseModel):
    timestamp: str; node: str; unit: str; template_index: int; message: str


class UnitResult(BaseModel):
    prediction_id: int; unit: str; score: float; mc_spread: Optional[float]; mc_samples: int
    alarm_at_far: dict; thresholds: dict; recommendation: str
    n_local: int; n_global: int; unk_rate_local: float; unk_rate_global: float; n_new_texts: int; out_of_distribution: bool
    new_texts_embedded: int = 0; encoder: str = ""
    window_start: str; window_end: str; local_lines: list[LineOut]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_run: str; model_version: str; lead_time_requested: int; lead_time_used: int; fallback_lead_time: bool
    n_input_lines: int; n_parsed: int; n_malformed: int; units_detected: list[str]; latency_ms: float
    score_meaning: str; limitations: list[str]; results: list[UnitResult]


class FeedbackRequest(BaseModel):
    prediction_id: int; correct: bool; note: Optional[str] = None
