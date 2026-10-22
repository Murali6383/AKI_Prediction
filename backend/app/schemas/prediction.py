from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float|None = Field(None, ge=0, le=120)
    sex: Literal["female","male","other","unknown"]|None = "unknown"
    creatinine: float|None = Field(None, ge=0, le=100)
    baseline_creatinine: float|None = Field(None, ge=0, le=100)
    bun: float|None = Field(None, ge=0, le=400)
    egfr: float|None = Field(None, ge=0, le=250)
    heart_rate: float|None = Field(None, ge=0, le=350)
    systolic_bp: float|None = Field(None, ge=0, le=350)
    diastolic_bp: float|None = Field(None, ge=0, le=250)
    sodium: float|None = Field(None, ge=50, le=250)
    potassium: float|None = Field(None, ge=1, le=15)
    bicarbonate: float|None = Field(None, ge=1, le=80)
    urine_output_ml_6h: float|None = Field(None, ge=0, le=10000)
    comorbidities: list[str] = Field(default_factory=list, max_length=30)
    @field_validator("comorbidities")
    @classmethod
    def clean_comorbidities(cls, values):
        return [v.strip().lower()[:60] for v in values if v and v.strip()]

class PredictionRequest(PatientFeatures):
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
class Observation(PatientFeatures):
    timestamp: datetime
class SequenceRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=500)
class ExplainRequest(PatientFeatures):
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: Literal["LOW","MODERATE","HIGH"]
    threshold: float
    model_name: str
    prediction_timestamp: datetime
    dataset_mode: str
    warnings: list[str] = []
