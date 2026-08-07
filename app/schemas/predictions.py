
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.well_logs import WellCurves

DistributionStatus = Literal["ok", "warning", "out_of_distribution", "unknown"]


class LithologyInterval(BaseModel):
    """One predicted lithology zone."""
    top: float = Field(description="Top of the zone, metres MD")
    base: float = Field(description="Base of the zone, metres MD")
    thickness: float
    lithology_code: int = Field(description="FORCE 2020 numeric code, e.g. 65000")
    lithology: str = Field(description="Human-readable name, e.g. 'Shale'")
    confidence: float = Field(ge=0, le=1, description="Mean confidence across the zone")
    n_samples: int


class SampleRow(BaseModel):
    """One depth sample -- for plotting only, never for an LLM context."""
    depth: float
    lithology_code: int
    confidence: float


class DistributionCheck(BaseModel):
    """Whether the upload resembles the training data."""
    status: DistributionStatus
    message: str
    flagged_curves: list[str] = []
    missing_curves: list[str] = []


class CurveReport(BaseModel):
    """What the uploaded file actually contains, against the model contract."""
    detected: list[str]
    required_present: list[str]
    required_missing: list[str]
    optional_present: list[str]


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    n_features: int
    classes: list[str]
    required_curves: list[str]
    optional_curves: list[str]
    test_accuracy: float
    test_weighted_f1: float
    xgboost_version: str
    sklearn_version: str


class LithologyShare(BaseModel):
    """Thickness share of one lithology, for the pie and bar charts."""
    lithology: str
    lithology_code: int
    thickness: float
    fraction: float
    mean_confidence: float


class PredictionResponse(BaseModel):
    well_name: str
    depth_range: tuple[float, float]
    n_samples: int
    curves: CurveReport
    distribution: DistributionCheck
    intervals: list[LithologyInterval]
    samples: list[SampleRow] | None = Field(
        default=None,
        description="Per-depth predictions for plotting. Omitted unless "
                    "include_samples=true. Never send this to an LLM.",
    )
    distribution_by_lithology: list[LithologyShare] = Field(
        default_factory=list,
        description="Thickness share per predicted lithology, descending.",
    )
    curve_data: WellCurves | None = Field(
        default=None,
        description="Downsampled curve data for the log-track chart.",
    )
    model_version: str
