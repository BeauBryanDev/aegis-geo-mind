"""Well log and curve schemas."""
from pydantic import BaseModel, Field


class CurvePoint(BaseModel):
    """One depth sample of the curves the HUD plots."""
    depth: float
    GR: float | None = None
    RDEP: float | None = None
    RHOB: float | None = None
    NPHI: float | None = None
    DTC: float | None = None


class SampleWell(BaseModel):
    """A bundled well the UI can load without an upload."""
    id: str
    well_name: str
    label: str
    description: str
    depth_range: tuple[float, float]


class WellCurves(BaseModel):
    """Downsampled curves for plotting."""
    well_name: str
    depth_range: tuple[float, float]
    n_source_samples: int
    points: list[CurvePoint]
    available_curves: list[str]
