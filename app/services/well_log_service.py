
import logging
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.schemas.well_logs import CurvePoint, SampleWell, WellCurves
from app.utils.las2csv_parser import parse_upload

logger = logging.getLogger(__name__)
"""Loads well logs and prepares curve data for the UI."""

PLOT_CURVES = ["GR", "RDEP", "RHOB", "NPHI", "DTC"]

# Held-out test wells, so the demo shows generalization rather than memorised
# training data. Chosen for contrast: one sandstone-rich, one shale/evaporite.
SAMPLE_WELLS = [
    SampleWell(
        id="25_5-1",
        well_name="25/5-1",
        label="North Sea 25/5-1",
        description="Sandstone Reservoir",
        depth_range=(145.0, 3432.0),
    ),
    SampleWell(
        id="17_11-1",
        well_name="17/11-1",
        label="North Sea 17/11-1",
        description="Shale / Evaporite Sequence",
        depth_range=(148.0, 3269.0),
    ),
]


class SampleWellNotFound(ValueError):
    pass


def list_samples() -> list[SampleWell]:
    return SAMPLE_WELLS


def sample_path(sample_id: str) -> Path:
    """Resolve a sample id to a LAS file on disk."""
    match = next((s for s in SAMPLE_WELLS if s.id == sample_id), None)
    
    if match is None:
        
        raise SampleWellNotFound(f"unknown sample well: {sample_id}")
    
    path = Path(settings.las_sample_dir) / f"{sample_id}.las"
    
    if not path.exists():
        
        raise SampleWellNotFound(
            f"sample LAS not found at {path}. Set LAS_SAMPLE_DIR in .env."
        )
        
    return path


def build_curves(df: pd.DataFrame, max_points: int = 600) -> WellCurves:
    """Downsample curves to a size the SVG chart can render and the wire can carry.

    A full well is ~10-20k samples; the chart is 230px tall, so 600 points is
    already more resolution than the display has.
    """
    depth_col = "DEPT" if "DEPT" in df.columns else "DEPTH_MD"
    
    available = [c for c in PLOT_CURVES if c in df.columns and df[c].notna().any()]

    frame = df[[depth_col] + available].dropna(subset=[depth_col])
    
    step = max(1, len(frame) // max_points)
    
    frame = frame.iloc[::step]

    points = [
        CurvePoint(
            depth=round(float(row[depth_col]), 2),
            **{c: (None if pd.isna(row[c]) else round(float(row[c]), 3)) for c in available},
        )
        for _, row in frame.iterrows()
    ]
    depths = df[depth_col].dropna()
    
    return WellCurves(
        
        well_name=str(df["WELL"].dropna().iloc[0]) if "WELL" in df.columns and df["WELL"].notna().any() else "",
        depth_range=(float(depths.min()), float(depths.max())),
        n_source_samples=len(df),
        points=points,
        available_curves=available,
    )


def load_sample(sample_id: str) -> tuple[pd.DataFrame, SampleWell]:
    """Parse a bundled sample well."""
    meta = next(s for s in SAMPLE_WELLS if s.id == sample_id)
    
    path = sample_path(sample_id)
    
    logger.info("loading sample well %s from %s", sample_id, path)
    
    return parse_upload(path, path.name), meta
