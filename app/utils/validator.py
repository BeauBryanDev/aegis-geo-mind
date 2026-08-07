
from pathlib import Path

import pandas as pd

from app.schemas.predictions import CurveReport
"""
Validate an uploaded well log against the model's curve contract.

The contract is read from the loaded model, never hardcoded -- so retraining with
a different curve set does not require a code change here or in the frontend.
"""

def build_curve_report(df: pd.DataFrame, bundle: dict) -> CurveReport:
    """Compare the upload's columns against required/optional curves."""
    cols = set(df.columns)
    
    required = bundle["required_curves"]
    optional = bundle["optional_curves"]

    # A curve that is present but entirely null is effectively missing.
    def present(curve: str) -> bool:
        
        return curve in cols and bool(df[curve].notna().any())

    return CurveReport(
        detected=sorted(cols),
        required_present=[c for c in required if present(c)],
        required_missing=[c for c in required if not present(c)],
        optional_present=[c for c in optional if present(c)],
    )


def depth_range(df: pd.DataFrame) -> tuple[float, float]:
    """Logged depth interval.

    DEPT wins when a file carries both DEPT and DEPTH_MD -- every LAS file in this
    dataset does, and DEPTH_MD there is a float-roundoff duplicate that is
    sometimes partially empty. Reading DEPTH_MD blindly gives a range that
    disagrees with the predictions, which resolve the collision the same way.
    """
    col = "DEPT" if "DEPT" in df.columns else "DEPTH_MD"
    
    s = df[col].dropna()
    
    return (float(s.min()), float(s.max())) if len(s) else (0.0, 0.0)


def well_name(df: pd.DataFrame, fallback: str) -> str:
    """Well name from the data, falling back to the filename without its suffix."""
    if "WELL" in df.columns:
        
        vals = df["WELL"].dropna().unique()
        
        if len(vals):
            
            return str(vals[0])
        
    return Path(fallback).stem
