
import logging

import pandas as pd
import petrologix

from app.core.config import settings
from app.schemas.predictions import (
    DistributionCheck,
    LithologyInterval,
    ModelInfo,
    SampleRow,
)
"""[ADAPTER] Call the petrologix model.

All modelling logic lives in the wheel. This module only orchestrates the calls
and maps the result onto the response contract.
"""

logger = logging.getLogger(__name__)


def check_distribution(df: pd.DataFrame, bundle: dict) -> DistributionCheck:
    """Is this well like the wells the model was trained on?"""
    report = petrologix.check_distribution(df, bundle=bundle)
    
    return DistributionCheck(
        
        status=report["status"],
        message=report["message"],
        flagged_curves=report.get("flagged_required", []),
        missing_curves=report.get("missing_required", []),
    )


def predict(df: pd.DataFrame, bundle: dict, min_thickness: float | None = None):
    """Run the model and collapse to intervals.

    Returns (intervals, samples). `samples` is one row per depth and is for the
    log-track plot only -- it must never be sent to an LLM (~10,000 rows).
    """
    min_thickness = (settings.min_interval_thickness_m
                     if min_thickness is None else min_thickness)

    result = petrologix.predict_lithology(df, bundle=bundle)
    zones = petrologix.to_intervals(result, min_thickness=min_thickness)

    intervals = [
        LithologyInterval(
            top=float(r.top), base=float(r.base), thickness=float(r.thickness),
            lithology_code=int(r.lithology_code), lithology=str(r.lithology_name),
            confidence=float(r.confidence), n_samples=int(r.n_samples),
        )
        for r in zones.itertuples()
    ]
    samples = [
        SampleRow(depth=float(d), lithology_code=int(c), confidence=float(p))
        
        for d, c, p in zip(result.DEPTH_MD, result.lithology_code, result.confidence)
    ]
    return intervals, samples


def model_info(bundle: dict) -> ModelInfo:
    
    info = petrologix.model_info(bundle=bundle)
    
    return ModelInfo(
        version=bundle["version"],
        trained_at=info["trained_at"],
        n_features=info["n_features"],
        classes=info["classes"],
        required_curves=info["required_curves"],
        optional_curves=info["optional_curves"],
        test_accuracy=info["test_metrics"]["accuracy"],
        test_weighted_f1=info["test_metrics"]["weighted_f1"],
        xgboost_version=info["xgboost_version"],
        sklearn_version=info["sklearn_version"],
    )
