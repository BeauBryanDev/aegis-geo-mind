
import logging

import petrologix

from app.core.config import settings
from app.petrologix import predictor
from app.petrologix.loader import get_model
from app.schemas.predictions import LithologyShare, PredictionResponse
from app.utils import validator
from app.utils.las2csv_parser import parse_upload

logger = logging.getLogger(__name__)
"""Orchestrates a lithology prediction request.
"""

class InvalidWellLogError(ValueError):
    """Upload is missing curves the model requires."""


def lithology_shares(intervals) -> list[LithologyShare]:
    """Thickness share per lithology, descending. Feeds the pie and bar charts."""
    if not intervals:
        return []
    total = sum(i.thickness for i in intervals) or 1.0
    agg: dict[str, dict] = {}
    for iv in intervals:
        e = agg.setdefault(iv.lithology, {"code": iv.lithology_code, "t": 0.0, "wc": 0.0})
        e["t"] += iv.thickness
        e["wc"] += iv.confidence * iv.thickness
    shares = [
        LithologyShare(
            lithology=name,
            lithology_code=e["code"],
            thickness=round(e["t"], 2),
            fraction=round(e["t"] / total, 4),
            mean_confidence=round(e["wc"] / e["t"], 3) if e["t"] else 0.0,
        )
        for name, e in agg.items()
    ]
    return sorted(shares, key=lambda s: -s.thickness)


def predict_from_upload(
    path: str,
    filename: str,
    include_samples: bool = False,
    min_thickness: float | None = None,
    include_curves: bool = False,
) -> PredictionResponse:
    bundle = get_model()

    df = parse_upload(path, filename)
    curves = validator.build_curve_report(df, bundle)
    if curves.required_missing:
        raise InvalidWellLogError(
            f"missing required curve(s): {', '.join(curves.required_missing)}. "
            f"Required: {', '.join(bundle['required_curves'])}."
        )

    distribution = predictor.check_distribution(df, bundle)

    name = validator.well_name(df, fallback=filename)
    lo, hi = validator.depth_range(df)

    curve_data = None
    
    if include_curves:
        
        from app.services import well_log_service
        
        curve_data = well_log_service.build_curves(df)

    if distribution.status == "out_of_distribution":
        
        logger.warning("refusing prediction for %s: %s", name, distribution.message)
        
        return PredictionResponse(
            
            well_name=name, depth_range=(lo, hi), n_samples=len(df),
            curves=curves, distribution=distribution,
            intervals=[], samples=None, model_version=bundle["version"],
        )

    intervals, samples = predictor.predict(df, bundle, min_thickness)
    
    logger.info("predicted %s: %d samples -> %d intervals (%s)",
                
                name, len(samples), len(intervals), distribution.status)

    return PredictionResponse(
        
        well_name=name, depth_range=(lo, hi), n_samples=len(samples),
        curves=curves, distribution=distribution,
        intervals=intervals,
        samples=samples if include_samples else None,
        distribution_by_lithology=lithology_shares(intervals),
        curve_data=curve_data,
        model_version=bundle["version"],
    )


def get_model_info():
    
    return predictor.model_info(get_model())
