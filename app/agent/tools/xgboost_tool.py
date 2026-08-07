
import logging
from pathlib import Path

from app.schemas.predictions import PredictionResponse
from app.services import prediction_service
from app.services.prediction_service import InvalidWellLogError
from app.utils.las2csv_parser import UnsupportedFormatError

logger = logging.getLogger(__name__)
"""Lithology prediction tool for the geologist agent.

Wraps the XGBoost well-log model as something an LLM can call. The heavy lifting
is in `app.services.prediction_service`; this module's job is to turn a
`PredictionResponse` into something a 7B model can actually read.
""" 
TOOL_NAME = "predict_well_lithology"

# Lithologies this model does NOT reliably detect (test recall < 0.1 across 17
# held-out wells: Chalk 0.000, Dolomite 0.003, Tuff 0.032, Marl 0.064).

UNRELIABLE_CLASSES = ["Chalk", "Tuff", "Marl", "Dolomite"]

# Confidence below this is flagged as uncertain. Calibration on the held-out set:
# >=0.9 -> 92% accurate, 0.7-0.9 -> ~70-75%, <0.7 -> 65% and falling.
LOW_CONFIDENCE = 0.65

LITHOLOGY_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Predict rock type (lithology) from an uploaded well log. Use this "
            "whenever the user asks what rock types are present in a well, where a "
            "particular lithology sits, how thick a unit is, or for a lithology "
            "summary of an uploaded log. "
            "Requires the curves GR, RDEP, RMED, RHOB, NPHI, DTC, CALI and DRHO. "
            "Trained on 118 Norwegian Continental Shelf wells; it reliably "
            "distinguishes Shale, Sandstone, Limestone and Halite, and does NOT "
            "reliably detect Chalk, Tuff, Marl or Dolomite."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the uploaded well log (.las or .csv).",
                },
                "min_thickness_m": {
                    "type": "number",
                    "description": (
                        "Ignore zones thinner than this, in metres. Default 0.5. "
                        "Raise it (e.g. 5) when the user wants only major units."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
}


def summarize_prediction(response: PredictionResponse, max_zones: int = 8) -> str:
    """Render a PredictionResponse as a compact, LLM-readable summary.

    Kept near 200 tokens: totals, thickest zones, confidence, blind spots.
    """
    lo, hi = response.depth_range

    # The input check comes first: if the well is unlike the training data, every
    # number below it is untrustworthy, so return only the refusal.
    if response.distribution.status == "out_of_distribution":
        
        return (f"Well {response.well_name}: prediction REFUSED. "
                f"{response.distribution.message}")

    lines = [
        f"Well {response.well_name} — logged {lo:.1f}–{hi:.1f} m MD "
        f"({response.n_samples} samples, {len(response.intervals)} zones)."
    ]
    if response.distribution.status == "warning":
        lines.append(f"CAUTION: {response.distribution.message}")

    if not response.intervals:
        lines.append("No lithology zones were resolved.")
        return "\n".join(lines)

    total = sum(i.thickness for i in response.intervals) or 1.0

    totals: dict[str, float] = {}
    for iv in response.intervals:
        totals[iv.lithology] = totals.get(iv.lithology, 0.0) + iv.thickness

    lines.append("\nLithology totals:")
    
    for name, thick in sorted(totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<17} {thick:8.1f} m ({thick / total:5.1%})")

    thickest = sorted(response.intervals, key=lambda i: -i.thickness)[:max_zones]
    lines.append(f"\nThickest zones (top {len(thickest)}):")
    
    for iv in sorted(thickest, key=lambda i: i.top):
        
        lines.append(
            f"  {iv.top:7.1f}–{iv.base:7.1f} m {iv.thickness:7.1f} m "
            f"{iv.lithology:<17} conf {iv.confidence:.2f}"
        )

    weighted_conf = sum(i.confidence * i.thickness for i in response.intervals) / total
    # Count how much of the column is below the low-confidence threshold, for the
    # agent to warn the user that the model is uncertain about a large fraction of
    # the well.
    low = sum(i.thickness for i in response.intervals if i.confidence < LOW_CONFIDENCE)
    
    lines.append(
        f"\nConfidence {weighted_conf:.2f} mean (thickness-weighted); "
        f"{low / total:.0%} of the column is below {LOW_CONFIDENCE} and should be "
        f"treated as uncertain."
    )
    lines.append(
        "Model blind spots — NOT reliably detected, so do not state their "
        f"absence as fact: {', '.join(UNRELIABLE_CLASSES)}."
    )
    return "\n".join(lines)


def run_lithology_tool(
    file_path: str,
    min_thickness_m: float | None = None,
    filename: str | None = None,
) -> str:
    """Execute the tool. Always returns text -|| -never raises.

    An agent loop that receives an exception loses the turn; a sentence lets the
    model recover ("that file is missing GR, ask the user to re-export").
    """
    path = Path(file_path)
    
    if not path.exists():
        return f"Cannot predict: no file at {file_path}."

    try:
        response = prediction_service.predict_from_upload(
            str(path),
            filename or path.name,
            include_samples=False,          # never send ~10k rows to an LLM
            min_thickness=min_thickness_m,
        )
    except InvalidWellLogError as e:
        return f"Cannot predict: {e}"
    
    except UnsupportedFormatError as e:
        return f"Cannot predict: {e}"
    
    except Exception:
        logger.exception("lithology tool failed for %s", file_path)
        return ("Cannot predict: the lithology model failed on this file. "
                "The log may be malformed.")

    return summarize_prediction(response)


def as_langchain_tool():
    """Wrap the tool for LangChain / LangGraph.

    Imported lazily so this module stays usable without an agent framework
    installed -- the `/predict` route does not need one.
    """
    from langchain_core.tools import tool

    @tool(TOOL_NAME, description=LITHOLOGY_TOOL_SPEC["function"]["description"])
    
    def predict_well_lithology(file_path: str, min_thickness_m: float = 0.5) -> str:
        
        return run_lithology_tool(file_path, min_thickness_m=min_thickness_m)

    return predict_well_lithology
