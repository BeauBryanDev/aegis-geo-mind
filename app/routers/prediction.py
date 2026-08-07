
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.schemas.predictions import ModelInfo, PredictionResponse
from app.services import prediction_service
from app.services.prediction_service import InvalidWellLogError
from app.utils.las2csv_parser import UnsupportedFormatError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])
"""Lithology prediction endpoints."""

@router.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Model version, curve contract, class list and held-out metrics.
    """
    return prediction_service.get_model_info()


@router.post("/predict", response_model=PredictionResponse)

async def predict(
    
    file: UploadFile = File(..., description="Well log, .las or .csv"),
    
    include_samples: bool = Query(
        
        False, description="Include per-depth rows for plotting (~10k). "
                           "Never pass these to an LLM."),
    min_thickness_m: float | None = Query(
        None, ge=0, description="Minimum zone thickness in metres."),
    include_curves: bool = Query(
        False, description="Include downsampled curve data for the log-track chart."),
    session_id: str | None = Query(
        None, description="Attach this prediction to a chat session so follow-up "
                          "questions at POST /chat can refer to the well without "
                          "re-uploading it."),
) -> PredictionResponse:
    
    """Predict lithology for an uploaded well log.

    Returns depth intervals. An out-of-distribution well returns an empty
    `intervals` list with `distribution.status == "out_of_distribution"` and an
    explanatory message -- a 200, not an error, so the agent can relay the reason.
    """
    suffix = Path(file.filename or "upload").suffix or ".csv"
    max_bytes = settings.max_upload_mb * 1024 * 1024

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
            tmp.write(chunk)
            
        tmp_path = tmp.name

    try:
        result = prediction_service.predict_from_upload(
            tmp_path, file.filename or "upload",
            include_samples=include_samples, min_thickness=min_thickness_m,
            include_curves=include_curves,
        )
        if session_id:
            # Cache the LLM-sized summary so /chat can answer follow-ups without
            # re-running the model or asking the user to upload again.
            from app.agent.tools.xgboost_tool import summarize_prediction
            from app.services import chat_service
            chat_service.store.attach_prediction(
                session_id, result, summarize_prediction(result)
            )
        return result
    except UnsupportedFormatError as e:
        raise HTTPException(415, str(e))
    
    except InvalidWellLogError as e:
        raise HTTPException(422, str(e))
    
    except Exception:
        logger.exception("prediction failed for %s", file.filename)
        raise HTTPException(500, "prediction failed")
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)
