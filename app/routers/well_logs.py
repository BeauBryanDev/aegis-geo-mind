
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.predictions import PredictionResponse
from app.schemas.well_logs import SampleWell
from app.services import prediction_service, well_log_service
from app.services.well_log_service import SampleWellNotFound

logger = logging.getLogger(__name__)
router = APIRouter(tags=["well-logs"])
"""Sample well endpoints."""

@router.get("/wells/samples", response_model=list[SampleWell])
def list_samples() -> list[SampleWell]:
    """Wells the UI can load without an upload."""
    return well_log_service.list_samples()


@router.get("/wells/samples/{sample_id}/curves")
def sample_curves(sample_id: str):
    """Curve data for a sample well, downsampled for plotting."""
    try:
        df, _ = well_log_service.load_sample(sample_id)
        
    except SampleWellNotFound as e:
        raise HTTPException(404, str(e))
    
    return well_log_service.build_curves(df)


@router.get("/wells/samples/{sample_id}/analyze", response_model=PredictionResponse)
def analyze_sample(sample_id: str, session_id: str | None = None):
    """Predict lithology for a bundled sample well, with curves for the chart."""
    try:
        path = well_log_service.sample_path(sample_id)
        
    except SampleWellNotFound as e:
        
        raise HTTPException(404, str(e))

    result = prediction_service.predict_from_upload(
        str(path), path.name, include_samples=True,
        
        include_curves=True,
    )
    if session_id:
        from app.agent.tools.xgboost_tool import summarize_prediction
        from app.services import chat_service
        
        chat_service.store.attach_prediction(
            session_id, result, summarize_prediction(result)
        )
        
    return result
