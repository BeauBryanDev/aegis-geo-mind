
from fastapi import APIRouter

from app.core.config import settings
from app.petrologix.loader import get_model

router = APIRouter(tags=["health"])
"""Liveness and readiness."""

@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@router.get("/ready")
def ready() -> dict:
    """Ready means the lithology model is loaded and usable."""
    bundle = get_model()
    return {
        "status": "ready",
        "model_version": bundle["version"],
        "model_classes": len(bundle["classes_name"]),
    }
