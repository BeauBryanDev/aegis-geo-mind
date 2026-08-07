
import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)
"""Loads the lithology model once, at application startup.
"""

_bundle = None


def load_model() -> dict:
    """Load and cache the petrologix bundle. Call once from the app lifespan."""
    global _bundle
    
    if _bundle is not None:
        
        return _bundle

    if settings.petrologix_models_dir:
        os.environ["PETROLOGIX_MODELS_DIR"] = settings.petrologix_models_dir

    import petrologix

    _bundle = petrologix.load_bundle()
    
    meta = _bundle["metadata"]
    
    logger.info(
        
        "lithology model loaded: version=%s features=%d classes=%d "
        "test_accuracy=%.4f xgboost=%s sklearn=%s",
        _bundle["version"], len(_bundle["feature_list"]), len(_bundle["classes_name"]),
        _bundle["metrics"]["test"]["accuracy"],
        
        meta["xgboost_version"], meta["sklearn_version"],
    )
    if meta.get("known_issue"):
        
        logger.warning("model known issue: %s", meta["known_issue"])
        
    return _bundle


def get_model() -> dict:
    """Return the loaded bundle. Raises if startup did not run."""
    if _bundle is None:
        raise RuntimeError("lithology model not loaded -- load_model() must run at startup")
    
    return _bundle
