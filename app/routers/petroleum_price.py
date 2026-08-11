from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.market_data.cache import get_oil_prices
from app.market_data.client import EIAClientError
from app.schemas.market import OilPriceResponse

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/oil-prices", response_model=OilPriceResponse)
def get_oil_prices_endpoint() -> OilPriceResponse:
    """Return current WTI/Brent spot prices, served from a 1-hour cache.
    Consumed by the frontend header (SEC status bar)."""
    try:
        
        data = get_oil_prices()
        
    except EIAClientError as exc:
        
        raise HTTPException(
            
            status_code=503, detail=f"EIA data unavailable: {exc}"
        ) from exc

    return OilPriceResponse(
        wti_usd=data["wti_usd"],
        wti_date=data["wti_date"],
        brent_usd=data["brent_usd"],
        brent_date=data["brent_date"],
        fetched_at=datetime.fromtimestamp(data["fetched_at"], tz=timezone.utc),
        source="EIA",
    )