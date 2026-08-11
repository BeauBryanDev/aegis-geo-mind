import httpx


from app.core.config import settings

EIA_BASE_URL =  "https://api.eia.gov/v2/petroleum/pri/spt/data"


WTI_SERIES = "RWTC"
BRENT_SERIES = "RBRTE"
 

class EIAClientError(Exception):
    """Raised when the EIA API returns an unexpected or empty response."""
 
 
def _fetch_series(series_id: str) -> tuple[float, str]:
    """Fetch the most recent value for a given EIA series_id.
 
    Returns:
        (price, period_date) — price in USD per barrel, period_date as
        reported by EIA (e.g. "2026-08-06").
    """
    
    params = {
    "api_key": settings.EIA_API_KEY,
    "frequency": "daily",
    "data[0]": "value",
    "facets[series][0]":  series_id,
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "length": 1,
    }
    
    
    # Wrap transport/HTTP failures so callers only have to handle EIAClientError.
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(EIA_BASE_URL, params=params)
            response.raise_for_status()
            payload = response.json()

    except httpx.HTTPStatusError as exc:
        raise EIAClientError(
            f"EIA returned HTTP {exc.response.status_code} for series_id={series_id}"
        ) from exc

    except httpx.HTTPError as exc:
        # exc's text can embed the request URL, which carries api_key — use the type only.
        raise EIAClientError(
            f"EIA request failed for series_id={series_id}: {type(exc).__name__}"
        ) from exc
 
    try:
        rows = payload["response"]["data"]
        
    except KeyError as exc:
        
        raise EIAClientError(f"Unexpected EIA response shape: {payload}") from exc
 
    if not rows:
        
        raise EIAClientError(f"EIA returned no data for series_id={series_id}")
 
    latest = rows[0]  # EIA returns most recent period first
    
    value = latest.get("value")
    period = latest.get("period")

    if value is None or period is None:
        
        raise EIAClientError(f"EIA returned incomplete row for series_id={series_id}: {latest}")
    
    return float(value), period
 
 
def fetch_oil_prices() -> dict:
    """
    Fetch current WTI and Brent spot prices from EIA.
 
    Raises EIAClientError on failure. The caller should catch and handle this
    fallback behavior — this function always makes a real network call.
    """
    wti_price, wti_date = _fetch_series(WTI_SERIES)
    brent_price, brent_date = _fetch_series(BRENT_SERIES)
    
    return {
        "wti_usd": wti_price,
        "wti_date": wti_date,
        "brent_usd": brent_price,
        "brent_date": brent_date,
    }