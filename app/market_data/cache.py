import logging
import time
from threading import Lock

from app.market_data.client import EIAClientError, fetch_oil_prices

logger = logging.getLogger(__name__)
 
TTL_SECONDS = 60 * 60  # 1 hour
 
_cache: dict = {"data": None, "fetched_at": 0.0}
_lock = Lock()
 
 

def get_oil_prices(force_refresh: bool = False) -> dict:
    """Return cached oil prices, refreshing from EIA if the TTL expired.
 
    """
    now = time.time()
    
    with _lock:
        
        is_stale = (now - _cache["fetched_at"]) > TTL_SECONDS
        if _cache["data"] is None or is_stale or force_refresh:
            
            try:
                
                fresh = fetch_oil_prices()
                
                _cache["data"] = fresh
                _cache["fetched_at"] = now
                logger.info("EIA oil prices refreshed")
                
            except EIAClientError as exc:
                
                if _cache["data"] is not None:
                    
                    logger.warning("EIA refresh failed, serving stale cache: %s", exc)
                else:
                    
                    raise
 
        return {**_cache["data"], "fetched_at": _cache["fetched_at"]}

## end of file 
