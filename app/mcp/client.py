from app.market_data.cache import get_oil_prices as _get_oil_prices
 
 
def get_oil_prices() -> dict:
    """Facade entry point for oil price data.
 
    See app/market_data/cache.py for the actual caching/fetch logic.
    """
    return _get_oil_prices()
 
 
# --- Placeholder for future real MCP integrations -------------------------
# def get_onepetro_data(...) -> dict:
#     """Will connect to a real MCP server exposing OnePetro data once
#     available. Not implemented yet."""
#     raise NotImplementedError("OnePetro MCP integration pending")
 
 
 