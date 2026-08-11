from langchain.tools import tool
 
from app.mcp.client import get_oil_prices
 
 
@tool
def get_current_oil_prices() -> str:
    """Get the current WTI and Brent crude oil spot prices in USD per
    barrel. Use this when the user asks about oil prices, market
    conditions, or wants to relate a well's economics to current
    commodity prices."""
    prices = get_oil_prices()
    return (
        f"WTI: ${prices['wti_usd']:.2f}/bbl (as of {prices['wti_date']}), "
        f"Brent: ${prices['brent_usd']:.2f}/bbl (as of {prices['brent_date']})"
    )
 