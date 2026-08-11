from datetime import datetime
 
from pydantic import BaseModel, Field
 
 
class OilPriceResponse(BaseModel):
    wti_usd: float = Field(..., description="WTI spot price, USD per barrel")
    wti_date: str = Field(
        ..., description="Date of the WTI value, as reported by EIA"
    )
    brent_usd: float = Field(..., description="Brent spot price, USD per barrel")
    brent_date: str = Field(
        ..., description="Date of the Brent value, as reported by EIA"
    )
    fetched_at: datetime = Field(
        ..., description="When this data was last fetched from EIA (or cache)"
    )
    source: str = "EIA"