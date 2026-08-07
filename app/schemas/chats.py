from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.predictions import LithologyInterval

Role = Literal["user", "assistant"]
"""Chat request/response contract."""


class ChatTurn(BaseModel):
    """One prior turn, sent back by the client."""
    role: Role
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Ties this turn to a well log analysed earlier via "
                    "POST /predict?session_id=... . Without it the assistant has "
                    "no well context.",
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns. Only the last few are used -- the fine-tune has "
                    "a 1024-token window.",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    llm_used: bool = Field(
        description="False when the LLM Space was unavailable and the raw model "
                    "summary was returned instead."
    )
    has_well_context: bool = Field(
        description="Whether a lithology prediction informed this answer."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Model caveats and any corrections applied to the answer. "
                    "Surface these in the UI -- they are not decoration.",
    )
    # Convenience for the chat bubble (frontend ChatMessage.faciesContext /
    # confidenceContext) so it does not have to re-derive them from /predict.
    dominant_lithology: str | None = None
    mean_confidence: float | None = None
    
    trace: list[str] = Field(
        default_factory=list,
        description="Node path taken. Useful in development; hide in production.",
    )


class SessionInfo(BaseModel):
    session_id: str
    has_well_context: bool
    well_name: str | None = None
    n_intervals: int | None = None
    created_at: str
