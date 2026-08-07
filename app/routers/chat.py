
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.chats import ChatRequest, ChatResponse, SessionInfo
from app.services import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])
"""Chat endpoints for the geologist assistant."""

# The chat endpoint is the main entry point for the geologist assistant. It is a thin wrapper around the chat_service, which orchestrates the LLM and lithology model calls. The service handles session state, well context, and error handling.
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask the geologist assistant a question.

    Pass the `session_id` returned by `POST /predict?session_id=...` to ask about
    a well log analysed earlier; without it the assistant has no well context and
    answers from the fine-tuned model's geology knowledge alone.

    Always returns 200 when the request is well-formed. If the LLM Space is
    asleep, `llm_used` is false and `answer` carries the raw model summary rather
    than failing the request.
    """
    try:
        return await chat_service.chat(request)
    
    except Exception:
        
        logger.exception("chat failed")
        raise HTTPException(500, "chat failed")

# The session info endpoint is not documented because it is not used by the
# demo frontend. It is used by the agent to remember the well context.
@router.get("/chat/session/{session_id}", response_model=SessionInfo)
def session_info(session_id: str) -> SessionInfo:
    """What the assistant currently remembers about a session."""
    info = chat_service.store.info(session_id)
    
    if info is None:
        
        raise HTTPException(404, f"unknown or expired session: {session_id}")
    
    return info
