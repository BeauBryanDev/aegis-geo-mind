
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.agent import graph
from app.schemas.chats import ChatRequest, ChatResponse, SessionInfo
from app.schemas.predictions import PredictionResponse

logger = logging.getLogger(__name__)
"""Chat orchestration and per-session well context.
"""
SESSION_TTL_SECONDS = 60 * 60 * 4      # 4 hours
MAX_SESSIONS = 500


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    lithology_summary: str | None = None
    well_name: str | None = None
    n_intervals: int | None = None
    dominant_lithology: str | None = None
    mean_confidence: float | None = None


class SessionStore:
    """In-memory session cache with TTL and a hard size cap."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {} # session_id -> state

    def _evict(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items()
                   if now - v.created_at > SESSION_TTL_SECONDS]
        
        for k in expired:
            
            del self._sessions[k]
        # Hard cap as a backstop against a burst of sessions inside the TTL.
        while len(self._sessions) > MAX_SESSIONS:
            
            oldest = min(self._sessions, key=lambda k: self._sessions[k].created_at)
            
            del self._sessions[oldest]

    def get(self, session_id: str | None) -> SessionState | None:
        
        if not session_id:
            
            return None
        
        self._evict()
        
        return self._sessions.get(session_id)

    def create(self, session_id: str | None = None) -> SessionState:
        
        self._evict()
        
        sid = session_id or uuid.uuid4().hex[:16]
        state = self._sessions.get(sid) or SessionState(session_id=sid)
        self._sessions[sid] = state
        
        return state

    def attach_prediction(
        self, session_id: str, prediction: PredictionResponse, summary: str
    ) -> SessionState:
        """Record a prediction so later chat turns can refer to it."""
        state = self.create(session_id)
        state.lithology_summary = summary
        state.well_name = prediction.well_name
        state.n_intervals = len(prediction.intervals)
        
        if prediction.intervals:
            
            total = sum(i.thickness for i in prediction.intervals) or 1.0
            by_lith: dict[str, float] = {}
            
            for iv in prediction.intervals:
                
                by_lith[iv.lithology] = by_lith.get(iv.lithology, 0.0) + iv.thickness
            state.dominant_lithology = max(by_lith, key=by_lith.get)
            state.mean_confidence = round(
                sum(i.confidence * i.thickness for i in prediction.intervals) / total, 3
            )
        logger.info("session %s: attached %s (%d zones)",
                    session_id, state.well_name, state.n_intervals or 0)
        
        return state


    def info(self, session_id: str) -> SessionInfo | None:
        state = self.get(session_id)
        
        if not state:
            
            return None
        
        return SessionInfo(
            session_id=state.session_id,
            has_well_context=state.lithology_summary is not None,
            well_name=state.well_name,
            n_intervals=state.n_intervals,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state.created_at)),
        )


store = SessionStore()


async def chat(request: ChatRequest) -> ChatResponse:
    """Answer one chat turn, using the session's well context when present."""
    state = store.get(request.session_id) or store.create(request.session_id)

    result = await graph.run(
        question=request.message,
        history=[t.model_dump() for t in request.history],
        prior_lithology_summary=state.lithology_summary,
    )

    return ChatResponse(
        answer=result.answer,
        session_id=state.session_id,
        llm_used=result.llm_used,
        has_well_context=state.lithology_summary is not None,
        warnings=result.warnings,
        dominant_lithology=state.dominant_lithology,
        mean_confidence=state.mean_confidence,
        trace=result.trace,
    )
