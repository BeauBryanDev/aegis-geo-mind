
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from app.agent.prompts import GENERAL_SYSTEM_PROMPT, LITHOLOGY_RULES
from app.agent.tools.xgboost_tool import (
    UNRELIABLE_CLASSES,
    run_lithology_tool,
    summarize_prediction,
)
from app.llm.anthropic_client import LLMUnavailableError, get_llm_client
from app.schemas.predictions import PredictionResponse
from app.services import prediction_service
from app.services.prediction_service import InvalidWellLogError
from app.utils.las2csv_parser import UnsupportedFormatError

logger = logging.getLogger(__name__)

Route = Literal["lithology", "chat"]


@dataclass
class GeoMindState:
    """State passed between nodes. One instance per request."""

    # inputs
    question: str
    well_log_path: str | None = None
    well_log_filename: str | None = None
    history: list[dict] = field(default_factory=list)
    min_thickness_m: float | None = None
    # A summary carried over from an earlier turn. Lets follow-up questions reuse
    # a prediction instead of re-running it
    prior_lithology_summary: str | None = None

    # populated by nodes
    route: Route | None = None
    prediction: PredictionResponse | None = None
    lithology_summary: str | None = None
    rag_context: str | None = None          # reserved: app/rag/ is not built yet
    answer: str | None = None

    # observability
    llm_used: bool = False
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def note(self, step: str) -> None:
        self.trace.append(step)



# nodes
def route_node(state: GeoMindState) -> GeoMindState:
    """Decide which path the request takes.

    Deterministic on purpose: presence of an uploaded file is a fact the backend
    already has, not something to ask a 7B model to infer.
    """
    if state.well_log_path:
        state.route = "lithology"
    else:
        # No new file: reuse a summary from earlier in the session if there is one.
        state.lithology_summary = state.prior_lithology_summary
        state.route = "chat"
    state.note(f"route={state.route}")
    
    return state


def lithology_node(state: GeoMindState) -> GeoMindState:
    """Run the well-log model and produce both structured and text output.

    Structured (`prediction`) goes to the UI for the log track; the text summary
    (~250 tokens) is what the LLM sees.
    """
    try:
        state.prediction = prediction_service.predict_from_upload(
            state.well_log_path,
            state.well_log_filename or "upload",
            include_samples=True,        # the UI plots these; the LLM never sees them
            min_thickness=state.min_thickness_m,
        )
        state.lithology_summary = summarize_prediction(state.prediction)
        status = state.prediction.distribution.status
        state.note(f"lithology ok ({len(state.prediction.intervals)} zones, {status})")
        
        if status != "ok":
            state.warnings.append(state.prediction.distribution.message)
            
    except (InvalidWellLogError, UnsupportedFormatError) as e:
        # A rejected upload is a normal outcome, not a crash: tell the user what
        # is wrong with their file rather than failing the request.
        state.lithology_summary = f"Cannot predict: {e}"
        state.warnings.append(str(e))
        state.note("lithology rejected")
        
    except Exception:
        logger.exception("lithology node failed")
        state.lithology_summary = run_lithology_tool(state.well_log_path or "")
        state.note("lithology fallback")
        
    return state


def retrieve_node(state: GeoMindState) -> GeoMindState:
    """Geology RAG. Not built yet -- see app/rag/.

    Left as an explicit no-op node so the wiring point is obvious and adding
    Qdrant later does not require restructuring the flow.
    """
    state.note("retrieve=skipped (RAG not wired)")

    return state


async def generate_node(state: GeoMindState) -> GeoMindState:
    """Build system + user turns and call Claude.

    On LLM failure the tool summary is returned directly. The prediction is the
    expensive, trustworthy part of the answer; losing it because the API is down
    would be the wrong trade.
    """
    system = GENERAL_SYSTEM_PROMPT
    user_message = state.question.strip()

    if state.lithology_summary:
        
        system = f"{GENERAL_SYSTEM_PROMPT}\n\n{LITHOLOGY_RULES}"
        
        user_message = (
            "Lithology prediction for the uploaded well log:\n\n"
            f"{state.lithology_summary}\n\n"
            "This table has already been shown to the user, so do not repeat it.\n\n"
            f"Question: {user_message}"
        )

    state.note(f"prompt~{len(user_message) // 4}tok")

    try:
        state.answer = await get_llm_client().generate(
            user_message, system=system, history=state.history
        )
        state.llm_used = True
        state.note("llm ok")
        
    except LLMUnavailableError as e:
        
        logger.warning("LLM unavailable, falling back to the tool summary: %s", e)
        state.warnings.append(
            "The geologist assistant is unavailable, so the raw model output is "
            "shown below."
        )
        state.answer = state.lithology_summary or (
            "The geologist assistant is currently not available. Please try again."
        )
        state.note("llm unavailable -> fallback")

    return state


# output guard

_NEGATION = r"(?:no|not|none|without|absent|absence of|lacks?|free of|didn't|did not|isn't|is not|wasn't|was not)"
_ABSENCE_RE = [
    re.compile(rf"\b{_NEGATION}\b[^.]{{0,60}}\b{cls}\b", re.I) for cls in UNRELIABLE_CLASSES
] + [
    re.compile(rf"\b{cls}\b[^.]{{0,40}}\b(?:{_NEGATION}|absent)\b", re.I)
    for cls in UNRELIABLE_CLASSES
]

_BLIND_SPOT_NOTE_ONE = (
    "Correction: this model cannot detect {classes}. Its absence from the result "
    "is a limitation of the model, not evidence that it is absent from the well."
)


def find_blind_spot_claims(answer: str) -> list[str]:
    """Return blind-spot lithologies the answer wrongly claims are absent."""
    return [cls for cls in UNRELIABLE_CLASSES
            if any(rx.search(answer) for rx in _ABSENCE_RE if cls.lower() in rx.pattern.lower())]


# The model is given lithology, depth, thickness and confidence -- nothing else.
# It has no saturation, porosity, permeability, API gravity or net-pay data, so a
# NUMERIC claim about any of them is fabricated by construction.
_UNSUPPORTED_QUANTITIES = [
    "porosity", "permeability", "saturation", "api gravity", "net pay",
    "net-to-gross", "net to gross", "toc", "water cut", "pore pressure",
    "reserves", "flow rate", "viscosity",
]
# Require a digit within ~40 characters, so "the model does not report porosity"
# is not flagged while "17.0 porosity" is.
_QUANTITY_RE = [
    (q, re.compile(rf"(?:\d[\d.,]*\s*%?[^.]{{0,40}}\b{re.escape(q)}\b"
                   rf"|\b{re.escape(q)}\b[^.]{{0,40}}\d[\d.,]*)", re.I))
    for q in _UNSUPPORTED_QUANTITIES
]

_FABRICATION_NOTE = (
    "Correction: the lithology model provides only rock type, depth and "
    "confidence. It has no {quantities} data."
)

_FORMULA_REQUEST_RE = re.compile(
    r"\b(equation|formula|how (?:do you|is it) calculat(?:e|ing|ion)?|how to calculate|"
    r"derive|what is the (?:equation|formula)|worked example)\b",
    re.I,
)


def is_formula_request(question: str) -> bool:
    """True when the user asks for a general equation or derivation rather than
    an interpretation of this well's specific measured properties. Worked
    examples in a formula answer use illustrative numbers, not claims about
    the well, so the unsupported-quantity guard should not fire on them.
    """
    return bool(_FORMULA_REQUEST_RE.search(question))


def find_unsupported_quantities(answer: str) -> list[str]:
    """Return petrophysical quantities the answer quotes numbers for but cannot know."""
    return sorted({q for q, rx in _QUANTITY_RE if rx.search(answer)})


_CONFIDENCE_RE = re.compile(
    r"\b(?:confidence|probability|certainty)\b[^.\n]{0,30}?(\d\d?\d?(?:\.\d+)?)\s*(%?)"
    r"|(\d\d?\d?(?:\.\d+)?)\s*(%?)[^.\n]{0,20}?\b(?:confidence|probability|certainty)\b",
    re.I,
)
_SUMMARY_NUMBER_RE = re.compile(r"\b(0\.\d+|1\.0+)\b")

_NO_PREDICTION_NOTE = (
    "Correction: no well log has been analysed in this session, so the "
    "confidence figures above were not produced by the lithology model. "
    "Disregard them and upload a log to get a real prediction."
)
_MISMATCHED_CONFIDENCE_NOTE = (
    "Correction: the confidence values {values} do not appear in this well's "
    "prediction. Trust the measured result above, not these figures. "
)
_THRESHOLD_RE = re.compile(
    r"\b(?:below|above|under|over|less than|greater than|at least|at most|"
    r"exceeds?|threshold|beneath)\b",
    re.I,
)


def _confidence_values(answer: str) -> list[float]:
    """Every confidence-like figure in the answer, normalised to 0-1."""
    found: list[float] = []
    
    for m in _CONFIDENCE_RE.finditer(answer):
        
        if _THRESHOLD_RE.search(m.group(0)):
            continue
        
        raw, pct = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            value = float(raw)
            
        except (TypeError, ValueError):
            continue
        
        if pct == "%" or value > 1:
            value /= 100.0
            
        found.append(value)
        
    return found


def find_unsupported_confidence(
    answer: str, lithology_summary: str | None
) -> tuple[bool, list[str]]:
    """Check confidence claims against the figures the model was actually given.

    Returns (no_prediction_at_all, mismatched_values). Matching is exact to two
    decimals -- the precision the summary prints.
    """
    claimed = _confidence_values(answer)
    
    if not claimed:
        return False, []
    
    if not lithology_summary:
        return True, []

    shown = {round(float(m), 2) for m in _SUMMARY_NUMBER_RE.findall(lithology_summary)}
    mismatched = [f"{c:.2f}" for c in claimed if round(c, 2) not in shown]
    
    return False, sorted(set(mismatched))


def guard_node(state: GeoMindState) -> GeoMindState:
    """Append corrections for claims the answer cannot support."""
    if not state.answer or not state.llm_used:
        return state

    notes: list[str] = []

    no_prediction, mismatched = find_unsupported_confidence(
        state.answer, state.lithology_summary
    )
    if no_prediction:
        notes.append(_NO_PREDICTION_NOTE)
        state.note("guard: confidence claimed with no prediction")
        logger.error("model quoted confidence with no prediction in session")
        
    elif mismatched:
        notes.append(_MISMATCHED_CONFIDENCE_NOTE.format(values=", ".join(mismatched)))
        state.note(f"guard: mismatched confidence ({', '.join(mismatched)})")
        logger.error("model quoted confidence %s absent from prediction", mismatched)

    if not state.lithology_summary:
        
        if notes:
            state.answer = "\n\n".join([state.answer.rstrip(), *notes])
            state.warnings.extend(notes)
            
        return state

    claimed = find_blind_spot_claims(state.answer)
    
    _BLIND_SPOT_NOTE_ONE = (
        "Correction: this model cannot detect {classes}. Its absence from the result "
        "is a limitation of the model, not evidence that it is absent from the well."
    )
    _BLIND_SPOT_NOTE_MANY = (
        "Correction: this model cannot detect {classes}. Their absence from the "
        "result is a limitation of the model, not evidence that they are absent "
        "from the well."
    )
    if claimed:
        
        template = _BLIND_SPOT_NOTE_ONE if len(claimed) == 1 else _BLIND_SPOT_NOTE_MANY
        notes.append(template.format(classes=", ".join(claimed)))
        state.note(f"guard: corrected absence claim ({', '.join(claimed)})")
        logger.warning("model claimed %s absent; correction appended", claimed)

    if not is_formula_request(state.question):
        
        invented = find_unsupported_quantities(state.answer)
        
        if invented:
            notes.append(_FABRICATION_NOTE.format(quantities=", ".join(invented)))
            state.note(f"guard: flagged fabricated quantities ({', '.join(invented)})")
            logger.error("model quoted unsupported quantities %s -- fabricated", invented)
    else:
        state.note("guard: skipped quantity check (formula request)")

    if notes:
        state.answer = "\n\n".join([state.answer.rstrip(), *notes])
        state.warnings.extend(notes)

    return state


# graph
@dataclass
class GeoMindResult:
    """What a caller (router/service) gets back."""

    answer: str
    prediction: PredictionResponse | None
    llm_used: bool
    warnings: list[str]
    trace: list[str]
    # Carried out so the caller can cache it and reuse it on follow-up turns
    # without re-running the model or asking the user to re-upload.
    lithology_summary: str | None = None


async def run(
    question: str,
    well_log_path: str | None = None,
    well_log_filename: str | None = None,
    history: list[dict] | None = None,
    min_thickness_m: float | None = None,
    prior_lithology_summary: str | None = None,
) -> GeoMindResult:
    """Execute the flow.

        route -> [lithology] -> [retrieve] -> generate -> guard

    Both structured prediction and prose answer come back, so the caller can
    render the log track and the chat bubble from one request.
    """
    state = GeoMindState(
        question=question,
        well_log_path=well_log_path,
        well_log_filename=well_log_filename,
        history=history or [],
        min_thickness_m=min_thickness_m,
        prior_lithology_summary=prior_lithology_summary,
    )

    state = route_node(state)
    
    if state.route == "lithology":
        state = lithology_node(state)
        
    state = retrieve_node(state)
    state = await generate_node(state)
    state = guard_node(state)

    logger.info("graph: %s", " -> ".join(state.trace))
    
    return GeoMindResult(
        answer=state.answer or "",
        prediction=state.prediction,
        llm_used=state.llm_used,
        lithology_summary=state.lithology_summary,
        warnings=state.warnings,
        trace=state.trace,
    )
