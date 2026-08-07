
import logging

from app.agent.prompts import GENERAL_SYSTEM_PROMPT, LITHOLOGY_RULES, LITHOLOGY_CONTEXT_TEMPLATE


logger = logging.getLogger(__name__)


MAX_MESSAGE_TOKENS = 700


MAX_HISTORY_TURNS = 3

# this is for my  self own llm in HuggingFace Space
def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token)."""
    return len(text) // 4


def build_message(
    question: str,
    lithology_summary: str | None = None,
    history: list[dict] | None = None,
    include_rules: bool = True,
) -> str:
    """Compose one message for the LLM.
    question : the user's current question.
    lithology_summary : output of `xgboost_tool.run_lithology_tool`, when a well log has been analysed.
    history : prior turns as [{"role": "user"|"assistant", "content": str}].
    include_rules : whether to include the lithology rules.
    """
    def compose(hist_blocks: list[str]) -> str:
        
        blocks: list[str] = []
        
        if lithology_summary:
            
            blocks.append(
                LITHOLOGY_CONTEXT_TEMPLATE.format(summary=lithology_summary)
            )
        blocks.extend(hist_blocks)
        
        tail = (
            GENERAL_SYSTEM_PROMPT + "\n\n") if include_rules else ""
        
        blocks.append(
            f"{tail}Question: {question.strip()}"
                      )
        
        return "\n\n---\n\n".join(blocks)
    

    history_blocks = _format_history(history or [])
    message = compose(history_blocks)

    # Trim history oldest-first if we are over budget.
    while estimate_tokens(message) > MAX_MESSAGE_TOKENS and history_blocks:
        
        history_blocks.pop(0)
        
        message = compose(history_blocks)
        
        logger.debug("trimmed history to fit budget (%d tokens)", estimate_tokens(message))

    if estimate_tokens(message) > MAX_MESSAGE_TOKENS:
        logger.warning(
            "composed message is ~%d tokens, over the %d budget -- the reply may "
            "be truncated", estimate_tokens(message), MAX_MESSAGE_TOKENS,
        )
    if include_rules and lithology_summary:
        
        message += "\n\n" + LITHOLOGY_RULES 
        
        
    return message


def _format_history(history: list[dict]) -> list[str]:
    
    """Render the last few turns as plain text blocks."""
    
    turns = [h for h in history if h.get("role") in ("user", "assistant")]
    
    turns = turns[-MAX_HISTORY_TURNS * 2:]          # a "turn" is user + assistant
    
    if not turns:
    
        return []
    
    rendered = "\n".join(
    
        f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'].strip()}"
        for t in turns
    )
    
    return [f"Earlier in this conversation:\n{rendered}"]
