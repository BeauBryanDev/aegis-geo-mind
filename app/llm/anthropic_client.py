import logging
 
from anthropic import AsyncAnthropic, APIConnectionError, APIStatusError, APITimeoutError
 
from app.core.config import settings


logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Claude API is unreachable, rate-limited, or returned an error."""
 
 
class ClaudeLLMClient:
    """Async client for Claude (Anthropic API)."""
 
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        
        key = api_key or settings.anthropic_api_key
        
        if not key:
            raise LLMUnavailableError(
                "no Anthropic API key configured -- set ANTHROPIC_API_KEY in .env"
            )
            
        self.model = model or settings.anthropic_model or "claude-sonnet-5"
        self.max_tokens = max_tokens or settings.anthropic_max_tokens
        # Omitting `thinking` on Sonnet 5 means adaptive thinking, which draws
        # on max_tokens; send it explicitly so the setting decides.
        self.thinking = (
            {"type": "adaptive"} if settings.anthropic_thinking else {"type": "disabled"}
        )
        self._client = AsyncAnthropic(api_key=key, timeout=timeout)
 
    async def generate(
        self,
        message: str,
        system: str | None = None,
        history: list[dict] | None = None,
        retries: int = 1,
    ) -> str:
        """Send one turn and return Claude's reply.
 
        message : the current user turn.
        system : system prompt (persona, rules). Sent via the API's dedicated
            `system` parameter, not folded into the message.
        history : prior turns as [{"role": "user"|"assistant", "content": str}].
 
        """
        messages = [*(history or []), {"role": "user", "content": message}]
 
        last: Exception | None = None
        
        for attempt in range(retries + 1):
            try:
                # The API returns a structured response with a list of content blocks, each
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking=self.thinking,
                    system=system or "",
                    messages=messages,
                )
                reply = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                if response.stop_reason == "max_tokens":
                    # The answer is cut mid-sentence. Say so rather than letting
                    # a truncated geology assessment read as a complete one.
                    logger.warning(
                        "Claude hit max_tokens=%d; answer truncated", self.max_tokens
                    )
                    reply = reply.rstrip() + "\n\n[Answer truncated at the length limit.]"
                    
                logger.info(
                    "Claude replied (%d chars, stop=%s, out=%d tokens)",
                    len(reply),
                    response.stop_reason,
                    response.usage.output_tokens,
                )
                return reply
            
            except (APIConnectionError, APITimeoutError) as e:
                last = e
                logger.warning(
                    "Claude call failed (attempt %d/%d): %s", attempt + 1, retries + 1, e
                )
            except APIStatusError as e:
                # 4xx/5xx from the API -- retrying blindly on a 4xx (e.g. bad
                # request) would just fail again, so only retry loop continues
                # for connection/timeout cases above; surface this immediately.
                raise LLMUnavailableError(
                    f"Claude API returned HTTP {e.status_code}: {e.message}"
                ) from e
 
        raise LLMUnavailableError(
            f"Claude API unreachable after {retries + 1} attempts. ({last})"
        ) from last
 
    async def is_awake(self) -> bool:
        """No cold starts with the API -- always True if a key is configured."""
        return True
 
 
_client: ClaudeLLMClient | None = None
 
 
def get_llm_client() -> ClaudeLLMClient:
    """Process-wide client (construction is cheap; keeps config in one place)."""
    global _client
    
    if _client is None:
        
        _client = ClaudeLLMClient()
        
    return _client
 
 
 