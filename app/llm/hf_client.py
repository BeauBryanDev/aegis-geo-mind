
import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# My  model (Qwen2.5-7B QLoRA, 4-bit) runs on the Space's T4.
class LLMUnavailableError(RuntimeError):
    """The Space is unreachable, asleep past our patience, or returned an error."""

class GeoMindLLMClient:
    """Async client for the geologist LLM Space."""

    def __init__(
        self,
        space_url: str | None = None,
        api_name: str | None = None,
        token: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        
        url = (space_url or settings.hf_llm_endpoint or "").rstrip("/")
        
        if not url:
            raise LLMUnavailableError(
                "no LLM endpoint configured -- set HF_LLM_ENDPOINT in .env"
            )
            
        self.base_url = url
        name = api_name or settings.hf_llm_api_name
        self.api_name = name if name.startswith("/") else f"/{name}"
        self.token = token or settings.hf_token
        self.timeout = httpx.Timeout(
            read_timeout or settings.llm_read_timeout_s,
            connect=connect_timeout or settings.llm_connect_timeout_s,
        )

    # internals
    @property
    def _headers(self) -> dict[str, str]:
        # A token is optional for a public Space but strongly preferred: an
        # authenticated call draws on your own HF quota rather than the shared
        # anonymous pool, and it is mandatory the moment the Space goes private.
        h = {"Content-Type": "application/json"}
        
        if self.token:
            
            h["Authorization"] = f"Bearer {self.token}"
            
        return h

    @staticmethod
    def _unwrap(payload: Any) -> str:
        """Gradio returns a list of outputs; this endpoint declares one JSON value."""
        if isinstance(payload, list):
            
            payload = payload[0] if payload else ""
            
        if isinstance(payload, dict):
            # Tolerate a Space that wraps its reply, e.g. {"response": "..."}
            for key in ("response", "text", "content", "generated_text"):
                
                if key in payload:
                    
                    payload = payload[key]
                    break
                
        return str(payload).strip()


    async def _submit(self, client: httpx.AsyncClient, message: str) -> str:
        """Send the message to the Space and return the event_id for the SSE stream."""
        r = await client.post(
            f"{self.base_url}/gradio_api/call{self.api_name}",
            headers=self._headers,
            # Gradio 5's ChatInterface endpoint takes (message, history). History
            # stays empty: prompt_builder already packs prior turns into `message`.
            json={"data": [message, []]},
        )
        if r.status_code == 404:
            
            raise LLMUnavailableError(
                f"endpoint {self.api_name} not found on {self.base_url} -- "
                "check the Space's api_name"
            )
            
        r.raise_for_status()
        
        event_id = r.json().get("event_id")
        
        if not event_id:
            
            raise LLMUnavailableError(f"Space returned no event_id: {r.text[:200]}")
        
        return event_id


    async def _collect(self, client: httpx.AsyncClient, event_id: str) -> str:
        """Read the SSE result stream until 'complete' or 'error'."""
        url = f"{self.base_url}/gradio_api/call{self.api_name}/{event_id}"
        event = None
        async with client.stream("GET", url, headers=self._headers) as resp:
            
            resp.raise_for_status()
            
            async for line in resp.aiter_lines():
                
                if not line:
                    continue
                
                if line.startswith("event:"):
                    
                    event = line.split(":", 1)[1].strip()
                    
                elif line.startswith("data:"):
                    
                    raw = line.split(":", 1)[1].strip()
                    
                    if event == "error":
                        
                        raise LLMUnavailableError(f"Space reported an error: {raw[:300]}")
                    
                    if event == "complete":
                        
                        try:
                            return self._unwrap(json.loads(raw))
                        
                        except json.JSONDecodeError:
                            return raw
        # If I exit the loop without returning, the stream closed before we got a 'complete' event.               
        raise LLMUnavailableError("stream closed before the Space completed")


    # public 
    async def generate(self, message: str, retries: int = 1) -> str:
        """Send one fully-composed message and return the model's reply."""
        last: Exception | None = None
        
        for attempt in range(retries + 1):
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Submit the message to the Space and get an event_id for the SSE stream.
                    event_id = await self._submit(client, message)
                    reply = await self._collect(client, event_id)
                    
                logger.info("LLM replied (%d chars)", len(reply))
                
                return reply
            
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                # Almost always a cold start: the Space is booting the T4.
                logger.warning("LLM call failed (attempt %d/%d): %s",
                               attempt + 1, retries + 1, e)
                
                if attempt < retries:
                    
                    await asyncio.sleep(5)
                    
            except httpx.HTTPStatusError as e:
                
                raise LLMUnavailableError(
                    f"Space returned HTTP {e.response.status_code}"
                ) from e
                
        raise LLMUnavailableError(
            f"LLM Space unreachable after {retries + 1} attempts. It may be "
            f"sleeping -- a cold T4 start takes 1-3 minutes. ({last})"
        ) from last


    async def is_awake(self, timeout: float = 10.0) -> bool:
        """Cheap liveness probe; does not run generation."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                
                r = await client.get(self.base_url, headers=self._headers)
                
                return r.status_code < 500
            
        except Exception:
            
            return False


    async def wake(self) -> bool:
        """Nudge a sleeping Space so the first real user does not pay the boot.

        Fire-and-forget from the app lifespan; never raises.
        """
        awake = await self.is_awake()
        
        logger.info("LLM Space %s", "awake" if awake else "waking (cold start)")
        
        return awake


_client: GeoMindLLMClient | None = None


def get_llm_client() -> GeoMindLLMClient:
    """Process-wide client (construction is cheap; this keeps config in one place)."""
    global _client
    
    if _client is None:
        
        _client = GeoMindLLMClient()
        
    return _client
