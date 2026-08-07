
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.llm.anthropic_client import LLMUnavailableError, get_llm_client
from app.petrologix.loader import load_model
from app.routers import chat, health, prediction, well_logs

logger = logging.getLogger(__name__)
"""  
Main FastAPI app.

The lifespan context manager loads the model at startup, so a missing or
 uvicorn app.main:app --reload --port 8006 will fail to start the first user.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.debug)
    # Load the lithology model at startup so a missing or corrupt artifact fails
    # the boot rather than the first user upload.
    load_model()

    # Validate the Anthropic client at startup. Failure is non-fatal: lithology
    # prediction does not need the LLM.
    if settings.llm_warm_on_startup:
        async def _warm() -> None:
            try:
                if await get_llm_client().is_awake():
                    logger.info("Anthropic LLM client ready")
            except LLMUnavailableError as e:
                logger.warning("Anthropic LLM client unavailable: %s", e)
            except Exception as e:  # noqa: BLE001 - never block startup on this
                logger.warning("could not initialize Anthropic LLM client: %s", e)
        asyncio.create_task(_warm())

    logger.info("%s ready", settings.app_name)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Petroleum agent: well-log lithology prediction, geology RAG, market data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(chat.router)
app.include_router(well_logs.router)
