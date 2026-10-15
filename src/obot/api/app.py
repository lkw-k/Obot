"""FastAPI app, lifespan, error handlers."""

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from obot.api.routes import public, router
from obot.config import BotConfig, Settings
from obot.core.builder import BuildManager, Locked
from obot.core.embedder import Embedder
from obot.core.store import open_qdrant
from obot.llm.base import LLMClient, LLMError


def create_app(
    settings: Settings,
    bots: list[BotConfig],
    llm: LLMClient,
    get_embedder: Callable[[], Embedder],
    storage_dir: str | Path = "storage",
) -> FastAPI:
    # Opened here, not in lifespan, so a locked folder fails before the server starts.
    qdrant = Locked(open_qdrant(Path(storage_dir) / "qdrant"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        embedder: Embedder | None = None
        embedder_lock = threading.Lock()

        def shared_embedder() -> Embedder:
            # Load lazily so startup does not wait for the model, and wrap once
            # so builds and chat requests share one lock.
            nonlocal embedder
            with embedder_lock:
                if embedder is None:
                    embedder = Locked(get_embedder())
            return embedder

        builds = BuildManager(bots, settings, shared_embedder, qdrant, storage_dir)
        app.state.settings = settings
        app.state.llm = llm
        app.state.builds = builds
        app.state.get_embedder = shared_embedder
        builds.start()
        try:
            yield
        finally:
            builds.stop()
            qdrant.close()

    app = FastAPI(title="OBot", lifespan=lifespan)
    app.include_router(public)
    app.include_router(router)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            error = exc.detail
        else:
            error = {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}
        return JSONResponse({"error": error}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "message": message}},
            status_code=422,
        )

    @app.exception_handler(LLMError)
    async def llm_error(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "LLM_ERROR", "message": str(exc)}}, status_code=502
        )

    return app
