"""API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from obot.api.auth import require_api_key
from obot.api.schemas import BotDetail, BotSummary, ChatRequest, ChatResponse, Source
from obot.core.bot import Bot
from obot.core.builder import BuildInProgressError, BuildManager
from obot.core.schema import load_schema

public = APIRouter()
router = APIRouter(dependencies=[Depends(require_api_key)])


@public.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/bots")
def list_bots(request: Request) -> list[BotSummary]:
    builds: BuildManager = request.app.state.builds
    return [BotSummary(name=n, status=s) for n, s in builds.status.items()]


@router.get("/bots/{name}", response_model_by_alias=True)
def get_bot(name: str, request: Request) -> BotDetail:
    builds = _builds_for(request, name)
    status = builds.status[name]
    detail = BotDetail(name=name, status=status, error=builds.errors.get(name))
    if status == "ready":
        detail.record_count = builds.record_counts[name]
        schema = load_schema(builds.storage_dir / name / "schema.json")
        detail.schema_ = schema.model_dump()
    return detail


@router.post("/bots/{name}/chat")
def chat(name: str, body: ChatRequest, request: Request) -> ChatResponse:
    # Sync def: FastAPI runs it in a threadpool, so the slow LLM call does not
    # block the event loop.
    builds = _builds_for(request, name)
    status = builds.status[name]
    if status != "ready":
        raise HTTPException(
            409, {"code": "BOT_NOT_READY", "message": f"bot '{name}' is {status}"}
        )
    state = request.app.state
    bot = Bot(
        name,
        builds.qdrant,
        state.get_embedder(),
        state.llm,
        state.settings.retrieval.top_k,
    )
    result = bot.ask(body.message, [m.model_dump() for m in body.history], body.top_k)
    return ChatResponse(
        answer=result.answer,
        route=result.route,
        sources=[
            Source(id=h.record_id, score=h.score, record=h.record)
            for h in result.sources
        ],
        sql=result.sql,
    )


@router.post("/bots/{name}/rebuild", status_code=202)
def rebuild(name: str, request: Request) -> BotSummary:
    builds = _builds_for(request, name)
    try:
        builds.rebuild(name)
    except BuildInProgressError as e:
        raise HTTPException(409, {"code": "BOT_BUILDING", "message": str(e)}) from e
    return BotSummary(name=name, status="building")


def _builds_for(request: Request, name: str) -> BuildManager:
    builds: BuildManager = request.app.state.builds
    if name not in builds.bots:
        raise HTTPException(
            404, {"code": "BOT_NOT_FOUND", "message": f"unknown bot '{name}'"}
        )
    return builds
