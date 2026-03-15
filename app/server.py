from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import LLMCacheService
from .config import get_settings
from .embeddings import EmbeddingService
from .evaluation import RagasEvaluationService
from .knowledge_base import KnowledgeBaseService
from .models import (
    CacheResetResponse,
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    DocumentsResponse,
    EvaluationRequest,
    EvaluationResponse,
    IngestResponse,
    PathIngestRequest,
    ResetRequest,
    SampleIngestRequest,
)
from .rag_chain import RAGService
from .reranker import CrossEncoderReranker
from .session_manager import SessionManager
from .vector_store import VectorIndex


settings = get_settings()
cache_service = LLMCacheService(settings)
embedding_service = EmbeddingService(settings.embedding_model_name)
reranker = None
if settings.enable_reranking:
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        batch_size=settings.rerank_batch_size,
    )
vector_index = VectorIndex(
    settings.vector_db_path,
    embedding_service,
    reranker=reranker,
    candidate_top_k=settings.candidate_top_k,
)
session_manager = SessionManager()
knowledge_base = KnowledgeBaseService(settings, session_manager, vector_index)
rag_service = RAGService(settings, vector_index)
evaluation_service = RagasEvaluationService(rag_service, embedding_service)


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "api_key_configured": bool(settings.llm_api_key),
                "default_model": settings.llm_model,
                "provider_label": settings.provider_label,
                "primary_api_key_env": settings.primary_api_key_env,
                "cache_enabled": settings.enable_llm_cache,
            },
        )

    @app.get("/api/health")
    def health():
        cache_snapshot = cache_service.snapshot()
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "embedding_model": settings.embedding_model_name,
            "cache_enabled": settings.enable_llm_cache,
            "cache_backend": cache_snapshot.backend,
            "cache_stats": cache_snapshot.model_dump(),
            "reranking_enabled": settings.enable_reranking,
            "reranker_model": settings.reranker_model_name if settings.enable_reranking else "",
            "api_key_configured": bool(settings.llm_api_key),
        }

    @app.post("/api/session", response_model=CreateSessionResponse)
    def create_session():
        session = session_manager.create_session()
        return CreateSessionResponse(session_id=session.session_id)

    @app.get("/api/sessions/{session_id}/documents", response_model=DocumentsResponse)
    def list_documents(session_id: str):
        session = session_manager.get_or_create(session_id)
        return DocumentsResponse(
            session_id=session.session_id,
            documents=session_manager.list_documents(session.session_id),
        )

    @app.post("/api/documents/sample", response_model=IngestResponse)
    def ingest_sample_documents(payload: SampleIngestRequest):
        session = session_manager.get_or_create(payload.session_id)
        try:
            documents = knowledge_base.ingest_samples(session.session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/documents/path", response_model=IngestResponse)
    def ingest_documents_by_path(payload: PathIngestRequest):
        session = session_manager.get_or_create(payload.session_id)
        paths = [path.strip() for path in payload.paths if path.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="请至少提供一个有效路径。")
        try:
            documents = knowledge_base.ingest_paths(session.session_id, paths)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/documents/upload", response_model=IngestResponse)
    async def ingest_documents_by_upload(
        session_id: str = Form(...),
        files: List[UploadFile] = File(...),
    ):
        session = session_manager.get_or_create(session_id)
        documents = []
        try:
            for upload in files:
                content = await upload.read()
                documents.append(
                    knowledge_base.ingest_upload(
                        session.session_id,
                        upload.filename or "uploaded.txt",
                        content,
                    )
                )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(session_id=session.session_id, documents=documents)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        session = session_manager.get_or_create(payload.session_id)
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")
        try:
            return rag_service.ask(session, question)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest):
        session = session_manager.get_or_create(payload.session_id)
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")
        try:
            rag_service.ensure_ready(session)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def event_stream():
            try:
                async for event in rag_service.stream_ask(session, question):
                    yield sse_event(event["event"], event["data"])
            except RuntimeError as exc:
                yield sse_event("error", {"detail": str(exc)})
            except Exception as exc:
                yield sse_event("error", {"detail": f"流式回答失败: {exc}"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/evaluate", response_model=EvaluationResponse)
    async def evaluate_sample_benchmark(payload: EvaluationRequest):
        session = session_manager.get_or_create(payload.session_id)
        try:
            return evaluation_service.run_sample_benchmark(session)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cache/reset", response_model=CacheResetResponse)
    def reset_cache():
        return CacheResetResponse(cache=cache_service.reset())

    @app.post("/api/session/reset", response_model=DocumentsResponse)
    def reset_session(payload: ResetRequest):
        session = session_manager.get_or_create(payload.session_id)
        session_manager.reset_history(session.session_id)
        if payload.clear_documents:
            knowledge_base.reset_session_documents(session.session_id)
        return DocumentsResponse(
            session_id=session.session_id,
            documents=session_manager.list_documents(session.session_id),
        )

    return app


app = create_app()
