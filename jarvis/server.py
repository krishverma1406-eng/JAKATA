"""FastAPI frontend server for JARVIS."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import BASE_DIR, SETTINGS
from core.agent import Agent
from core.interface_config import INTERFACE_CONFIG
from core.memory import Memory
from services.tts import synthesize_base64_sync


FRONTEND_DIR = BASE_DIR / "frontend"
CAM_BYPASS_TOKEN = "TTCAMTOKENTT"

app = FastAPI(title="JARVIS Web", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: str | None = None
    mode: str | None = None
    tts: bool = True
    imgbase64: str | None = None


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    mode: str | None = None


class SessionRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class SessionModeRequest(BaseModel):
    mode: str = Field(..., min_length=1, max_length=40)


MAX_SESSIONS = 50
_SESSIONS: dict[str, Agent] = {}
_SESSIONS_LOCK = threading.Lock()
_TASKS: dict[str, dict[str, Any]] = {}
_SERVER_MEMORY: Memory | None = None


def _frontend_file(name: str) -> Path:
    return FRONTEND_DIR / name


def _get_memory() -> Memory:
    global _SERVER_MEMORY
    if _SERVER_MEMORY is None:
        _SERVER_MEMORY = Memory(SETTINGS)
    return _SERVER_MEMORY


def _session_payload(agent: Agent) -> dict[str, Any]:
    meta = dict(agent.session_meta or {})
    mode_config = agent.interface.get_mode(agent.mode)
    return {
        "session_id": agent.session_id,
        "display_name": str(meta.get("display_name", "")).strip() or "Untitled session",
        "mode": agent.mode,
        "mode_label": str(mode_config.get("label", agent.mode)).strip() or agent.mode.title(),
        "turn_count": int(meta.get("turn_count", 0) or 0),
        "updated_at": str(meta.get("updated_at", "")).strip(),
    }


def _get_agent(session_id: str | None, mode: str | None = None) -> tuple[str, Agent]:
    with _SESSIONS_LOCK:
        sid = (session_id or "").strip() or uuid.uuid4().hex[:12]
        agent = _SESSIONS.get(sid)
        if agent is None:
            if len(_SESSIONS) >= MAX_SESSIONS:
                oldest_sid = next(iter(_SESSIONS))
                _SESSIONS.pop(oldest_sid, None)
            session_settings = replace(SETTINGS, tts_enabled=True)
            agent = Agent(settings=session_settings)
            agent.bind_session(sid, mode=mode)
            _SESSIONS[sid] = agent
        elif mode:
            agent.set_mode(mode)
        return sid, agent


def _clean_message(message: str) -> str:
    return re.sub(r"\s+", " ", message.replace(CAM_BYPASS_TOKEN, " ")).strip()


def _emit_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normalize_search_results(tool_result: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any] | None:
    inner = tool_result.get("result", {}) if isinstance(tool_result, dict) else {}
    if not isinstance(inner, dict) or not inner.get("ok"):
        return None

    results = []
    for item in inner.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "content": str(item.get("snippet", "") or item.get("content", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "score": item.get("score"),
                "published_date": str(item.get("published_date", "")).strip(),
            }
        )

    return {
        "query": str(inner.get("query") or arguments.get("query") or "").strip(),
        "answer": f"Results from {inner.get('provider', 'web search')}.",
        "results": results,
    }


def _persist_non_tool_turn(agent: Agent, user_message: str, assistant_message: str) -> None:
    agent._remember_turn(user_message, assistant_message)
    agent.memory.persist_conversation(
        user_message=user_message,
        assistant_message=assistant_message,
        conversation=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        tool_trace=[],
        session_id=agent.session_id,
        brain=agent.brain,
        should_extract=agent._should_extract_memory(user_message, []),
        background=agent.settings.background_memory_persistence,
    )


def _build_stream(request: ChatRequest) -> Any:
    session_id, agent = _get_agent(request.session_id, request.mode)
    message = _clean_message(request.message)
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    def event_stream() -> Any:
        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        started_at = time.perf_counter()
        first_chunk_at: float | None = None
        streamed_any = False
        route_emitted = False
        stream_started_emitted = False
        current_route = "chat"

        def emit(payload: dict[str, Any]) -> None:
            event_queue.put(payload)

        def stream_handler(chunk: str) -> None:
            nonlocal first_chunk_at, streamed_any, route_emitted, stream_started_emitted, current_route
            if not chunk:
                return
            streamed_any = True
            if not route_emitted:
                emit({"activity": {"event": "routing", "route": current_route}})
                route_emitted = True
            if not stream_started_emitted:
                emit({"activity": {"event": "streaming_started", "route": current_route}})
                stream_started_emitted = True
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                emit(
                    {
                        "activity": {
                            "event": "first_chunk",
                            "route": current_route,
                            "elapsed_ms": int((first_chunk_at - started_at) * 1000),
                        }
                    }
                )
            emit({"chunk": chunk})

        def event_handler(event: dict[str, Any]) -> None:
            nonlocal route_emitted, stream_started_emitted, current_route
            event_type = event.get("type")
            if event_type == "activity":
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    emit({"activity": payload})
                return
            tool_name = str(event.get("name", "")).strip()
            arguments = event.get("arguments", {})
            result = event.get("result", {})

            if event_type == "tool_started":
                if tool_name == "web_search":
                    current_route = "realtime"
                    emit(
                        {
                            "activity": {
                                "event": "searching_web",
                                "query": str(arguments.get("query", "")).strip(),
                            }
                        }
                    )
                    if not route_emitted:
                        emit({"activity": {"event": "routing", "route": "realtime"}})
                        route_emitted = True
                elif tool_name == "vision_tool":
                    current_route = "vision"
                    emit({"activity": {"event": "vision_analyzing", "message": "Analyzing live camera frame..."}})
                    if not route_emitted:
                        emit({"activity": {"event": "routing", "route": "vision"}})
                        route_emitted = True
                else:
                    emit(
                        {
                            "activity": {
                                "event": "tasks_executing",
                                "message": f"Running {tool_name}...",
                            }
                        }
                    )
                return

            if event_type != "tool_result":
                return

            if tool_name == "web_search":
                search_payload = _normalize_search_results(result, arguments if isinstance(arguments, dict) else {})
                emit(
                    {
                        "activity": {
                            "event": "search_completed",
                            "message": "Search completed",
                        }
                    }
                )
                if search_payload:
                    emit({"search_results": search_payload})
                    if not stream_started_emitted:
                        emit({"activity": {"event": "streaming_started", "route": "realtime"}})
                        stream_started_emitted = True
            else:
                emit(
                    {
                        "activity": {
                            "event": "tasks_completed",
                            "message": f"Completed {tool_name}.",
                        }
                    }
                )

        def worker() -> None:
            try:
                emit({"session_id": session_id, "session": _session_payload(agent)})
                for startup_message in agent.startup_messages():
                    emit({"startup_message": startup_message, "session": _session_payload(agent)})
                emit({"activity": {"event": "query_detected", "message": message}})

                response_text = agent.run(
                    message,
                    event_handler=event_handler,
                    runtime_context={"imgbase64": request.imgbase64 or ""},
                )
                if not streamed_any and response_text:
                    if not route_emitted:
                        emit({"activity": {"event": "routing", "route": current_route}})
                    if not stream_started_emitted:
                        emit({"activity": {"event": "streaming_started", "route": current_route}})
                    emit({"activity": {"event": "first_chunk", "route": current_route, "elapsed_ms": int((time.perf_counter() - started_at) * 1000)}})
                    emit({"chunk": response_text})
                if request.tts:
                    try:
                        audio = synthesize_base64_sync(response_text, agent.settings)
                    except Exception:
                        audio = None
                    if audio:
                        emit({"audio": audio})
                emit({"done": True, "session_id": session_id, "session": _session_payload(agent)})
            except Exception as exc:
                emit({"error": str(exc), "done": True, "session_id": session_id, "session": _session_payload(agent)})
            finally:
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = event_queue.get()
            if item is None:
                break
            yield _emit_sse(item)

    return event_stream()


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/index.html", status_code=307)


@app.get("/health")
def health() -> dict[str, Any]:
    live_providers = []
    if SETTINGS.nvidia_api_key.strip():
        live_providers.append("nvidia")
    if SETTINGS.openrouter_api_key.strip():
        live_providers.append("openrouter")
    status = "healthy" if live_providers else "degraded"
    return {
        "status": status,
        "frontend": FRONTEND_DIR.exists(),
        "live_providers": live_providers,
        "sessions": len(_SESSIONS),
        "default_mode": INTERFACE_CONFIG.default_mode(),
    }


@app.get("/interface/config")
def interface_config() -> dict[str, Any]:
    return {
        "default_mode": INTERFACE_CONFIG.default_mode(),
        "modes": INTERFACE_CONFIG.list_modes(),
        "briefing": INTERFACE_CONFIG.briefing(),
    }


@app.post("/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    session_id, agent = _get_agent(request.session_id, request.mode)
    return {
        "session_id": session_id,
        "session": _session_payload(agent),
        "startup_messages": agent.startup_messages(),
    }


@app.get("/sessions")
def list_sessions(limit: int = 20) -> dict[str, Any]:
    memory = _get_memory()
    return {"sessions": memory.list_sessions(limit=max(1, min(limit, 50)))}


@app.get("/sessions/search")
def search_sessions(query: str, limit: int = 8) -> dict[str, Any]:
    memory = _get_memory()
    return memory.session_search(query, limit=max(1, min(limit, 25)))


@app.post("/sessions/{session_id}/rename")
def rename_session(session_id: str, request: SessionRenameRequest) -> dict[str, Any]:
    memory = _get_memory()
    session = memory.rename_session(session_id, request.name)
    with _SESSIONS_LOCK:
        agent = _SESSIONS.get(session_id)
        if agent is not None:
            agent.session_meta = dict(session)
    return {"ok": True, "session": session}


@app.post("/sessions/{session_id}/mode")
def update_session_mode(session_id: str, request: SessionModeRequest) -> dict[str, Any]:
    with _SESSIONS_LOCK:
        agent = _SESSIONS.get(session_id)
    if agent is None:
        _sid, agent = _get_agent(session_id, request.mode)
    else:
        agent.set_mode(request.mode)
    return {
        "ok": True,
        "session": _session_payload(agent),
        "startup_messages": agent.startup_messages(),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/chat/jarvis/stream")
def chat_jarvis_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_build_stream(request), media_type="text/event-stream")


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = _TASKS.get(task_id)
    if task is None:
        return {"task_id": task_id, "status": "failed", "error": "Task not found."}
    return task


@app.get("/app")
def app_root() -> RedirectResponse:
    return RedirectResponse(url="/app/index.html", status_code=307)


@app.get("/app/index.html")
def app_index() -> FileResponse:
    path = _frontend_file("index.html")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend index not found.")
    return FileResponse(path)


@app.get("/app/viewer.html")
def app_viewer() -> FileResponse:
    path = _frontend_file("viewer.html")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Viewer not found.")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
