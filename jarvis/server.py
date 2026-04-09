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

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import BASE_DIR, SETTINGS
from core.agent import Agent
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
    tts: bool = True
    imgbase64: str | None = None


_SESSIONS: dict[str, Agent] = {}
_SESSIONS_LOCK = threading.Lock()
_TASKS: dict[str, dict[str, Any]] = {}


def _frontend_file(name: str) -> Path:
    return FRONTEND_DIR / name


def _get_agent(session_id: str | None) -> tuple[str, Agent]:
    with _SESSIONS_LOCK:
        sid = (session_id or "").strip() or uuid.uuid4().hex[:12]
        agent = _SESSIONS.get(sid)
        if agent is None:
            session_settings = replace(SETTINGS, tts_enabled=True)
            agent = Agent(settings=session_settings)
            agent.bind_session(sid)
            _SESSIONS[sid] = agent
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


def _vision_reply(prompt: str, imgbase64: str) -> dict[str, Any]:
    prompt = prompt.strip() or "Describe what is visible in this image."
    data_uri = f"data:image/jpeg;base64,{imgbase64}"
    providers = [
        (
            "nvidia",
            SETTINGS.nvidia_api_key.strip(),
            f"{SETTINGS.nvidia_base_url.rstrip('/')}/chat/completions",
            SETTINGS.nvidia_complex_model,
            {"chat_template_kwargs": {"thinking": False}},
            {
                "Authorization": f"Bearer {SETTINGS.nvidia_api_key.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        ),
        (
            "openrouter",
            SETTINGS.openrouter_api_key.strip(),
            f"{SETTINGS.openrouter_base_url.rstrip('/')}/chat/completions",
            SETTINGS.openrouter_complex_model,
            {},
            {
                "Authorization": f"Bearer {SETTINGS.openrouter_api_key.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "HTTP-Referer": "https://github.com/akyourowngames/JAKATA",
                "X-OpenRouter-Title": "JARVIS",
            },
        ),
    ]

    errors: list[str] = []
    for provider, api_key, url, model, extra, headers in providers:
        if not api_key:
            continue
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.2,
            **extra,
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            raw = response.json()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            code = exc.response.status_code if exc.response is not None else "unknown"
            errors.append(f"{provider}: HTTP {code}: {body}")
            continue
        except requests.RequestException as exc:
            errors.append(f"{provider}: {exc}")
            continue

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, str) and content.strip():
            return {"ok": True, "provider": provider, "content": content.strip()}

    return {
        "ok": False,
        "error": " | ".join(errors) if errors else "No vision-capable provider is configured.",
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
    session_id, agent = _get_agent(request.session_id)
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
                emit({"session_id": session_id})
                emit({"activity": {"event": "query_detected", "message": message}})

                if request.imgbase64:
                    emit({"activity": {"event": "routing", "route": "vision"}})
                    emit({"activity": {"event": "vision_analyzing", "message": "Analyzing image..."}})
                    vision = _vision_reply(message, request.imgbase64)
                    if not vision.get("ok"):
                        emit({"error": str(vision.get("error", "Vision analysis failed.")), "done": True, "session_id": session_id})
                        return
                    response_text = str(vision.get("content", "")).strip() or "No response generated."
                    emit({"activity": {"event": "streaming_started", "route": "vision"}})
                    emit({"activity": {"event": "first_chunk", "route": "vision", "elapsed_ms": int((time.perf_counter() - started_at) * 1000)}})
                    emit({"chunk": response_text})
                    if request.tts:
                        try:
                            audio = synthesize_base64_sync(response_text, agent.settings)
                        except Exception:
                            audio = None
                        if audio:
                            emit({"audio": audio})
                    _persist_non_tool_turn(agent, message, response_text)
                    emit({"done": True, "session_id": session_id})
                    return

                response_text = agent.run(message, stream_handler=stream_handler, event_handler=event_handler)
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
                emit({"done": True, "session_id": session_id})
            except Exception as exc:
                emit({"error": str(exc), "done": True, "session_id": session_id})
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
