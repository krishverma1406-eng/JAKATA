"""Music playback via VLC with local file and YouTube support."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
_STATE: dict[str, Any] = {
    "instance": None,
    "player": None,
    "queue": [],
    "index": -1,
}


TOOL_DEFINITION = {
    "name": "music_player",
    "description": "Play local music or YouTube audio and control playback.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "resume", "stop", "next", "volume", "queue", "status"],
                "description": "Music action to perform.",
            },
            "track": {
                "type": "string",
                "description": "Track path, folder, fuzzy name, or YouTube URL.",
            },
            "volume": {
                "type": "integer",
                "description": "Volume percent from 0 to 100.",
            },
            "replace_queue": {
                "type": "boolean",
                "description": "Replace the existing queue when adding tracks.",
                "default": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    if action == "play":
        return _play(params)
    if action == "pause":
        return _pause()
    if action == "resume":
        return _resume()
    if action == "stop":
        return _stop()
    if action == "next":
        return _next()
    if action == "volume":
        return _set_volume(params)
    if action == "queue":
        return _queue_status()
    if action == "status":
        return _status()
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _play(params: dict[str, Any]) -> dict[str, Any]:
    track = str(params.get("track", "")).strip()
    if not track:
        return {"ok": False, "error": "Missing track to play."}

    player = _ensure_player()
    if player is None:
        return {"ok": False, "error": "python-vlc is installed, but VLC runtime is not available on this machine."}

    replace_queue = bool(params.get("replace_queue", True))
    resolved_items = _resolve_tracks(track)
    if not resolved_items:
        return {"ok": False, "error": f"Could not resolve any playable media for: {track}"}

    if replace_queue:
        _STATE["queue"] = resolved_items
        _STATE["index"] = 0
    else:
        if not _STATE["queue"]:
            _STATE["index"] = 0
        _STATE["queue"].extend(resolved_items)
        if _STATE["index"] < 0:
            _STATE["index"] = 0

    _load_current_media(player)
    return {
        "ok": True,
        "status": "playing",
        "current": _STATE["queue"][_STATE["index"]],
        "queue_length": len(_STATE["queue"]),
    }


def _pause() -> dict[str, Any]:
    player = _STATE.get("player")
    if player is None:
        return {"ok": False, "error": "Nothing is currently playing."}
    player.pause()
    return {"ok": True, "status": "paused"}


def _resume() -> dict[str, Any]:
    player = _STATE.get("player")
    if player is None:
        return {"ok": False, "error": "Nothing is currently loaded."}
    player.play()
    return {"ok": True, "status": "playing"}


def _stop() -> dict[str, Any]:
    player = _STATE.get("player")
    if player is not None:
        player.stop()
    _STATE["queue"] = []
    _STATE["index"] = -1
    return {"ok": True, "status": "stopped"}


def _next() -> dict[str, Any]:
    player = _STATE.get("player")
    queue = _STATE.get("queue", [])
    if player is None or not queue:
        return {"ok": False, "error": "Queue is empty."}
    next_index = _STATE["index"] + 1
    if next_index >= len(queue):
        return {"ok": False, "error": "Already at the end of the queue."}
    _STATE["index"] = next_index
    _load_current_media(player)
    return {"ok": True, "status": "playing", "current": queue[next_index]}


def _set_volume(params: dict[str, Any]) -> dict[str, Any]:
    player = _STATE.get("player")
    if player is None:
        return {"ok": False, "error": "Nothing is currently loaded."}
    volume = max(0, min(int(params.get("volume", 50) or 50), 100))
    player.audio_set_volume(volume)
    return {"ok": True, "volume": volume}


def _queue_status() -> dict[str, Any]:
    queue = list(_STATE.get("queue", []))
    index = int(_STATE.get("index", -1))
    return {"ok": True, "queue": queue, "current_index": index}


def _status() -> dict[str, Any]:
    player = _STATE.get("player")
    if player is None:
        return {"ok": True, "status": "idle", "queue": []}
    return {
        "ok": True,
        "status": str(player.get_state()),
        "current": _STATE["queue"][_STATE["index"]] if _STATE["queue"] and _STATE["index"] >= 0 else "",
        "queue_length": len(_STATE["queue"]),
    }


def _ensure_player() -> Any | None:
    if _STATE["player"] is not None:
        return _STATE["player"]
    try:
        import vlc
    except ImportError:
        return None
    try:
        instance = vlc.Instance()
        player = instance.media_player_new()
    except Exception:
        return None
    _STATE["instance"] = instance
    _STATE["player"] = player
    return player


def _load_current_media(player: Any) -> None:
    current = _STATE["queue"][_STATE["index"]]
    instance = _STATE["instance"]
    media = instance.media_new(current)
    player.set_media(media)
    player.play()


def _resolve_tracks(track: str) -> list[str]:
    if track.startswith("http://") or track.startswith("https://"):
        if "youtube.com" in track or "youtu.be" in track:
            url = _resolve_youtube_stream(track)
            return [url] if url else []
        return [track]

    candidate = Path(track)
    if candidate.exists():
        if candidate.is_dir():
            return _audio_files_in_dir(candidate)
        return [str(candidate.resolve())]

    search_roots = [Path.cwd(), Path.home() / "Music"]
    for env_name in ("JARVIS_MUSIC_DIR", "MUSIC_DIR"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            search_roots.append(Path(raw))

    found: list[tuple[float, str]] = []
    query = track.lower()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            score = _match_score(query, path.stem.lower())
            if score > 0:
                found.append((score, str(path.resolve())))
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found[:10]]


def _audio_files_in_dir(folder: Path) -> list[str]:
    return [
        str(path.resolve())
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    ]


def _resolve_youtube_stream(url: str) -> str | None:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None

    options = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "noplaylist": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    if not info:
        return None
    if isinstance(info, dict):
        return str(info.get("url", "")).strip() or None
    return None


def _match_score(query: str, name: str) -> float:
    if query in name:
        return 1.0 + len(query) / max(len(name), 1)
    if not query or not name:
        return 0.0
    common = len(set(query.split()) & set(name.split()))
    ratio = common / max(len(set(query.split())), 1)
    return ratio
