"""Named JSONL session persistence for JARVIS."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


SESSION_INDEX_FILE = "session_index.json"


class SessionStore:
    """Persist chat sessions as named JSONL files plus a lightweight index."""

    def __init__(self, sessions_dir: Path | str) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.sessions_dir / SESSION_INDEX_FILE
        self._lock = threading.Lock()
        if not self.index_path.exists():
            self._save_index({"sessions": {}})
        self._migrate_legacy_json_sessions()

    def ensure_session(
        self,
        session_id: str,
        preferred_name: str | None = None,
        mode: str | None = "normal",
    ) -> dict[str, Any]:
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id is required.")

        cleaned_mode = self.normalize_mode(mode) if mode is not None else None
        cleaned_name = self._clean_session_name(preferred_name)
        with self._lock:
            index = self._load_index()
            session = self._ensure_session_unlocked(
                index,
                session_id,
                preferred_name=cleaned_name,
                mode=cleaned_mode,
            )
            self._save_index(index)
            return dict(session)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        session_id = str(session_id).strip()
        if not session_id:
            return None
        with self._lock:
            index = self._load_index()
            session = index.get("sessions", {}).get(session_id)
            if not isinstance(session, dict):
                return None
            return dict(session)

    def rename_session(self, session_id: str, new_name: str) -> dict[str, Any]:
        cleaned_name = self._clean_session_name(new_name)
        if not cleaned_name:
            raise ValueError("Session name cannot be empty.")

        with self._lock:
            index = self._load_index()
            session = self._ensure_session_unlocked(index, session_id)
            self._set_session_name_unlocked(session, cleaned_name, auto_named=False)
            self._save_index(index)
            return dict(session)

    def maybe_autoname_from_text(self, session_id: str, text: str) -> dict[str, Any] | None:
        candidate_name = self._derive_name_from_text(text)
        if not candidate_name:
            return self.get_session(session_id)

        with self._lock:
            index = self._load_index()
            session = index.get("sessions", {}).get(session_id)
            if not isinstance(session, dict):
                return None
            if not session.get("auto_named", True):
                return dict(session)
            self._set_session_name_unlocked(session, candidate_name, auto_named=True)
            self._save_index(index)
            return dict(session)

    def set_mode(self, session_id: str, mode: str) -> dict[str, Any]:
        with self._lock:
            index = self._load_index()
            session = self._ensure_session_unlocked(index, session_id)
            session["mode"] = self.normalize_mode(mode)
            session["updated_at"] = self._now_iso()
            self._save_index(index)
            return dict(session)

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            index = self._load_index()
            sessions = [
                dict(session)
                for session in index.get("sessions", {}).values()
                if isinstance(session, dict)
            ]
        sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return sessions[: max(1, limit)]

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_trace: list[dict[str, Any]] | None = None,
        turn_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_trace = tool_trace or []
        turn_meta = turn_meta or {}
        with self._lock:
            index = self._load_index()
            session = self._ensure_session_unlocked(index, session_id, mode=None)

            user_text = str(user_message).strip()
            assistant_text = str(assistant_message).strip()
            if user_text:
                self._maybe_autoname_unlocked(session, user_text)

            timestamp = self._now_iso()
            entries: list[dict[str, Any]] = []
            if user_text:
                entries.append(
                    {
                        "type": "message",
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "role": "user",
                        "content": user_text,
                    }
                )
            if assistant_text:
                assistant_entry = {
                    "type": "message",
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "role": "assistant",
                    "content": assistant_text,
                    "tools_used": [
                        str(item.get("name", "")).strip()
                        for item in tool_trace
                        if str(item.get("name", "")).strip()
                    ],
                }
                for key in (
                    "mode",
                    "provider",
                    "model",
                    "latency_ms",
                    "total_latency_ms",
                    "tool_count",
                    "plan_note",
                ):
                    value = turn_meta.get(key)
                    if value not in (None, "", []):
                        assistant_entry[key] = value
                entries.append(assistant_entry)

            session_path = self._session_path(session)
            with session_path.open("a", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

            if user_text and not session.get("first_user_message"):
                session["first_user_message"] = self._truncate(user_text, 240)
            if user_text:
                session["last_user_message"] = self._truncate(user_text, 240)
            if assistant_text:
                session["last_assistant_message"] = self._truncate(assistant_text, 240)
                session["turn_count"] = int(session.get("turn_count", 0) or 0) + 1
            session["updated_at"] = timestamp
            self._save_index(index)
            return dict(session)

    def load_messages(self, session_id: str, limit_messages: int = 12) -> list[dict[str, str]]:
        session = self.get_session(session_id)
        if session is None:
            return self._load_legacy_messages(session_id, limit_messages)

        messages = [
            {"role": record["role"], "content": record["content"]}
            for record in self._read_session_records(self._session_path(session))
            if record.get("role") in {"user", "assistant"} and str(record.get("content", "")).strip()
        ]
        if not messages:
            return self._load_legacy_messages(session_id, limit_messages)
        return messages[-max(1, limit_messages) :]

    def search_sessions(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_text = str(query).strip()
        if not query_text:
            return []

        query_lower = query_text.lower()
        query_tokens = self._tokens(query_text)
        matches: list[tuple[float, dict[str, Any]]] = []

        for session in self.list_sessions(limit=200):
            search_text = " ".join(
                [
                    str(session.get("display_name", "")),
                    str(session.get("first_user_message", "")),
                    str(session.get("last_user_message", "")),
                    str(session.get("last_assistant_message", "")),
                ]
            ).strip()
            if not search_text:
                continue
            score = self._score_text(query_lower, query_tokens, search_text)
            if score <= 0:
                continue
            session_copy = dict(session)
            session_copy["snippets"] = self._session_snippets(session_copy, query_tokens)
            matches.append((score, session_copy))

        matches.sort(key=lambda item: (-item[0], str(item[1].get("updated_at", ""))), reverse=False)
        return [session for _, session in matches[: max(1, limit)]]

    def recent_session_candidates(self, limit_files: int = 4, max_turns: int = 24) -> list[str]:
        candidates: list[str] = []
        for session in self.list_sessions(limit=max(1, limit_files)):
            display_name = str(session.get("display_name", "Untitled session")).strip()
            updated_at = str(session.get("updated_at", "")).strip()
            records = self._read_session_records(self._session_path(session))
            pending_user: dict[str, Any] | None = None
            for record in records:
                role = str(record.get("role", "")).strip().lower()
                content = str(record.get("content", "")).strip()
                timestamp = str(record.get("timestamp", "")).strip() or updated_at
                if not content:
                    continue
                if role == "user":
                    pending_user = {"content": content, "timestamp": timestamp}
                    continue
                if role == "assistant" and pending_user is not None:
                    candidates.append(
                        (
                            f'Session "{display_name}" on {timestamp}: '
                            f'User asked "{pending_user["content"]}" and JARVIS answered "{content}".'
                        )
                    )
                    pending_user = None
        return candidates[-max(1, max_turns) :]

    @staticmethod
    def normalize_mode(mode: str | None) -> str:
        cleaned = re.sub(r"[^a-z0-9_-]+", "-", str(mode or "").strip().lower()).strip("-")
        return cleaned or "normal"

    def _migrate_legacy_json_sessions(self) -> None:
        with self._lock:
            index = self._load_index()
            changed = False
            sessions = index.setdefault("sessions", {})
            for legacy_path in sorted(self.sessions_dir.glob("session_*.json")):
                session_id = legacy_path.stem.replace("session_", "", 1).strip()
                if not session_id or session_id == "index":
                    continue
                if session_id in sessions:
                    continue
                payload = self._safe_json(legacy_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                created_at = str(payload.get("created_at", "")).strip() or self._mtime_iso(legacy_path)
                preferred_name = self._clean_session_name(
                    str(payload.get("display_name") or payload.get("name") or "").strip()
                )
                session = self._ensure_session_unlocked(
                    index,
                    session_id,
                    preferred_name=preferred_name,
                    mode="normal",
                    created_at=created_at,
                    auto_named=not bool(preferred_name),
                )
                messages = payload.get("messages", [])
                if isinstance(messages, list):
                    session_path = self._session_path(session)
                    if session_path.exists() and session_path.stat().st_size == 0:
                        with session_path.open("a", encoding="utf-8") as handle:
                            for message in messages:
                                if not isinstance(message, dict):
                                    continue
                                role = str(message.get("role", "")).strip().lower()
                                if role not in {"user", "assistant"}:
                                    continue
                                content = str(message.get("content", "")).strip()
                                if not content:
                                    continue
                                entry = {
                                    "type": "message",
                                    "session_id": session_id,
                                    "timestamp": str(message.get("timestamp", "")).strip() or created_at,
                                    "role": role,
                                    "content": content,
                                }
                                if role == "assistant" and message.get("tools_used"):
                                    entry["tools_used"] = message.get("tools_used")
                                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
                changed = True
            if changed:
                self._save_index(index)

    def _ensure_session_unlocked(
        self,
        index: dict[str, Any],
        session_id: str,
        preferred_name: str | None = None,
        mode: str | None = "normal",
        created_at: str | None = None,
        auto_named: bool | None = None,
    ) -> dict[str, Any]:
        sessions = index.setdefault("sessions", {})
        existing = sessions.get(session_id)
        if isinstance(existing, dict):
            if preferred_name:
                self._set_session_name_unlocked(existing, preferred_name, auto_named=bool(auto_named))
            normalized_mode = self.normalize_mode(mode) if mode is not None else None
            if normalized_mode and normalized_mode != existing.get("mode"):
                existing["mode"] = normalized_mode
                existing["updated_at"] = self._now_iso()
            return existing

        timestamp = created_at or self._now_iso()
        display_name = preferred_name or self._default_session_name(timestamp)
        auto_name_flag = auto_named if auto_named is not None else not bool(preferred_name)
        session = {
            "session_id": session_id,
            "display_name": display_name,
            "slug": self._slugify(display_name),
            "file_name": self._build_file_name(timestamp, display_name, session_id),
            "created_at": timestamp,
            "updated_at": timestamp,
            "mode": self.normalize_mode(mode),
            "auto_named": bool(auto_name_flag),
            "turn_count": 0,
            "first_user_message": "",
            "last_user_message": "",
            "last_assistant_message": "",
        }
        sessions[session_id] = session
        self._session_path(session).touch(exist_ok=True)
        return session

    def _set_session_name_unlocked(self, session: dict[str, Any], new_name: str, auto_named: bool) -> None:
        cleaned_name = self._clean_session_name(new_name)
        if not cleaned_name:
            return
        old_path = self._session_path(session)
        session["display_name"] = cleaned_name
        session["slug"] = self._slugify(cleaned_name)
        session["file_name"] = self._build_file_name(
            str(session.get("created_at", "")).strip() or self._now_iso(),
            cleaned_name,
            str(session.get("session_id", "")).strip(),
        )
        session["auto_named"] = bool(auto_named)
        session["updated_at"] = self._now_iso()
        new_path = self._session_path(session)
        if old_path != new_path and old_path.exists():
            old_path.rename(new_path)
        new_path.touch(exist_ok=True)

    def _maybe_autoname_unlocked(self, session: dict[str, Any], text: str) -> None:
        if not session.get("auto_named", True):
            return
        candidate = self._derive_name_from_text(text)
        if not candidate:
            return
        self._set_session_name_unlocked(session, candidate, auto_named=True)

    def _session_path(self, session: dict[str, Any]) -> Path:
        return self.sessions_dir / str(session.get("file_name", "")).strip()

    def _read_session_records(self, session_path: Path) -> list[dict[str, Any]]:
        if not session_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in session_path.read_text(encoding="utf-8").splitlines():
                payload = self._safe_json(line)
                if isinstance(payload, dict):
                    records.append(payload)
        except OSError:
            return []
        return records

    def _load_legacy_messages(self, session_id: str, limit_messages: int) -> list[dict[str, str]]:
        legacy_path = self.sessions_dir / f"session_{session_id}.json"
        if not legacy_path.exists():
            return []
        payload = self._safe_json(legacy_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return []
        collected = [
            {"role": str(message.get("role", "")).strip(), "content": str(message.get("content", "")).strip()}
            for message in messages
            if isinstance(message, dict)
            and str(message.get("role", "")).strip() in {"user", "assistant"}
            and str(message.get("content", "")).strip()
        ]
        return collected[-max(1, limit_messages) :]

    def _session_snippets(self, session: dict[str, Any], query_tokens: list[str]) -> list[str]:
        session_path = self._session_path(session)
        records = [
            record
            for record in self._read_session_records(session_path)
            if record.get("role") in {"user", "assistant"}
        ]
        if not records:
            return []

        scored_snippets: list[tuple[float, str]] = []
        for record in records[-24:]:
            content = str(record.get("content", "")).strip()
            if not content:
                continue
            score = self._score_text(" ".join(query_tokens), query_tokens, content)
            role = str(record.get("role", "")).strip().capitalize() or "Message"
            snippet = f"{role}: {self._truncate(content, 180)}"
            scored_snippets.append((score, snippet))
        scored_snippets.sort(key=lambda item: item[0], reverse=True)
        selected = [snippet for _, snippet in scored_snippets[:2] if snippet]
        if selected:
            return selected
        return [f'Last exchange from "{session.get("display_name", "Untitled session")}": {self._truncate(str(records[-1].get("content", "")), 180)}']

    def _build_file_name(self, created_at: str, display_name: str, session_id: str) -> str:
        safe_stamp = re.sub(r"[^0-9]", "", created_at)[:14] or datetime.now().strftime("%Y%m%d%H%M%S")
        slug = self._slugify(display_name)
        return f"{safe_stamp}__{slug}__{session_id}.jsonl"

    def _load_index(self) -> dict[str, Any]:
        payload = self._safe_json(self.index_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            sessions = payload.get("sessions")
            if isinstance(sessions, dict):
                payload["sessions"] = sessions
                return payload
        return {"sessions": {}}

    def _save_index(self, payload: dict[str, Any]) -> None:
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _default_session_name(created_at: str) -> str:
        stamp = SessionStore._format_timestamp(created_at)
        return f"Session {stamp}"

    @staticmethod
    def _format_timestamp(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
        cleaned = cleaned.strip("-")
        return cleaned or "session"

    @staticmethod
    def _clean_session_name(value: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        return cleaned[:80].strip()

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        cleaned = str(value).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(1, limit - 1)].rstrip() + "..."

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(value).lower())

    def _score_text(self, query_lower: str, query_tokens: list[str], candidate_text: str) -> float:
        candidate_lower = str(candidate_text).lower()
        candidate_tokens = set(self._tokens(candidate_lower))
        if not candidate_tokens:
            return 0.0
        score = 0.0
        if query_lower and query_lower in candidate_lower:
            score += 4.0
        for token in query_tokens:
            if token in candidate_tokens:
                score += 1.0
        return score

    def _derive_name_from_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text).strip())
        if not cleaned:
            return ""
        cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", cleaned)
        stopwords = {
            "a",
            "about",
            "an",
            "and",
            "for",
            "from",
            "i",
            "me",
            "my",
            "please",
            "tell",
            "the",
            "to",
            "we",
            "what",
        }
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'._-]*", cleaned)
        meaningful = [word for word in words if word.lower() not in stopwords]
        selection = meaningful[:4] or words[:4]
        if not selection:
            return ""
        name = " ".join(selection)
        name = re.sub(r"[_-]+", " ", name).strip()
        if len(name) > 48:
            name = name[:48].rsplit(" ", 1)[0].strip() or name[:48].strip()
        return name or ""

    @staticmethod
    def _safe_json(value: str) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().isoformat()

    @staticmethod
    def _mtime_iso(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
