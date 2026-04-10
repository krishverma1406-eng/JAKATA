"""Persistent logs and semantic memory helpers."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import io
import importlib
import json
import logging
import math
import re
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import (
    DATA_USER_DIR,
    MEMORY_LOGS_DIR,
    MEMORY_SESSIONS_DIR,
    SETTINGS,
    USER_ENTITIES_FILE,
    USER_CHUNKS_DIR,
    USER_MEMORY_RECORDS_FILE,
    USER_MEMORY_SUMMARY_CACHE_FILE,
    USER_VECTOR_DIR,
    Settings,
)
from core.session_store import SessionStore


class Memory:
    """Read and write conversation memory plus local retrieval data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.logs_dir = MEMORY_LOGS_DIR
        self.sessions_dir = MEMORY_SESSIONS_DIR
        self.chunks_dir = USER_CHUNKS_DIR
        self.vector_dir = USER_VECTOR_DIR
        self.profile_path = DATA_USER_DIR / "profile.md"
        self.projects_path = DATA_USER_DIR / "projects.md"
        self.records_path = USER_MEMORY_RECORDS_FILE
        self.entities_path = USER_ENTITIES_FILE
        self.daily_summary_path = USER_MEMORY_SUMMARY_CACHE_FILE
        self.index_state_path = self.vector_dir / "index_state.json"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.profile_path.touch()
        if not self.projects_path.exists():
            self.projects_path.touch()
        if not self.records_path.exists():
            self.records_path.write_text(json.dumps(self._default_records_payload(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        if not self.entities_path.exists():
            self.entities_path.write_text(json.dumps(self._default_entities_payload(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        if not self.daily_summary_path.exists():
            self.daily_summary_path.write_text(json.dumps({}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        self._embedder = None
        self._collection = None
        self._vector_disabled = False
        self._embedder_disabled = False
        self._daily_summary_cache: tuple[str, str] | None = None
        self._semantic_candidate_cache_key: tuple[str, ...] | None = None
        self._semantic_candidate_cache_embeddings: list[list[float]] | None = None
        self._query_embedding_cache: dict[tuple[str, ...], list[list[float]]] = {}
        self.session_store = SessionStore(self.sessions_dir)
        self._bootstrap_memory_store()
        threading.Thread(target=self._safe_ensure_index, daemon=True).start()

    def remember(
        self,
        text: str,
        brain: Any | None = None,
        *,
        source: str = "explicit",
    ) -> dict[str, Any]:
        cleaned = str(text).strip()
        if not cleaned:
            return {"ok": False, "stored": 0, "items": [], "error": "No memory text provided."}

        items = self._analyze_explicit_memory(cleaned, brain)
        stored_records = self._upsert_memory_items(items, source_type=source, explicit=True)
        self._rebuild_materialized_memory(stored_records_changed=True)
        return {
            "ok": True,
            "stored": len(stored_records),
            "items": [record["text"] for record in stored_records],
        }

    def forget(self, query: str, limit: int | None = None) -> dict[str, Any]:
        cleaned = str(query).strip()
        if not cleaned:
            return {"ok": False, "deleted": 0, "items": [], "error": "No forget query provided."}

        records_payload = self._load_records_payload()
        records = records_payload.get("records", [])
        if not isinstance(records, list):
            records = []

        matches = self._match_records_for_forget(cleaned, records, limit or self.settings.memory_top_k)
        if not matches:
            return {"ok": True, "deleted": 0, "items": []}

        delete_ids = {match["id"] for match in matches}
        updated_records = [record for record in records if str(record.get("id", "")) not in delete_ids]
        records_payload["records"] = updated_records
        records_payload["updated_at"] = self._now_iso()
        self._save_records_payload(records_payload)
        self._rebuild_materialized_memory(stored_records_changed=True)
        return {
            "ok": True,
            "deleted": len(matches),
            "items": [match["text"] for match in matches],
        }

    def entity_lookup(self, query: str, limit: int | None = None) -> dict[str, Any]:
        cleaned = str(query).strip()
        if not cleaned:
            return {"ok": False, "entities": [], "error": "No entity query provided."}

        entities_payload = self._load_entities_payload()
        entities = entities_payload.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        query_tokens = self._normalized_tokens(cleaned)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entity in entities.values():
            if not isinstance(entity, dict):
                continue
            candidate_bits = [str(entity.get("name", ""))]
            candidate_bits.extend(str(alias) for alias in entity.get("aliases", []) if alias)
            candidate_bits.extend(str(fact) for fact in entity.get("facts", []) if fact)
            candidate_text = " ".join(candidate_bits)
            score = self._token_overlap_score(" ".join(query_tokens), candidate_text)
            if score > 0:
                scored.append((score, entity))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [entity for _, entity in scored[: (limit or self.settings.memory_top_k)]]
        return {"ok": True, "entities": selected}

    def session_search(self, query: str, limit: int | None = None) -> dict[str, Any]:
        cleaned = str(query).strip()
        if not cleaned:
            return {"ok": False, "sessions": [], "error": "No session query provided."}
        sessions = self.session_store.search_sessions(cleaned, limit or self.settings.memory_top_k)
        return {"ok": True, "query": cleaned, "sessions": sessions}

    def ensure_session(
        self,
        session_id: str,
        preferred_name: str | None = None,
        mode: str = "normal",
    ) -> dict[str, Any]:
        return self.session_store.ensure_session(session_id, preferred_name=preferred_name, mode=mode)

    def rename_session(self, session_id: str, new_name: str) -> dict[str, Any]:
        return self.session_store.rename_session(session_id, new_name)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.session_store.get_session(session_id)

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.session_store.list_sessions(limit=limit)

    def set_session_mode(self, session_id: str, mode: str) -> dict[str, Any]:
        return self.session_store.set_mode(session_id, mode)

    def active_project_items(self, limit: int = 5) -> list[str]:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            return []
        seen: set[str] = set()
        items: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("tag", "")).strip().upper() != "PROJECT":
                continue
            if not record.get("active", True):
                continue
            if record.get("demoted"):
                continue
            text = str(record.get("text", "")).strip()
            normalized = text.lower()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            items.append(text)
            if len(items) >= max(1, limit):
                break
        return items

    def get_daily_context_summary(self, brain: Any | None = None) -> str:
        today = datetime.now().date().isoformat()
        if self._daily_summary_cache and self._daily_summary_cache[0] == today:
            return self._daily_summary_cache[1]

        payload = self._safe_json(self.daily_summary_path.read_text(encoding="utf-8")) or {}
        if (
            isinstance(payload, dict)
            and payload.get("date") == today
            and payload.get("source_signature") == self._daily_summary_source_signature()
        ):
            summary = str(payload.get("summary", "")).strip()
            self._daily_summary_cache = (today, summary)
            return summary

        summary = self._build_daily_context_summary(brain)
        cache_payload = {
            "date": today,
            "summary": summary,
            "source_signature": self._daily_summary_source_signature(),
            "updated_at": self._now_iso(),
        }
        self.daily_summary_path.write_text(
            json.dumps(cache_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self._daily_summary_cache = (today, summary)
        return summary

    def recall(self, query: str, limit: int | None = None) -> list[str]:
        limit = limit or self.settings.memory_top_k
        lowered = query.lower()
        query_variants = self._query_variants(query)
        semantic_ready = self._semantic_retrieval_ready()
        record_payload = self._load_records_payload()
        records = record_payload.get("records", [])
        if not isinstance(records, list):
            records = []
        active_records = [
            record
            for record in records
            if record.get("active", True) and not record.get("demoted")
        ]
        entity_candidates = self._entity_candidates(query)
        session_candidates = self._session_candidates(query, limit * 2)
        record_modifiers = self._record_text_modifiers(active_records)
        recent_candidates = self._merge_unique(
            self._recent_session_candidates(),
            self._recent_log_candidates(),
        )
        recent_query = self._is_recent_conversation_query(lowered)
        if recent_query and recent_candidates:
            recent_ranked, recent_best = self._rank_candidates(
                query_variants,
                recent_candidates,
                allow_semantic=semantic_ready,
            )
            if recent_ranked and recent_best >= 0.05:
                if len(recent_ranked) >= limit:
                    return recent_ranked[:limit]
                supplemental_ranked, _ = self._rank_candidates(
                    query_variants,
                    self._merge_unique(
                        self._candidate_pool(include_facts=True),
                        entity_candidates,
                        session_candidates,
                    ),
                    score_modifiers=record_modifiers,
                    allow_semantic=semantic_ready,
                )
                return self._merge_unique(recent_ranked, supplemental_ranked)[:limit]

        if any(marker in lowered for marker in ("know about me", "remember about me", "who am i", "my profile")):
            profile_items = self._merge_unique(self._candidate_pool(include_facts=False), entity_candidates, session_candidates)
            vector_hit_records = (
                self._merge_unique_records(*[self._query_vector_store(variant, limit * 2) for variant in query_variants])
                if semantic_ready
                else []
            )
            vector_hits = [item["text"] for item in vector_hit_records]
            ranked, _ = self._rank_candidates(
                query_variants,
                self._merge_unique(profile_items, vector_hits),
                vector_hits=set(vector_hits),
                score_modifiers=record_modifiers,
                allow_semantic=semantic_ready,
            )
            self._mark_records_retrieved(ranked[:limit], records)
            return ranked[:limit]

        summary_candidates = self._merge_unique(self._candidate_pool(include_facts=False), entity_candidates, session_candidates)
        if recent_query:
            summary_candidates = self._merge_unique(recent_candidates, summary_candidates)
        summary_ranked, summary_best = self._rank_candidates(
            query_variants,
            summary_candidates,
            score_modifiers=record_modifiers,
            allow_semantic=semantic_ready,
        )
        if summary_ranked and summary_best >= 0.18:
            self._mark_records_retrieved(summary_ranked[:limit], records)
            return summary_ranked[:limit]

        full_candidates = self._merge_unique(self._candidate_pool(include_facts=True), entity_candidates, session_candidates)
        if recent_candidates:
            full_candidates = self._merge_unique(recent_candidates, full_candidates)
        vector_hit_records = (
            self._merge_unique_records(*[self._query_vector_store(variant, limit * 2) for variant in query_variants])
            if semantic_ready
            else []
        )
        vector_hits = [item["text"] for item in vector_hit_records]
        ranked, best_score = self._rank_candidates(
            query_variants,
            self._merge_unique(full_candidates, vector_hits),
            vector_hits=set(vector_hits),
            score_modifiers=record_modifiers,
            allow_semantic=semantic_ready,
        )
        if ranked and (best_score >= 0.08 or vector_hits):
            self._mark_records_retrieved(ranked[:limit], records)
            return ranked[:limit]

        fallback_hits, _ = self._fast_recall(query_variants[0], limit, include_facts=True)
        self._mark_records_retrieved(fallback_hits, records)
        return fallback_hits

    def persist_conversation(
        self,
        user_message: str,
        assistant_message: str,
        conversation: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        session_id: str | None = None,
        brain: Any | None = None,
        should_extract: bool = True,
        background: bool = False,
        turn_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._log_message("user", user_message, tool_trace)
        self._log_message("assistant", assistant_message, tool_trace)
        self._append_session_turns(session_id, user_message, assistant_message, tool_trace, turn_meta=turn_meta)

        if not should_extract:
            return {"stored": 0, "chunks": []}

        if background:
            threading.Thread(
                target=self._persist_extracted_memory,
                args=(conversation, brain),
                daemon=True,
            ).start()
            return {"stored": 0, "chunks": []}

        return self._persist_extracted_memory(conversation, brain)

    def _persist_extracted_memory(
        self,
        conversation: list[dict[str, Any]],
        brain: Any | None,
    ) -> dict[str, Any]:
        extracted_items = self._extract_memories(conversation, brain)
        if not extracted_items:
            return {"stored": 0, "chunks": []}

        stored_records = self._upsert_memory_items(
            extracted_items,
            source_type="conversation_extract_v2",
            explicit=False,
        )
        if not stored_records:
            return {"stored": 0, "chunks": []}

        # Do NOT write chunk files — records are the source of truth.
        # Chunk files are legacy write-logs that cause stale data in recall and
        # trigger unnecessary re-indexing via _memory_source_state.
        self._rebuild_materialized_memory(stored_records_changed=True)
        return {"stored": len(stored_records), "chunks": []}

    def load_recent_messages(self, limit_messages: int = 12) -> list[dict[str, str]]:
        collected: list[dict[str, str]] = []
        for session in self.session_store.list_sessions(limit=4):
            session_id = str(session.get("session_id", "")).strip()
            if not session_id:
                continue
            for message in reversed(self.session_store.load_messages(session_id, limit_messages=limit_messages)):
                if len(collected) >= limit_messages:
                    break
                role = str(message.get("role", "")).strip().lower()
                content = str(message.get("content", "")).strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                collected.append({"role": role, "content": content})
            if len(collected) >= limit_messages:
                break
        if not collected:
            return self._recent_log_messages(limit_messages)
        collected.reverse()
        return collected

    def load_session_messages(self, session_id: str, limit_messages: int = 12) -> list[dict[str, str]]:
        if not session_id:
            return self.load_recent_messages(limit_messages)
        messages = self.session_store.load_messages(session_id, limit_messages=limit_messages)
        return messages

    def ensure_index_current(self) -> None:
        current_state = self._memory_source_state()
        if current_state == self._read_index_state():
            return
        items = self._collect_memory_items_from_disk()
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None:
            self._write_index_state(current_state)
            return

        try:
            existing = collection.get(include=[])
            existing_ids = {
                str(item_id).strip()
                for item_id in (existing.get("ids", []) or [])
                if str(item_id).strip()
            }
        except Exception:
            existing_ids = set()

        item_map = {
            str(item.get("id", "")).strip(): item
            for item in items
            if str(item.get("id", "")).strip()
        }
        item_ids = set(item_map)

        stale_ids = sorted(existing_ids - item_ids)
        if stale_ids:
            with contextlib.suppress(Exception):
                collection.delete(ids=stale_ids)

        new_items = [item_map[item_id] for item_id in sorted(item_ids - existing_ids)]
        if new_items:
            try:
                documents = [item["text"] for item in new_items]
                embeddings = embedder.encode(documents).tolist()
                collection.add(
                    ids=[item["id"] for item in new_items],
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "tag": item["tag"],
                            "created_at": datetime.now().isoformat(),
                            "source": item.get("source", "records"),
                            "record_id": item["id"],
                        }
                        for item in new_items
                    ],
                )
            except Exception:
                # Write state anyway so we don't retry on every call and block forever.
                # The next records change will unlink index_state and trigger a fresh attempt.
                self._write_index_state(current_state)
                return

        self._write_index_state(current_state)

    def _safe_ensure_index(self) -> None:
        """Safely run ensure_index_current with exception handling for background threads."""
        try:
            self.ensure_index_current()
        except Exception:
            pass

    def _log_message(
        self,
        role: str,
        content: str,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "tools_used": [item.get("name") for item in (tool_trace or []) if item.get("name")],
        }
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _append_chunk(self, lines: list[str]) -> Path:
        chunk_path = self.chunks_dir / f"{datetime.now().strftime('%Y-%m-%d')}.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with chunk_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}]\n")
            for line in lines:
                handle.write(f"{line}\n")
            handle.write("\n")
        return chunk_path

    def _append_session_turns(
        self,
        session_id: str | None,
        user_message: str,
        assistant_message: str,
        tool_trace: list[dict[str, Any]] | None = None,
        turn_meta: dict[str, Any] | None = None,
    ) -> None:
        if not session_id:
            return
        self.session_store.append_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_trace=tool_trace,
            turn_meta=turn_meta,
        )

    def _extract_memories(
        self,
        conversation: list[dict[str, Any]],
        brain: Any | None,
    ) -> list[dict[str, Any]]:
        transcript = []
        for message in conversation:
            role = message.get("role", "unknown").upper()
            content = str(message.get("content", "")).strip()
            if content:
                transcript.append(f"{role}: {content}")

        if not transcript:
            return []

        return self._fallback_extract(transcript)

    def _normalize_extraction_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        raw_items = payload.get("items")
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                normalized = self._normalize_memory_item(raw_item)
                if normalized is not None:
                    items.append(normalized)
        if items:
            return items

        for tag in ("PERSONAL", "PROJECT", "FACT"):
            values = payload.get(tag, [])
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value).strip()
                if text:
                    items.append(
                        {
                            "tag": tag,
                            "text": text,
                            "memory_key": self._default_memory_key(tag, text),
                            "confidence": 8,
                            "importance": 8,
                            "entities": self._extract_entities_fallback(text),
                            "relations": [],
                        }
                    )
        return items

    def _fallback_extract(self, transcript: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        existing_texts = {
            str(record.get("text", "")).strip().lower()
            for record in records
            if record.get("active", True) and not record.get("demoted")
        }
        for line in transcript:
            lowered = line.lower()
            if not lowered.startswith("user:"):
                continue
            content = line[5:].strip()
            if not self._looks_like_durable_user_fact(content):
                continue
            if content.lower() in existing_texts:
                continue
            if any(marker in lowered for marker in ("i am ", "my ", "i like ", "i prefer ", "i want ", "remember ")):
                items.append(
                    {
                        "tag": "PERSONAL",
                        "text": content,
                        "memory_key": self._default_memory_key("PERSONAL", content),
                        "confidence": 8,
                        "importance": 8,
                        "entities": self._extract_entities_fallback(content),
                        "relations": [],
                    }
                )
            elif any(marker in lowered for marker in ("project", "build", "working on", "developing", "making", "jarvis")):
                items.append(
                    {
                        "tag": "PROJECT",
                        "text": content,
                        "memory_key": self._default_memory_key("PROJECT", content),
                        "confidence": 8,
                        "importance": 8,
                        "entities": self._extract_entities_fallback(content),
                        "relations": [],
                    }
                )
        return items

    def _default_records_payload(self) -> dict[str, Any]:
        return {"version": 2, "updated_at": self._now_iso(), "records": []}

    def _default_entities_payload(self) -> dict[str, Any]:
        return {"version": 1, "updated_at": self._now_iso(), "entities": {}}

    def _bootstrap_memory_store(self) -> None:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        if not records:
            records = self._migrate_legacy_files_to_records()
            payload["records"] = records
            payload["updated_at"] = self._now_iso()
            self._save_records_payload(payload)
        else:
            records = self._sync_external_memory_files(records)
            records = self._apply_memory_decay(records)
            records = self._sanitize_records(records)
            payload["records"] = records
            payload["updated_at"] = self._now_iso()
            self._save_records_payload(payload)
        self._rebuild_entities(records)
        self._rewrite_summary_docs_from_records(records)

    def _load_records_payload(self) -> dict[str, Any]:
        payload = self._safe_json(self.records_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return self._default_records_payload()
        payload.setdefault("records", [])
        return payload

    def _save_records_payload(self, payload: dict[str, Any]) -> None:
        self.records_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_entities_payload(self) -> dict[str, Any]:
        payload = self._safe_json(self.entities_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return self._default_entities_payload()
        payload.setdefault("entities", {})
        return payload

    def _save_entities_payload(self, payload: dict[str, Any]) -> None:
        self.entities_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _now_iso(self) -> str:
        return datetime.now().astimezone().isoformat()

    def _coerce_score(self, value: Any, default: int = 7) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, min(score, 10))

    def _sanitize_tag(self, value: Any) -> str:
        tag = str(value or "FACT").strip().upper()
        return tag if tag in {"PERSONAL", "PROJECT", "FACT"} else "FACT"

    def _memory_topic_key(self, tag: str, text: str) -> str | None:
        lowered = str(text).strip().lower()
        if not lowered:
            return None

        if tag == "PERSONAL":
            if "@" in lowered or "email" in lowered:
                return "email"
            if "github username" in lowered or "github" in lowered:
                return "github"
            if any(marker in lowered for marker in ("school", "studies at", "study at", "studying in")):
                return "school"
            if any(marker in lowered for marker in ("my name", "name is ", "user's name", "the user name")):
                return "name"
            if any(marker in lowered for marker in ("live in", "lives in", "living in", "originally from", "from hisar", "from delhi")):
                return "location"
            if any(marker in lowered for marker in ("class ", "grade ", "11th", "12th", "10th")):
                return "class_grade"
        return None

    def _default_memory_key(self, tag: str, text: str) -> str:
        topic_key = self._memory_topic_key(tag, text)
        if topic_key:
            key_basis = topic_key
        else:
            normalized_tokens = sorted(self._normalized_tokens(text))
            normalized = " ".join(normalized_tokens) or re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            key_basis = "-".join(normalized.split()[:6]) or uuid.uuid4().hex[:12]
        return f"{tag.lower()}.{key_basis}"

    def _normalize_memory_item(self, raw_item: Any) -> dict[str, Any] | None:
        if not isinstance(raw_item, dict):
            return None
        text = str(raw_item.get("text", "")).strip()
        if not text:
            return None
        tag = self._sanitize_tag(raw_item.get("tag"))
        entities = raw_item.get("entities", [])
        relations = raw_item.get("relations", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []
        normalized_entities = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            normalized_entities.append(
                {
                    "name": name,
                    "type": str(entity.get("type", "UNKNOWN")).strip().upper() or "UNKNOWN",
                }
            )
        if not normalized_entities:
            normalized_entities = self._extract_entities_fallback(text)
        normalized_relations = []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            predicate = str(relation.get("predicate", "")).strip()
            target = str(relation.get("target", "")).strip()
            if source and predicate and target:
                normalized_relations.append(
                    {"source": source, "predicate": predicate, "target": target}
                )
        return {
            "tag": tag,
            "text": text,
            "memory_key": str(raw_item.get("memory_key", "")).strip() or self._default_memory_key(tag, text),
            "confidence": self._coerce_score(raw_item.get("confidence"), 7),
            "importance": self._coerce_score(raw_item.get("importance"), 7),
            "entities": normalized_entities,
            "relations": normalized_relations,
        }

    def _extract_entities_fallback(self, text: str) -> list[dict[str, str]]:
        entities: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            name = match.group(1).strip()
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            entities.append({"name": name, "type": "UNKNOWN"})
        return entities

    def _analyze_explicit_memory(self, text: str, brain: Any | None) -> list[dict[str, Any]]:
        if brain is not None:
            prompt = (
                "You are storing an explicit user memory.\n"
                "Return strict JSON with key items, where items is a list of memory objects.\n"
                "Each memory object must contain tag, text, memory_key, confidence, importance, entities, relations.\n"
                "confidence and importance must both be 10 unless the statement is ambiguous.\n"
                "Preserve the user's meaning exactly."
            )
            response = brain.chat(
                messages=[{"role": "user", "content": text}],
                task_kind="memory",
                response_format="json",
                system_override=prompt,
            )
            payload = self._safe_json(response.get("content", ""))
            if isinstance(payload, dict):
                normalized = self._normalize_extraction_payload(payload)
                if normalized:
                    return normalized

        return [
            {
                "tag": "FACT",
                "text": text,
                "memory_key": self._default_memory_key("FACT", text),
                "confidence": 10,
                "importance": 10,
                "entities": self._extract_entities_fallback(text),
                "relations": [],
            }
        ]

    def _migrate_legacy_files_to_records(self) -> list[dict[str, Any]]:
        items = self._legacy_disk_items()
        records: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for item in items:
            normalized = self._normalize_memory_item(item)
            if normalized is None or normalized["text"] in seen_texts:
                continue
            seen_texts.add(normalized["text"])
            records.append(
                {
                    "id": f"memory_{uuid.uuid4().hex}",
                    **normalized,
                    "active": True,
                    "demoted": False,
                    "explicit": item.get("source_type") == "manual_file",
                    "source_type": item.get("source_type", "legacy_import"),
                    "source_ref": item.get("source_ref", ""),
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                    "last_retrieved_at": None,
                    "retrieval_count": 0,
                    "superseded_by": None,
                }
            )
        return records

    def _sync_external_memory_files(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        known_texts = {str(record.get("text", "")).strip() for record in records}
        for item in self._legacy_disk_items():
            normalized = self._normalize_memory_item(item)
            if normalized is None or normalized["text"] in known_texts:
                continue
            known_texts.add(normalized["text"])
            records.append(
                {
                    "id": f"memory_{uuid.uuid4().hex}",
                    **normalized,
                    "active": True,
                    "demoted": False,
                    "explicit": item.get("source_type") == "manual_file",
                    "source_type": item.get("source_type", "manual_file"),
                    "source_ref": item.get("source_ref", ""),
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                    "last_retrieved_at": None,
                    "retrieval_count": 0,
                    "superseded_by": None,
                }
            )
        return records

    def _legacy_disk_items(self) -> list[dict[str, Any]]:
        # profile.md, projects.md, and chunk files are derived artifacts generated from
        # memory_records.json, so importing them back would re-poison the source of truth.
        return []

    def _upsert_memory_items(
        self,
        items: list[dict[str, Any]],
        *,
        source_type: str,
        explicit: bool,
    ) -> list[dict[str, Any]]:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []

        accepted_records: list[dict[str, Any]] = []
        for item in items:
            normalized = self._normalize_memory_item(item)
            if normalized is None:
                continue
            if not explicit and normalized["confidence"] < self.settings.memory_confidence_threshold:
                continue
            if not explicit and not self._is_safe_auto_memory(normalized["text"], normalized["tag"]):
                continue

            existing_exact = next(
                (
                    record for record in records
                    if str(record.get("text", "")).strip() == normalized["text"]
                    and str(record.get("memory_key", "")).strip() == normalized["memory_key"]
                ),
                None,
            )
            if existing_exact is not None:
                existing_exact["confidence"] = max(int(existing_exact.get("confidence", 0)), normalized["confidence"])
                existing_exact["importance"] = max(int(existing_exact.get("importance", 0)), normalized["importance"])
                existing_exact["entities"] = normalized["entities"]
                existing_exact["relations"] = normalized["relations"]
                existing_exact["active"] = True
                existing_exact["demoted"] = False
                existing_exact["updated_at"] = self._now_iso()
                accepted_records.append(existing_exact)
                continue

            new_id = f"memory_{uuid.uuid4().hex}"
            conflicting = self._find_conflicting_record(normalized["text"], normalized["tag"], records)
            if conflicting is not None:
                conflicting["active"] = False
                conflicting["demoted"] = True
                conflicting["superseded_by"] = new_id
                conflicting["updated_at"] = self._now_iso()
            for record in records:
                if (
                    record.get("active", True)
                    and not record.get("demoted")
                    and str(record.get("memory_key", "")).strip()
                    and str(record.get("memory_key", "")).strip() == normalized["memory_key"]
                    and str(record.get("text", "")).strip() != normalized["text"]
                ):
                    record["active"] = False
                    record["demoted"] = True
                    record["superseded_by"] = new_id
                    record["updated_at"] = self._now_iso()

            new_record = {
                "id": new_id,
                **normalized,
                "active": True,
                "demoted": False,
                "explicit": explicit,
                "source_type": source_type,
                "source_ref": "",
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "last_retrieved_at": None,
                "retrieval_count": 0,
                "superseded_by": None,
            }
            records.append(new_record)
            accepted_records.append(new_record)

        payload["records"] = self._apply_memory_decay(records)
        payload["updated_at"] = self._now_iso()
        self._save_records_payload(payload)
        return accepted_records

    def _find_conflicting_record(
        self,
        new_text: str,
        tag: str,
        records: list[dict[str, Any]],
        overlap_threshold: float = 0.55,
    ) -> dict[str, Any] | None:
        new_tokens = self._normalized_tokens(new_text)
        new_topic = self._memory_topic_key(tag, new_text)
        if len(new_tokens) < 2 and not new_topic:
            return None

        best_score = 0.0
        best_record: dict[str, Any] | None = None
        for record in records:
            if not record.get("active", True) or record.get("demoted"):
                continue
            if self._sanitize_tag(record.get("tag")) != tag:
                continue
            existing_text = str(record.get("text", "")).strip()
            if not existing_text or existing_text == new_text:
                continue
            existing_topic = self._memory_topic_key(tag, existing_text)
            if new_topic and existing_topic and new_topic == existing_topic:
                return record
            existing_tokens = self._normalized_tokens(existing_text)
            if not existing_tokens:
                continue
            intersection = len(new_tokens & existing_tokens)
            union = len(new_tokens | existing_tokens)
            overlap = (intersection / union) if union else 0.0
            if overlap > overlap_threshold and overlap > best_score:
                best_score = overlap
                best_record = record
        return best_record

    def _sanitize_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for record in records:
            tag = self._sanitize_tag(record.get("tag"))
            text = str(record.get("text", "")).strip()
            if not text:
                record["active"] = False
                record["demoted"] = True
                continue
            if record.get("explicit"):
                continue
            if not self._is_safe_auto_memory(text, tag):
                record["active"] = False
                record["demoted"] = True
        return records

    def _apply_memory_decay(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cutoff = datetime.now().astimezone() - timedelta(days=self.settings.memory_decay_days)
        for record in records:
            if not record.get("active", True):
                continue
            if record.get("explicit"):
                continue
            last_retrieved_at = record.get("last_retrieved_at") or record.get("updated_at") or record.get("created_at")
            try:
                last_seen = datetime.fromisoformat(str(last_retrieved_at))
            except ValueError:
                continue
            if last_seen.tzinfo is None:
                last_seen = last_seen.astimezone()
            if last_seen < cutoff and int(record.get("retrieval_count", 0)) == 0:
                record["demoted"] = True
        return records

    def _rebuild_materialized_memory(self, *, stored_records_changed: bool) -> None:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        records = self._apply_memory_decay(records)
        payload["records"] = records
        payload["updated_at"] = self._now_iso()
        self._save_records_payload(payload)
        self._rebuild_entities(records)
        self._rewrite_summary_docs_from_records(records)
        self._semantic_candidate_cache_key = None
        self._semantic_candidate_cache_embeddings = None
        self._query_embedding_cache = {}
        if stored_records_changed:
            with contextlib.suppress(OSError):
                self.index_state_path.unlink()
            threading.Thread(target=self._safe_ensure_index, daemon=True).start()

    def _rewrite_summary_docs_from_records(self, records: list[dict[str, Any]]) -> None:
        personal_lines = [
            str(record.get("text", "")).strip()
            for record in records
            if record.get("active", True)
            and not record.get("demoted")
            and self._sanitize_tag(record.get("tag")) == "PERSONAL"
        ]
        project_lines = [
            str(record.get("text", "")).strip()
            for record in records
            if record.get("active", True)
            and not record.get("demoted")
            and self._sanitize_tag(record.get("tag")) == "PROJECT"
        ]
        # Remove fallback to chunks - records are the only source of truth
        self._rewrite_summary_file(self.profile_path, "## User Profile", personal_lines)
        self._rewrite_summary_file(self.projects_path, "## Active Projects", project_lines)

    def _rewrite_summary_file(self, path: Path, title: str, lines: list[str]) -> None:
        unique_lines = self._merge_unique(lines)
        output = [title, ""]
        output.extend(f"- {line}" for line in unique_lines)
        path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")

    def _rebuild_entities(self, records: list[dict[str, Any]]) -> None:
        entities: dict[str, dict[str, Any]] = {}
        for record in records:
            if not record.get("active", True):
                continue
            if record.get("demoted"):
                continue
            record_id = str(record.get("id", ""))
            facts = [str(record.get("text", "")).strip()]
            for entity in record.get("entities", []) or []:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name", "")).strip()
                if not name:
                    continue
                slug = self._entity_slug(name)
                node = entities.setdefault(
                    slug,
                    {
                        "name": name,
                        "type": str(entity.get("type", "UNKNOWN")).strip().upper() or "UNKNOWN",
                        "aliases": [name],
                        "facts": [],
                        "memory_ids": [],
                        "relations": [],
                    },
                )
                if name not in node["aliases"]:
                    node["aliases"].append(name)
                if record_id and record_id not in node["memory_ids"]:
                    node["memory_ids"].append(record_id)
                for fact in facts:
                    if fact and fact not in node["facts"]:
                        node["facts"].append(fact)
            for relation in record.get("relations", []) or []:
                if not isinstance(relation, dict):
                    continue
                source_name = str(relation.get("source", "")).strip()
                predicate = str(relation.get("predicate", "")).strip()
                target_name = str(relation.get("target", "")).strip()
                if not (source_name and predicate and target_name):
                    continue
                source_node = entities.setdefault(
                    self._entity_slug(source_name),
                    {
                        "name": source_name,
                        "type": "UNKNOWN",
                        "aliases": [source_name],
                        "facts": [],
                        "memory_ids": [],
                        "relations": [],
                    },
                )
                relation_payload = {
                    "predicate": predicate,
                    "target": target_name,
                    "memory_id": record_id,
                }
                if relation_payload not in source_node["relations"]:
                    source_node["relations"].append(relation_payload)

        payload = self._default_entities_payload()
        payload["updated_at"] = self._now_iso()
        payload["entities"] = entities
        self._save_entities_payload(payload)

    def _entity_slug(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or uuid.uuid4().hex[:8]

    def _build_daily_context_summary(self, brain: Any | None) -> str:
        profile_lines = self._summary_candidates_from_file(self.profile_path)
        project_lines = self._summary_candidates_from_file(self.projects_path)
        if not profile_lines and not project_lines:
            # profile.md/projects.md are derived from records — if they're empty,
            # read directly from records rather than falling back to stale chunk files.
            payload = self._load_records_payload()
            records = payload.get("records", [])
            if isinstance(records, list):
                profile_lines = [
                    str(r.get("text", "")).strip()
                    for r in records
                    if r.get("active", True) and not r.get("demoted")
                    and self._sanitize_tag(r.get("tag")) == "PERSONAL"
                    and str(r.get("text", "")).strip()
                ]
                project_lines = [
                    str(r.get("text", "")).strip()
                    for r in records
                    if r.get("active", True) and not r.get("demoted")
                    and self._sanitize_tag(r.get("tag")) == "PROJECT"
                    and str(r.get("text", "")).strip()
                ]
        if not profile_lines and not project_lines:
            return ""

        context_block = [
            "Profile:",
            *[f"- {line}" for line in profile_lines[:10]],
            "",
            "Projects:",
            *[f"- {line}" for line in project_lines[:10]],
        ]
        if brain is not None:
            prompt = (
                "Create a concise 2-sentence internal summary of this user's durable context.\n"
                "Focus on identity, location, preferences, and active project direction.\n"
                "Do not mention that this is a summary."
            )
            response = brain.chat(
                messages=[{"role": "user", "content": "\n".join(context_block)}],
                task_kind="memory",
                system_override=prompt,
            )
            summary = str(response.get("content", "")).strip()
            if summary:
                return summary

        blended = self._merge_unique(profile_lines[:2], project_lines[:2])
        if not blended:
            return ""
        return " ".join(blended[:2])

    def _daily_summary_source_signature(self) -> str:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        active_records = [
            {
                "id": record.get("id"),
                "updated_at": record.get("updated_at"),
                "text": record.get("text"),
            }
            for record in records
            if record.get("active", True)
        ]
        return json.dumps(active_records, ensure_ascii=True, sort_keys=True)

    def _match_records_for_forget(
        self,
        query: str,
        records: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        candidates = [record for record in records if record.get("active", True)]
        scored: list[tuple[float, dict[str, Any]]] = []
        query_lower = query.lower().strip()
        query_tokens = self._normalized_tokens(query)
        for record in candidates:
            text = str(record.get("text", "")).strip()
            lowered_text = text.lower()
            overlap_score = self._token_overlap_score(query, text)
            sequence_score = difflib.SequenceMatcher(None, query_lower, lowered_text).ratio()
            contains_query = query_lower in lowered_text
            candidate_tokens = self._normalized_tokens(text)
            token_fraction = (
                len(query_tokens & candidate_tokens) / len(query_tokens)
                if query_tokens else 0.0
            )
            score = max(overlap_score, sequence_score)
            if contains_query:
                score += 0.5
            if token_fraction >= 0.9:
                score += 0.25
            is_long_query = len(query_tokens) >= 4
            if contains_query or (is_long_query and sequence_score >= 0.86) or (not is_long_query and sequence_score >= 0.72) or token_fraction >= (0.9 if is_long_query else 0.8):
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def _query_vector_store(self, query: str, limit: int) -> list[dict[str, Any]]:
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None:
            return []

        results = collection.query(
            query_embeddings=embedder.encode([query]).tolist(),
            n_results=limit,
        )
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if not documents or not ids:
            return []
        rows: list[dict[str, Any]] = []
        metadata_rows = metadatas[0] if metadatas else [None] * len(documents[0])
        for item_id, doc, metadata in zip(ids[0], documents[0], metadata_rows):
            if isinstance(doc, str):
                rows.append({"id": item_id, "text": doc, "metadata": metadata or {}})
        return rows

    def _summary_candidates_from_file(self, path: Path) -> list[str]:
        candidates: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if cleaned.startswith("- "):
                candidates.append(cleaned[2:].strip())
            else:
                candidates.append(cleaned)
        return candidates

    def _looks_like_durable_user_fact(self, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered or "?" in lowered:
            return False
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "open ",
                "search ",
                "play ",
                "list ",
                "show ",
                "tell me",
                "can you",
                "could you",
                "please ",
                "pls ",
                "what ",
                "who ",
                "where ",
                "when ",
                "why ",
                "how ",
            )
        ):
            return False
        return True

    def _is_safe_auto_memory(self, text: str, tag: str) -> bool:
        lowered = str(text).strip().lower()
        if not lowered:
            return False
        if any(
            marker in lowered
            for marker in (
                "assistant can ",
                "jarvis can ",
                "user asked",
                "assistant acknowledged",
                "current session",
                "session id",
                "current weather",
                "weather in ",
                "current time",
                "current date",
                "cpu",
                "ram",
                "disk",
                "battery",
                "downloads directory",
                "email was sent",
                "message id",
                "bookmarked",
                "unread notifications",
                "no scheduled events",
                "system has ",
                "user's system",
                "screenshot",
                "browser",
                "http://",
                "https://",
            )
        ):
            return False
        if "\\" in lowered and ":" in lowered:
            return False

        if tag == "PERSONAL":
            return any(
                marker in lowered
                for marker in (
                    "i am ",
                    "my ",
                    "i like ",
                    "i prefer ",
                    "i want ",
                    "name is ",
                    "my name",
                    "email",
                    "@",
                    "i live",
                    "live in",
                    "lives in",
                    "from ",
                    "originally from",
                    "studies at",
                    "studying in",
                    "school",
                    "class ",
                    "grade ",
                    "github",
                    "stud",
                    "enjoy",
                    "hate ",
                    "prefer",
                    "like ",
                )
            )
        if tag == "PROJECT":
            return any(
                marker in lowered
                for marker in (
                    "project",
                    "working on",
                    "building",
                    "developing",
                    "making",
                    "app",
                    "website",
                    "assistant",
                    "jarvis",
                    "yantra",
                )
            )
        return False

    def _is_trusted_context_candidate(self, text: str, tag: str) -> bool:
        lowered = str(text).strip().lower()
        if not lowered:
            return False
        if any(
            marker in lowered
            for marker in (
                "assistant can ",
                "jarvis can ",
                "current session",
                "session id",
                "current weather",
                "current time",
                "current date",
                "cpu",
                "ram",
                "disk",
                "battery",
                "downloads directory",
                "email was sent",
                "message id",
                "bookmarked",
                "unread notifications",
                "screenshot",
                "browser",
                "calendar for ",
                "system has ",
                "user asked to open",
                "samay raina",
                "india's got latent",
                "comicstaan",
                "instagram has",
                "youtube has",
                "openai released",
                "dall-e",
                "whisper v3",
                "sora,",
                "vance to lead",
                "contest",
                "article",
            )
        ):
            return False
        if "\\" in lowered and ":" in lowered and "jarvis" not in lowered and "yantra" not in lowered:
            return False
        return self._is_safe_auto_memory(text, tag)

    def _trusted_chunk_candidates(self, tag: str | None = None) -> list[str]:
        candidates: list[str] = []
        for chunk_file in sorted(self.chunks_dir.glob("*.txt"), reverse=True):
            for item in self._chunk_items_from_file(chunk_file):
                item_tag = self._sanitize_tag(item.get("tag"))
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                if tag is not None and item_tag != tag:
                    continue
                if item_tag not in {"PERSONAL", "PROJECT"}:
                    continue
                if not self._is_trusted_context_candidate(text, item_tag):
                    continue
                candidates.append(text)
        return self._merge_unique(candidates)

    def _token_overlap_score(self, query: str, candidate: str) -> float:
        query_tokens = self._normalized_tokens(query)
        candidate_tokens = self._normalized_tokens(candidate)
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        return overlap / math.sqrt(len(query_tokens) * len(candidate_tokens))

    def _fast_recall(self, query: str, limit: int, include_facts: bool) -> tuple[list[str], float]:
        candidates = self._candidate_pool(include_facts)
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            score = self._token_overlap_score(query, candidate)
            if score > 0:
                scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return [], 0.0
        return [candidate for _, candidate in scored[:limit]], scored[0][0]

    def _query_variants(self, query: str) -> list[str]:
        normalized = self._normalize_query_against_memory(query)
        variants = [query.strip()]
        if normalized and normalized not in variants:
            variants.append(normalized)
        return [variant for variant in variants if variant]

    def _normalize_query_against_memory(self, query: str) -> str:
        vocabulary = self._memory_vocabulary()
        if not vocabulary:
            return query

        corrected_tokens: list[str] = []
        changed = False
        for token in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", query):
            if not re.fullmatch(r"[A-Za-z0-9]+", token):
                corrected_tokens.append(token)
                continue
            lowered = token.lower()
            normalized = self._normalize_token(lowered)
            if len(lowered) < 4 or lowered in vocabulary or normalized in self._memory_stopwords():
                corrected_tokens.append(token)
                continue
            match = difflib.get_close_matches(lowered, list(vocabulary), n=1, cutoff=0.7)
            if match:
                corrected_tokens.append(match[0])
                changed = True
            else:
                corrected_tokens.append(token)
        normalized = "".join(corrected_tokens).strip()
        return normalized if changed else query

    def _memory_vocabulary(self) -> set[str]:
        vocab: set[str] = set()
        candidates = self._merge_unique(
            self._candidate_pool(include_facts=True),
            self._entity_candidates(""),
        )
        for candidate in candidates:
            for token in re.findall(r"[a-zA-Z0-9]+", candidate.lower()):
                normalized = self._normalize_token(token)
                if len(normalized) >= 4:
                    vocab.add(normalized)
        return vocab

    def _merge_unique(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                cleaned = str(item).strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                merged.append(cleaned)
        return merged

    def _merge_unique_records(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                if not isinstance(item, dict):
                    continue
                identity = str(item.get("id", "")).strip() or str(item.get("text", "")).strip()
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                merged.append(item)
        return merged

    def _candidate_pool(self, include_facts: bool) -> list[str]:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        candidates: list[str] = []
        for record in records:
            if not record.get("active", True):
                continue
            if record.get("demoted"):
                continue
            if not include_facts and self._sanitize_tag(record.get("tag")) == "FACT":
                continue
            text = str(record.get("text", "")).strip()
            if text:
                candidates.append(text)
        # DO NOT read chunks - they are derived artifacts, records are source of truth
        return self._merge_unique(candidates)

    def _entity_candidates(self, query: str) -> list[str]:
        payload = self._load_entities_payload()
        entities = payload.get("entities", {})
        if not isinstance(entities, dict):
            return []
        query_tokens = self._normalized_tokens(query)
        candidates: list[str] = []
        for entity in entities.values():
            if not isinstance(entity, dict):
                continue
            search_blob = " ".join(
                [
                    str(entity.get("name", "")).strip(),
                    *[str(alias).strip() for alias in entity.get("aliases", []) if alias],
                    *[str(fact).strip() for fact in entity.get("facts", []) if fact],
                ]
            )
            if query_tokens and self._token_overlap_score(" ".join(query_tokens), search_blob) <= 0:
                continue
            for fact in entity.get("facts", [])[:3]:
                fact_text = str(fact).strip()
                if fact_text:
                    candidates.append(fact_text)
            for relation in entity.get("relations", [])[:3]:
                if not isinstance(relation, dict):
                    continue
                predicate = str(relation.get("predicate", "")).replace("_", " ").strip()
                target = str(relation.get("target", "")).strip()
                if predicate and target:
                    candidates.append(f"{entity.get('name', 'Entity')} {predicate} {target}.")
        return self._merge_unique(candidates)

    def _record_text_modifiers(self, records: list[dict[str, Any]]) -> dict[str, float]:
        modifiers: dict[str, float] = {}
        for record in records:
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            modifier = 0.0
            modifier += max(0, int(record.get("confidence", 7)) - self.settings.memory_confidence_threshold) * 0.01
            modifier += max(0, int(record.get("importance", 7)) - self.settings.memory_confidence_threshold) * 0.01
            if record.get("demoted"):
                modifier -= 0.08
            modifiers[text] = max(modifiers.get(text, float("-inf")), modifier)
        return modifiers

    def _mark_records_retrieved(self, matched_texts: list[str], records: list[dict[str, Any]]) -> None:
        text_set = {str(text).strip() for text in matched_texts if str(text).strip()}
        if not text_set:
            return
        touched = False
        for record in records:
            text = str(record.get("text", "")).strip()
            if text not in text_set:
                continue
            record["last_retrieved_at"] = self._now_iso()
            record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
            if record.get("demoted"):
                record["demoted"] = False
            touched = True
        if not touched:
            return
        payload = self._load_records_payload()
        payload["records"] = records
        payload["updated_at"] = self._now_iso()
        self._save_records_payload(payload)

    def _recent_session_files(self, limit_files: int = 4) -> list[Path]:
        files: list[Path] = []
        for session in self.session_store.list_sessions(limit=limit_files):
            file_name = str(session.get("file_name", "")).strip()
            if file_name:
                files.append(self.sessions_dir / file_name)
        return files

    def _load_session_payload(self, session_path: Path) -> dict[str, Any]:
        if not session_path.exists():
            return {}
        if session_path.suffix.lower() == ".jsonl":
            messages = []
            try:
                for line in session_path.read_text(encoding="utf-8").splitlines():
                    payload = self._safe_json(line)
                    if isinstance(payload, dict) and payload.get("role") in {"user", "assistant"}:
                        messages.append(payload)
            except OSError:
                return {}
            return {
                "session_id": session_path.stem,
                "messages": messages,
            }
        try:
            payload = self._safe_json(session_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except OSError:
            return {}
        return {}

    def _recent_session_candidates(self, max_turns: int = 24) -> list[str]:
        return self.session_store.recent_session_candidates(limit_files=4, max_turns=max_turns)

    def _session_candidates(self, query: str, limit: int) -> list[str]:
        lowered = query.lower()
        session_markers = (
            "session",
            "discuss",
            "talk about",
            "worked on",
            "working on",
            "what did we",
            "last time",
            "earlier session",
        )
        if not any(marker in lowered for marker in session_markers):
            return []

        matches = self.session_store.search_sessions(query, max(1, limit))
        candidates: list[str] = []
        for match in matches:
            display_name = str(match.get("display_name", "Untitled session")).strip()
            updated_at = str(match.get("updated_at", "")).strip()
            snippets = [
                str(snippet).strip()
                for snippet in match.get("snippets", [])
                if str(snippet).strip()
            ]
            if snippets:
                candidates.append(
                    f'Session "{display_name}" updated {updated_at}: ' + " | ".join(snippets)
                )
            else:
                candidates.append(f'Session "{display_name}" updated {updated_at}.')
        return candidates

    def _recent_log_candidates(self, max_turns: int = 24) -> list[str]:
        log_files = sorted(self.logs_dir.glob("*.jsonl"), reverse=True)
        if not log_files:
            return []

        entries: list[dict[str, Any]] = []
        for log_file in log_files:
            try:
                lines = log_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                if len(entries) >= max_turns * 2:
                    break
                payload = self._safe_json(line)
                if isinstance(payload, dict):
                    entries.append(payload)
            if len(entries) >= max_turns * 2:
                break

        entries.reverse()
        candidates: list[str] = []
        pending_user: dict[str, Any] | None = None
        for entry in entries:
            role = str(entry.get("role", "")).strip().lower()
            content = str(entry.get("content", "")).strip()
            timestamp = str(entry.get("timestamp", "")).strip()
            if not content:
                continue
            if role == "user":
                pending_user = {"content": content, "timestamp": timestamp}
                continue
            if role == "assistant" and pending_user is not None:
                if self._should_skip_recent_log_entry(content):
                    pending_user = None
                    continue
                user_text = pending_user.get("content", "")
                summary = (
                    f"Recent chat on {timestamp or pending_user.get('timestamp', '')}: "
                    f"User asked '{user_text}' and JARVIS answered '{content}'."
                ).strip()
                candidates.append(summary)
                pending_user = None

        return candidates[-max_turns:]

    def _recent_log_messages(self, limit_messages: int = 12) -> list[dict[str, str]]:
        log_files = sorted(self.logs_dir.glob("*.jsonl"), reverse=True)
        if not log_files:
            return []

        entries: list[dict[str, Any]] = []
        for log_file in log_files:
            try:
                lines = log_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                if len(entries) >= limit_messages:
                    break
                payload = self._safe_json(line)
                if not isinstance(payload, dict):
                    continue
                role = str(payload.get("role", "")).strip().lower()
                content = str(payload.get("content", "")).strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                if role == "assistant" and self._should_skip_recent_log_entry(content):
                    continue
                entries.append({"role": role, "content": content})
            if len(entries) >= limit_messages:
                break
        entries.reverse()
        return entries

    def _should_skip_recent_log_entry(self, content: str) -> bool:
        lowered = content.lower()
        return any(
            marker in lowered
            for marker in (
                "toolcall>",
                "olcall>",
                "live providers failed",
                "running in offline mode",
            )
        )

    def _is_recent_conversation_query(self, lowered_query: str) -> bool:
        markers = (
            "recently",
            "recent chat",
            "chat so far",
            "what we have done",
            "what have we done",
            "what did we talk",
            "what we talked",
            "conversation so far",
            "our chat",
            "today's chat",
            "today chat",
            "recent conversation",
            "so far",
        )
        return any(marker in lowered_query for marker in markers)

    def _rank_candidates(
        self,
        query_variants: list[str],
        candidates: list[str],
        vector_hits: set[str] | None = None,
        score_modifiers: dict[str, float] | None = None,
        allow_semantic: bool = True,
    ) -> tuple[list[str], float]:
        vector_hits = vector_hits or set()
        score_modifiers = score_modifiers or {}
        lexical_scores: list[tuple[float, str]] = []
        for candidate in candidates:
            best_token_score = max((self._token_overlap_score(variant, candidate) for variant in query_variants), default=0.0)
            best_sequence_score = max(
                (self._sequence_similarity_score(variant, candidate) for variant in query_variants),
                default=0.0,
            )
            best_phrase_score = max(
                (self._phrase_overlap_score(variant, candidate) for variant in query_variants),
                default=0.0,
            )
            lexical_score = best_token_score + (best_sequence_score * 0.2) + best_phrase_score
            lexical_score += max((self._query_focus_bonus(variant, candidate) for variant in query_variants), default=0.0)
            if candidate in vector_hits:
                lexical_score += 0.03
            lexical_score += score_modifiers.get(candidate, 0.0)
            if lexical_score > 0:
                lexical_scores.append((lexical_score, candidate))

        lexical_scores.sort(key=lambda item: item[0], reverse=True)
        if not lexical_scores:
            return [], 0.0

        # Semantic reranking is expensive. Apply it only to the strongest lexical slice
        # instead of embedding the entire candidate pool on every first recall.
        semantic_scores: dict[str, float] = {}
        if allow_semantic:
            semantic_window = [candidate for _, candidate in lexical_scores[: min(12, len(lexical_scores))]]
            semantic_scores = self._semantic_score_candidates(query_variants, semantic_window)

        scored = [
            (score + semantic_scores.get(candidate, 0.0), candidate)
            for score, candidate in lexical_scores
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return self._merge_unique([candidate for _, candidate in scored]), scored[0][0]

    def _semantic_retrieval_ready(self) -> bool:
        current_state = self._memory_source_state()
        return self._read_index_state() == current_state

    def _query_focus_bonus(self, query: str, candidate: str) -> float:
        lowered_query = query.lower()
        lowered_candidate = candidate.lower()
        bonus = 0.0

        if any(marker in lowered_query for marker in ("name", "who am i")) and any(
            marker in lowered_candidate for marker in ("name is ", "user's name", "the user name")
        ):
            bonus += 0.18

        if any(marker in lowered_query for marker in ("school", "study")) and any(
            marker in lowered_candidate for marker in ("school", "studies at", "study at")
        ):
            bonus += 0.18

        if any(marker in lowered_query for marker in ("where do i live", "where i live", "live", "location", "where am i from", "from where")):
            if any(marker in lowered_candidate for marker in ("live in", "lives in", "living in", "originally from", "from ")):
                bonus += 0.18
            elif "there" in lowered_candidate and "live" not in lowered_candidate:
                bonus -= 0.06

        if any(marker in lowered_query for marker in ("project", "working on", "build")) and any(
            marker in lowered_candidate for marker in ("working on", "building", "project", "app", "website", "assistant")
        ):
            bonus += 0.12

        return bonus

    def _semantic_score_candidates(self, query_variants: list[str], candidates: list[str]) -> dict[str, float]:
        unique_candidates = [candidate for candidate in self._merge_unique(candidates) if candidate]
        if not unique_candidates:
            return {}
        embedder = self._get_embedder()
        if embedder is None:
            return {}

        # Skip semantic scoring if we have very few candidates (lexical is enough)
        if len(unique_candidates) <= 3:
            return {}

        candidate_key = tuple(unique_candidates)
        candidate_embeddings = self._semantic_candidate_cache_embeddings
        if self._semantic_candidate_cache_key != candidate_key or candidate_embeddings is None:
            try:
                raw_embeddings = embedder.encode(unique_candidates).tolist()
            except Exception:
                return {}
            candidate_embeddings = [self._normalize_embedding(vector) for vector in raw_embeddings]
            self._semantic_candidate_cache_key = candidate_key
            self._semantic_candidate_cache_embeddings = candidate_embeddings

        # Cache query embeddings keyed by the variants tuple to avoid re-encoding on every call
        query_key = tuple(query_variants)
        if query_key not in self._query_embedding_cache:
            try:
                raw_query = embedder.encode(query_variants).tolist()
                self._query_embedding_cache[query_key] = [self._normalize_embedding(v) for v in raw_query]
                # Keep cache bounded — drop oldest entries beyond 32
                if len(self._query_embedding_cache) > 32:
                    oldest = next(iter(self._query_embedding_cache))
                    del self._query_embedding_cache[oldest]
            except Exception:
                return {}
        query_embeddings = self._query_embedding_cache[query_key]

        scores: dict[str, float] = {}
        for candidate, candidate_embedding in zip(unique_candidates, candidate_embeddings):
            best_similarity = max(
                (self._cosine_similarity(query_embedding, candidate_embedding) for query_embedding in query_embeddings),
                default=0.0,
            )
            if best_similarity > 0:
                scores[candidate] = best_similarity * 0.18
        return scores

    @staticmethod
    def _normalize_embedding(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(float(value) * float(value) for value in vector))
        if magnitude <= 0:
            return [0.0 for _ in vector]
        return [float(value) / magnitude for value in vector]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(float(a) * float(b) for a, b in zip(left, right))

    def _sequence_similarity_score(self, query: str, candidate: str) -> float:
        normalized_query = self._normalize_match_text(query)
        normalized_candidate = self._normalize_match_text(candidate)
        if not normalized_query or not normalized_candidate:
            return 0.0
        return difflib.SequenceMatcher(None, normalized_query, normalized_candidate).ratio()

    def _phrase_overlap_score(self, query: str, candidate: str) -> float:
        query_tokens = self._meaningful_query_tokens(query)
        candidate_tokens = self._meaningful_query_tokens(candidate)
        if len(query_tokens) < 2 or len(candidate_tokens) < 2:
            return 0.0
        candidate_phrases = {
            f"{first} {second}"
            for first, second in zip(candidate_tokens, candidate_tokens[1:])
        }
        phrase_hits = sum(
            1
            for first, second in zip(query_tokens, query_tokens[1:])
            if f"{first} {second}" in candidate_phrases
        )
        return min(phrase_hits * 0.12, 0.24)

    def _meaningful_query_tokens(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", self._normalize_match_text(text))
            if token and token not in self._memory_stopwords()
        ]

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).lower().replace("_", " ")).strip()

    def _normalized_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for raw in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            token = self._normalize_token(raw)
            if not token or token in self._memory_stopwords():
                continue
            tokens.add(token)
        return tokens

    def _memory_stopwords(self) -> set[str]:
        return {
            "a", "an", "and", "are", "at", "am", "about", "be", "bud", "bro", "buddy",
            "can", "could", "did", "do", "does", "for", "from", "hey", "hi", "how",
            "i", "in", "is", "it", "kind", "know", "man", "me", "my", "of",
            "on", "or", "please", "pls", "tell", "the", "to", "what", "whats",
            "which", "who", "you", "your", "also", "we", "have", "nah", "so", "far",
        }

    def _normalize_token(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("s"):
            return token[:-1]
        return token

    def _get_embedder(self) -> Any | None:
        if self._embedder_disabled:
            return None
        if self._embedder is not None:
            return self._embedder

        try:
            suppress_out = io.StringIO()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            with contextlib.redirect_stdout(suppress_out), contextlib.redirect_stderr(suppress_out):
                sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError:
            return None

        try:
            suppress_out = io.StringIO()
            with contextlib.redirect_stdout(suppress_out), contextlib.redirect_stderr(suppress_out):
                self._embedder = sentence_transformers.SentenceTransformer(
                    self.settings.memory_embedding_model
                )
        except Exception:
            self._embedder_disabled = True
            return None
        return self._embedder

    def _get_collection(self) -> Any | None:
        if self._vector_disabled:
            return None
        if self._collection is not None:
            return self._collection

        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError:
            return None

        try:
            client = chromadb.PersistentClient(path=str(self.vector_dir))
            self._collection = client.get_or_create_collection("jarvis_memories")
        except Exception:
            self._vector_disabled = True
            return None
        return self._collection

    def _safe_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _collect_memory_items_from_disk(self) -> list[dict[str, str]]:
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        items: list[dict[str, str]] = []
        for record in records:
            if not record.get("active", True):
                continue
            if int(record.get("confidence", 0)) < self.settings.memory_confidence_threshold:
                continue
            if record.get("demoted"):
                continue
            items.append(
                {
                    "tag": self._sanitize_tag(record.get("tag")),
                    "text": str(record.get("text", "")).strip(),
                    "source": str(record.get("source_ref", "")).strip() or "records",
                    "id": str(record.get("id", "")).strip(),
                }
            )

        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item["tag"], item["text"])
            if key in seen or not item["text"].strip():
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _chunk_items_from_file(self, path: Path) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            match = re.match(r"^\[(PERSONAL|PROJECT|FACT)\]\s+(.*)$", cleaned)
            if match:
                text = re.sub(r"\s+\|\s+confidence=\d+\s+importance=\d+\s*$", "", match.group(2).strip())
                items.append(
                    {
                        "tag": match.group(1),
                        "text": text,
                        "source": str(path),
                    }
                )
        return items

    def _memory_source_state(self) -> dict[str, Any]:
        # Fingerprint only the record content that actually feeds the vector index.
        # Retrieval counters/timestamps should not invalidate semantic readiness.
        payload = self._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        indexed_snapshot: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            indexed_snapshot.append(
                {
                    "id": str(record.get("id", "")).strip(),
                    "tag": self._sanitize_tag(record.get("tag")),
                    "text": str(record.get("text", "")).strip(),
                    "active": bool(record.get("active", True)),
                    "demoted": bool(record.get("demoted")),
                }
            )
        indexed_snapshot.sort(key=lambda item: (item["id"], item["tag"], item["text"]))
        digest = hashlib.sha256(
            json.dumps(indexed_snapshot, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"records_digest": digest, "count": len(indexed_snapshot)}

    def _read_index_state(self) -> dict[str, Any] | None:
        if not self.index_state_path.exists():
            return None
        try:
            return json.loads(self.index_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_index_state(self, state: dict[str, Any] | None = None) -> None:
        payload = state or self._memory_source_state()
        self.index_state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _reset_collection(self, collection: Any) -> None:
        try:
            records = collection.get(limit=100000)
            ids = records.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass
