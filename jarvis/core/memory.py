"""Persistent logs and semantic memory helpers."""

from __future__ import annotations

import contextlib
import difflib
import io
import json
import importlib
import logging
import math
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import DATA_USER_DIR, MEMORY_LOGS_DIR, SETTINGS, USER_CHUNKS_DIR, USER_VECTOR_DIR, Settings


class Memory:
    """Read and write conversation memory plus local retrieval data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.logs_dir = MEMORY_LOGS_DIR
        self.chunks_dir = USER_CHUNKS_DIR
        self.vector_dir = USER_VECTOR_DIR
        self.profile_path = DATA_USER_DIR / "profile.md"
        self.projects_path = DATA_USER_DIR / "projects.md"
        self.index_state_path = self.vector_dir / "index_state.json"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.profile_path.touch()
        if not self.projects_path.exists():
            self.projects_path.touch()

        self._embedder = None
        self._collection = None
        self._vector_disabled = False
        self._embedder_disabled = False
        self.ensure_index_current()

    def recall(self, query: str, limit: int | None = None) -> list[str]:
        limit = limit or self.settings.memory_top_k
        lowered = query.lower()
        query_variants = self._query_variants(query)
        if any(marker in lowered for marker in ("know about me", "remember about me", "who am i", "my profile")):
            profile_items = self._candidate_pool(include_chunks=False)
            vector_hits = self._merge_unique(
                *[self._query_vector_store(variant, limit * 2) for variant in query_variants]
            )
            ranked, _ = self._rank_candidates(
                query_variants,
                self._merge_unique(profile_items, vector_hits),
                vector_hits=set(vector_hits),
            )
            return ranked[:limit]

        summary_ranked, summary_best = self._rank_candidates(
            query_variants,
            self._candidate_pool(include_chunks=False),
        )
        if summary_ranked and summary_best >= 0.18:
            return summary_ranked[:limit]

        full_candidates = self._candidate_pool(include_chunks=True)
        vector_hits = self._merge_unique(
            *[self._query_vector_store(variant, limit * 2) for variant in query_variants]
        )
        ranked, best_score = self._rank_candidates(
            query_variants,
            self._merge_unique(full_candidates, vector_hits),
            vector_hits=set(vector_hits),
        )
        if ranked and (best_score >= 0.08 or vector_hits):
            return ranked[:limit]

        fallback_hits, _ = self._fast_recall(query_variants[0], limit, include_chunks=True)
        return fallback_hits

    def persist_conversation(
        self,
        user_message: str,
        assistant_message: str,
        conversation: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        brain: Any | None = None,
        should_extract: bool = True,
        background: bool = False,
    ) -> dict[str, Any]:
        self._log_message("user", user_message, tool_trace)
        self._log_message("assistant", assistant_message, tool_trace)

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

        chunk_lines = [f"[{item['tag']}] {item['text']}" for item in extracted_items]
        chunk_path = self._append_chunk(chunk_lines)
        self._update_summary_docs(extracted_items)
        self._index_memories(extracted_items)
        self._write_index_state()
        return {"stored": len(extracted_items), "chunks": [str(chunk_path)]}

    def ensure_index_current(self) -> None:
        current_state = self._memory_source_state()
        if current_state == self._read_index_state():
            return
        items = self._collect_memory_items_from_disk()
        if not items:
            self._write_index_state(current_state)
            return
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None:
            return
        self._reset_collection(collection)
        documents = [item["text"] for item in items]
        embeddings = embedder.encode(documents).tolist()
        collection.add(
            ids=[f"memory_{uuid.uuid4().hex}" for _ in items],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "tag": item["tag"],
                    "created_at": datetime.now().isoformat(),
                    "source": item.get("source", "disk"),
                }
                for item in items
            ],
        )
        self._write_index_state(current_state)

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

    def _extract_memories(
        self,
        conversation: list[dict[str, Any]],
        brain: Any | None,
    ) -> list[dict[str, str]]:
        transcript = []
        for message in conversation:
            role = message.get("role", "unknown").upper()
            content = str(message.get("content", "")).strip()
            if content:
                transcript.append(f"{role}: {content}")

        if not transcript:
            return []

        if brain is not None:
            extraction_prompt = (
                "You extract durable memory from a conversation.\n"
                "Return strict JSON with keys PERSONAL, PROJECT, FACT.\n"
                "Each value must be a list of short factual strings worth remembering.\n"
                "Only include information that should persist beyond this conversation."
            )
            response = brain.chat(
                messages=[{"role": "user", "content": "\n".join(transcript)}],
                task_kind="memory",
                response_format="json",
                system_override=extraction_prompt,
            )
            payload = self._safe_json(response.get("content", ""))
            if isinstance(payload, dict):
                structured = self._normalize_extraction_payload(payload)
                if structured:
                    return structured

        return self._fallback_extract(transcript)

    def _normalize_extraction_payload(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for tag in ("PERSONAL", "PROJECT", "FACT"):
            values = payload.get(tag, [])
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value).strip()
                if text:
                    items.append({"tag": tag, "text": text})
        return items

    def _fallback_extract(self, transcript: list[str]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for line in transcript:
            lowered = line.lower()
            if not lowered.startswith("USER:"):
                continue
            content = line[6:].strip()
            if any(marker in lowered for marker in ("i am ", "my ", "i like ", "remember ")):
                items.append({"tag": "PERSONAL", "text": content})
            elif any(marker in lowered for marker in ("project", "build", "working on", "jarvis")):
                items.append({"tag": "PROJECT", "text": content})
        return items

    def _update_summary_docs(self, items: list[dict[str, str]]) -> None:
        personal_lines = [item["text"] for item in items if item["tag"] == "PERSONAL"]
        project_lines = [item["text"] for item in items if item["tag"] == "PROJECT"]
        if personal_lines:
            self._rewrite_summary_file(self.profile_path, "## User Profile", personal_lines)
        if project_lines:
            self._rewrite_summary_file(self.projects_path, "## Active Projects", project_lines)

    def _rewrite_summary_file(self, path: Path, title: str, new_lines: list[str]) -> None:
        existing_lines: list[str] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                if cleaned.startswith("- "):
                    existing_lines.append(cleaned[2:].strip())
                else:
                    existing_lines.append(cleaned)

        merged = []
        seen = set()
        for line in existing_lines + new_lines:
            cleaned = line.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                merged.append(cleaned)

        output = [title, ""]
        output.extend(f"- {line}" for line in merged)
        path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")

    def _index_memories(self, items: list[dict[str, str]]) -> None:
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None:
            return

        documents = [item["text"] for item in items]
        embeddings = embedder.encode(documents).tolist()
        collection.add(
            ids=[f"memory_{uuid.uuid4().hex}" for _ in items],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "tag": item["tag"],
                    "created_at": datetime.now().isoformat(),
                }
                for item in items
            ],
        )

    def _query_vector_store(self, query: str, limit: int) -> list[str]:
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None:
            return []

        results = collection.query(
            query_embeddings=embedder.encode([query]).tolist(),
            n_results=limit,
        )
        documents = results.get("documents", [])
        if not documents:
            return []
        return [doc for doc in documents[0] if isinstance(doc, str)]

    def _fallback_recall(self, query: str, limit: int) -> list[str]:
        hits, _ = self._fast_recall(query, limit, include_chunks=True)
        return hits

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

    def _token_overlap_score(self, query: str, candidate: str) -> float:
        query_tokens = self._normalized_tokens(query)
        candidate_tokens = self._normalized_tokens(candidate)
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        return overlap / math.sqrt(len(query_tokens) * len(candidate_tokens))

    def _fast_recall(self, query: str, limit: int, include_chunks: bool) -> tuple[list[str], float]:
        candidates = self._candidate_pool(include_chunks)
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
        candidates: list[str] = []
        for path in [self.profile_path, self.projects_path]:
            if path.exists():
                candidates.extend(self._summary_candidates_from_file(path))
        for chunk_file in sorted(self.chunks_dir.glob("*.txt"), reverse=True):
            candidates.extend(
                line.strip()
                for line in chunk_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("[")
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

    def _candidate_pool(self, include_chunks: bool) -> list[str]:
        candidates: list[str] = []
        for path in [self.profile_path, self.projects_path]:
            if path.exists():
                candidates.extend(self._summary_candidates_from_file(path))
        if include_chunks:
            for chunk_file in sorted(self.chunks_dir.glob("*.txt"), reverse=True):
                candidates.extend(
                    line.strip()
                    for line in chunk_file.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("[")
                )
        return candidates

    def _rank_candidates(
        self,
        query_variants: list[str],
        candidates: list[str],
        vector_hits: set[str] | None = None,
    ) -> tuple[list[str], float]:
        vector_hits = vector_hits or set()
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            best_score = max((self._token_overlap_score(variant, candidate) for variant in query_variants), default=0.0)
            if candidate in vector_hits:
                best_score += 0.03
            if best_score > 0:
                scored.append((best_score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return [], 0.0
        return self._merge_unique([candidate for _, candidate in scored]), scored[0][0]

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
            "which", "who", "you", "your", "also",
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
        items: list[dict[str, str]] = []
        items.extend(
            {"tag": "PERSONAL", "text": line, "source": str(self.profile_path)}
            for line in self._summary_candidates_from_file(self.profile_path)
        )
        items.extend(
            {"tag": "PROJECT", "text": line, "source": str(self.projects_path)}
            for line in self._summary_candidates_from_file(self.projects_path)
        )

        for chunk_file in sorted(self.chunks_dir.glob("*.txt")):
            items.extend(self._chunk_items_from_file(chunk_file))

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
                items.append(
                    {
                        "tag": match.group(1),
                        "text": match.group(2).strip(),
                        "source": str(path),
                    }
                )
        return items

    def _memory_source_state(self) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in [self.profile_path, self.projects_path, *sorted(self.chunks_dir.glob("*.txt"))]:
            if not path.exists():
                continue
            stat = path.stat()
            files.append(
                {
                    "path": str(path.resolve()),
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
            )
        return {"files": files}

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
