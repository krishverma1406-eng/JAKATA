"""One-time cleanup for JARVIS memory records and derived artifacts."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DATA_USER_DIR, USER_MEMORY_RECORDS_FILE, USER_VECTOR_DIR


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "with",
}

USER_MARKERS = (
    "the user",
    "user ",
    "user's",
    "my ",
    "i am ",
    "i'm ",
    "name is ",
    "github",
    "email",
    "school",
    "live",
    "stud",
    "prefer",
    "like ",
    "want ",
    "working on",
    "building",
    "developing",
    "project",
)

BLOCKED_CONTEXT_MARKERS = (
    "assistant can ",
    "jarvis can ",
    "asked to open",
    "open youtube",
    "search for ",
    "searches for ",
    "calendar",
    "weather",
    "current time",
    "cpu",
    "ram",
    "disk",
    "battery",
    "session id",
    "current session",
    "message id",
    "bookmarked",
    "notifications",
    "unread notifications",
    "intend to call",
    "assistant's name",
    "refers to assistant",
    "email was sent",
    "temperature",
    "humidity",
    "wind speed",
    "samay raina",
    "comicstaan",
    "india's got latent",
)


def normalize_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sanitize_tag(value: Any) -> str:
    tag = str(value or "FACT").strip().upper()
    return tag if tag in {"PERSONAL", "PROJECT", "FACT"} else "FACT"


def source_rank(record: dict[str, Any]) -> tuple[int, str]:
    source_type = str(record.get("source_type", "")).strip().lower()
    updated_at = str(record.get("updated_at", "") or record.get("created_at", "") or "")
    priority = 0
    if source_type in {"canonicalized", "explicit"}:
        priority = 3
    elif source_type.startswith("conversation_extract"):
        priority = 2
    elif source_type == "manual_file":
        priority = 1
    return (priority, updated_at)


def canonical_tag(record: dict[str, Any]) -> str:
    text = str(record.get("text", "")).strip().lower()
    if any(
        marker in text
        for marker in ("working on", "building", "developing", "project", "app", "website", "mvp", "vercel", "founder", "team")
    ):
        return "PROJECT"
    if any(
        marker in text
        for marker in ("name is ", "email", "@", "github", "school", "stud", "live", "class ", "grade ", "prefer", "like ", "hate ")
    ):
        return "PERSONAL"
    return sanitize_tag(record.get("tag"))


def topic_key(tag: str, text: str) -> str | None:
    lowered = text.lower()
    if tag == "PERSONAL":
        if "@" in lowered or "email" in lowered:
            return "email"
        if "github username" in lowered or "github" in lowered:
            return "github"
        if any(marker in lowered for marker in ("school", "studies at", "study at")):
            return "school"
        if any(marker in lowered for marker in ("my name", "name is ", "user's name", "the user name")):
            return "name"
        if any(marker in lowered for marker in ("live in", "lives in", "living in", "originally from")):
            return "location"
        if any(marker in lowered for marker in ("class ", "grade ", "10th", "11th", "12th")):
            return "class_grade"
    return None


def is_candidate_record(record: dict[str, Any]) -> bool:
    text = str(record.get("text", "")).strip()
    if not text:
        return False

    tag = canonical_tag(record)
    explicit = bool(record.get("explicit"))
    lowered = text.lower()

    if any(marker in lowered for marker in BLOCKED_CONTEXT_MARKERS):
        return False

    has_user_marker = any(marker in lowered for marker in USER_MARKERS)
    looks_like_email = "@" in lowered and "." in lowered
    looks_like_username = lowered.startswith("github username")
    durable_tokens = normalize_tokens(text)
    if len(durable_tokens) <= 1 and not looks_like_email and not looks_like_username:
        return False

    if tag in {"PERSONAL", "PROJECT"}:
        return has_user_marker or looks_like_email or looks_like_username

    if tag == "FACT":
        return explicit and (
            has_user_marker
            or any(
                marker in lowered
                for marker in (
                    "hate ",
                    "like ",
                    "love ",
                    "prefer ",
                    "always ",
                    "never ",
                )
            )
        )

    return False


def build_summary(records: list[dict[str, Any]], tag: str, title: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not record.get("active", True) or record.get("demoted"):
            continue
        if sanitize_tag(record.get("tag")) != tag:
            continue
        text = str(record.get("text", "")).strip()
        lowered = text.lower()
        if not text or lowered in seen:
            continue
        seen.add(lowered)
        lines.append(f"- {text}")
    body = [title, ""]
    body.extend(lines)
    return "\n".join(body).rstrip() + "\n"


def archive_chunks() -> None:
    chunks_dir = DATA_USER_DIR / "chunks"
    archive_dir = DATA_USER_DIR / "chunks_archive"
    archive_dir.mkdir(exist_ok=True)

    for chunk in sorted(chunks_dir.glob("*.txt")):
        target = archive_dir / chunk.name
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            target = archive_dir / f"{chunk.stem}_{stamp}{chunk.suffix}"
        shutil.move(str(chunk), str(target))
        print(f"Archived chunk: {chunk.name}")


def clear_vector_index() -> None:
    USER_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    for path in USER_VECTOR_DIR.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except PermissionError:
            print(f"Skipped locked vector file: {path.name}")
    print(f"Cleared vector index: {USER_VECTOR_DIR}")


def canonicalize() -> None:
    records_path = Path(USER_MEMORY_RECORDS_FILE)
    backup_path = records_path.with_suffix(".json.bak")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload.get("records", [])
    active_now = [record for record in records if record.get("active", True) and not record.get("demoted")]
    if backup_path.exists() and len(active_now) < 20:
        payload = json.loads(backup_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        print(f"Using backup source because current active set is only {len(active_now)} records")

    print(f"Loaded {len(records)} total records")

    candidates = [record for record in records if is_candidate_record(record)]
    print(f"Candidate records: {len(candidates)}")

    by_tag: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        by_tag.setdefault(canonical_tag(record), []).append(record)

    canonical_ids: set[str] = set()
    for tag, tag_records in by_tag.items():
        tag_records.sort(key=source_rank, reverse=True)
        kept: list[dict[str, Any]] = []
        for record in tag_records:
            text = str(record.get("text", "")).strip()
            tokens = normalize_tokens(text)
            current_topic = topic_key(tag, text)
            conflict = False
            for existing in kept:
                existing_tokens = normalize_tokens(str(existing.get("text", "")).strip())
                existing_topic = topic_key(tag, str(existing.get("text", "")).strip())
                shorter = min(len(tokens), len(existing_tokens))
                if (
                    (current_topic and existing_topic and current_topic == existing_topic)
                    or (shorter > 0 and (tokens <= existing_tokens or existing_tokens <= tokens))
                    or jaccard(tokens, existing_tokens) > 0.50
                ):
                    conflict = True
                    print(
                        f"  DEDUP [{tag}]: '{text[:60]}' ~ "
                        f"'{str(existing.get('text', '')).strip()[:60]}'"
                    )
                    break
            if not conflict:
                kept.append(record)
                canonical_ids.add(str(record.get("id", "")).strip())
        print(f"  {tag}: {len(tag_records)} -> {len(kept)} records")

    if not canonical_ids:
        raise RuntimeError("No canonical records were selected. Aborting cleanup.")

    if not backup_path.exists():
        backup_path.write_text(records_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup saved to {backup_path}")

    now_iso = datetime.now().astimezone().isoformat()
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if record_id in canonical_ids:
            record["tag"] = canonical_tag(record)
            record["active"] = True
            record["demoted"] = False
            record["explicit"] = True
            record["source_type"] = "canonicalized"
            record["source_ref"] = ""
            record["superseded_by"] = None
            record["updated_at"] = now_iso
        else:
            record["active"] = False
            record["demoted"] = True

    payload["records"] = records
    payload["updated_at"] = now_iso
    records_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved canonical records to {records_path}")

    profile_path = DATA_USER_DIR / "profile.md"
    projects_path = DATA_USER_DIR / "projects.md"
    entities_path = DATA_USER_DIR / "entities.json"
    summary_path = DATA_USER_DIR / "memory_daily_summary.json"

    profile_path.write_text(build_summary(records, "PERSONAL", "## User Profile"), encoding="utf-8")
    projects_path.write_text(build_summary(records, "PROJECT", "## Active Projects"), encoding="utf-8")
    entities_path.write_text(
        json.dumps({"version": 1, "updated_at": now_iso, "entities": {}}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps({}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print("Rewrote derived summary files.")

    archive_chunks()
    clear_vector_index()
    print("Done. Start JARVIS again so it rebuilds entities and the vector index from clean records.")


if __name__ == "__main__":
    canonicalize()
