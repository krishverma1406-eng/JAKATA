"""Shared Gemini-based vision analysis helpers for screenshots and frontend camera frames."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import requests

from config.settings import SETTINGS


def analyze_image_path(path: Path, prompt: str) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"Image does not exist: {path}"}

    image_bytes = path.read_bytes()
    result = analyze_image_bytes(image_bytes, "image/png", prompt)
    result["path"] = str(path)
    return result


def analyze_base64_frame(imgbase64: str, prompt: str, mime_type: str = "image/jpeg") -> dict[str, Any]:
    image_b64 = str(imgbase64 or "").strip()
    if not image_b64:
        return {"ok": False, "error": "No image data was provided."}
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        return {"ok": False, "error": "Attached image data is not valid base64."}
    return analyze_image_bytes(image_bytes, mime_type, prompt)


def analyze_image_bytes(image_bytes: bytes, mime_type: str, prompt: str) -> dict[str, Any]:
    clean_prompt = str(prompt or "").strip() or "Describe what is visible in this image."
    api_key = SETTINGS.gemini_api_key.strip()
    if not api_key:
        return {
            "ok": False,
            "error": "GEMINI_API_KEY or GOOGLE_API_KEY is not configured for vision analysis.",
            "prompt": clean_prompt,
        }

    configured_model = SETTINGS.gemini_vision_model.strip() or "gemini-flash-lite-latest"
    candidate_models: list[str] = []
    for model_name in (
        configured_model,
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-flash-latest",
    ):
        cleaned = str(model_name).strip()
        if cleaned and cleaned not in candidate_models:
            candidate_models.append(cleaned)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": clean_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    errors: list[str] = []
    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        last_error = ""
        raw: dict[str, Any] = {}
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=90,
                )
                response.raise_for_status()
                raw = response.json()
                last_error = ""
                break
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                code = exc.response.status_code if exc.response is not None else "unknown"
                last_error = f"{model}: HTTP {code}: {body}"
                if code == 503 and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break
            except requests.RequestException as exc:
                last_error = f"{model}: {exc}"
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break

        if last_error:
            errors.append(last_error)
            continue

        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text_parts = [str(part.get("text", "")).strip() for part in parts if isinstance(part, dict) and str(part.get("text", "")).strip()]
        analysis = "\n".join(text_parts).strip()
        if analysis:
            return {
                "ok": True,
                "provider": "gemini",
                "model": model,
                "analysis": analysis,
                "prompt": clean_prompt,
            }

        finish_reason = raw.get("candidates", [{}])[0].get("finishReason", "")
        errors.append(f"{model}: no text output. finishReason={finish_reason or 'unknown'}")

    return {
        "ok": False,
        "error": " | ".join(errors) if errors else "Gemini returned no usable vision response.",
        "prompt": clean_prompt,
        "model": configured_model,
    }
