"""AI image generation via NVIDIA."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.settings import SCREENSHOTS_DIR, SETTINGS


TOOL_DEFINITION = {
    "name": "image_gen_tool",
    "description": (
        "Generate images from text prompts using AI. Use for requests like draw, create an image, "
        "make a picture, visualize an idea, concept art, mockups, or illustrations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate.",
            },
            "style": {
                "type": "string",
                "enum": ["realistic", "anime", "digital-art", "sketch", "cinematic"],
                "description": "Visual style. Defaults to digital-art.",
                "default": "digital-art",
            },
            "size": {
                "type": "string",
                "enum": ["square", "portrait", "landscape"],
                "description": "Image dimensions. Defaults to square.",
                "default": "square",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}

_SIZE_MAP = {
    "square": (1024, 1024),
    "portrait": (768, 1344),
    "landscape": (1344, 768),
}

_STYLE_SUFFIXES = {
    "realistic": "photorealistic, highly detailed, natural lighting, realistic textures",
    "anime": "anime style, expressive line work, vibrant colors, polished key art",
    "digital-art": "digital art, concept art, detailed illustration, clean composition",
    "sketch": "pencil sketch, hand-drawn linework, textured shading, concept sketch",
    "cinematic": "cinematic lighting, dramatic framing, atmospheric, movie still quality",
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    prompt = str(params.get("prompt", "")).strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required."}

    style = str(params.get("style", "digital-art")).strip().lower() or "digital-art"
    size = str(params.get("size", "square")).strip().lower() or "square"
    width, height = _SIZE_MAP.get(size, _SIZE_MAP["square"])
    full_prompt = _compose_prompt(prompt, style)

    api_key = SETTINGS.nvidia_api_key.strip()
    if not api_key:
        return {"ok": False, "error": "No image generation provider configured. Set NVIDIA_API_KEY."}

    return _generate_nvidia(
        prompt=prompt,
        styled_prompt=full_prompt,
        style=style,
        size=size,
        width=width,
        height=height,
        api_key=api_key,
    )


def _compose_prompt(prompt: str, style: str) -> str:
    suffix = _STYLE_SUFFIXES.get(style, _STYLE_SUFFIXES["digital-art"])
    return (
        f"{prompt}, {suffix}, high quality, coherent composition, no text, no watermark, "
        "clean details, visually striking"
    )


def _generate_nvidia(
    *,
    prompt: str,
    styled_prompt: str,
    style: str,
    size: str,
    width: int,
    height: int,
    api_key: str,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": styled_prompt,
            "height": height,
            "width": width,
            "cfg_scale": 0,
            "mode": "base",
            "samples": 1,
            "seed": 0,
            "steps": 4,
        }
    ).encode("utf-8")

    request = Request(
        url="https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            body = response.read()
            content_type = str(response.headers.get("Content-Type", "")).lower()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "error": f"HTTP {exc.code}: {body[:240]}"}
    except URLError as exc:
        return {"ok": False, "error": f"Network error: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if content_type.startswith("image/"):
        saved = _save_image_bytes(body, prompt, content_type=content_type)
        return _success_payload(prompt, style, size, saved, provider="nvidia")

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return {"ok": False, "error": "NVIDIA returned an unexpected non-JSON response."}

    image_bytes = _extract_image_bytes(data)
    if image_bytes is not None:
        saved = _save_image_bytes(image_bytes, prompt)
        return _success_payload(prompt, style, size, saved, provider="nvidia")

    image_url = _extract_image_url(data)
    if image_url:
        downloaded = _download_image(image_url, prompt)
        if downloaded is not None:
            payload = _success_payload(prompt, style, size, downloaded, provider="nvidia")
            payload["source_url"] = image_url
            return payload
        return {
            "ok": True,
            "url": image_url,
            "prompt": prompt,
            "style": style,
            "size": size,
            "provider": "nvidia",
            "model": "black-forest-labs/flux.1-schnell",
        }

    return {"ok": False, "error": "NVIDIA returned no image bytes or image URL."}


def _extract_image_bytes(payload: Any) -> bytes | None:
    if not isinstance(payload, dict):
        return None

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        base64_value = str((artifacts[0] or {}).get("base64", "")).strip()
        if base64_value:
            return base64.b64decode(base64_value)

    data_items = payload.get("data")
    if isinstance(data_items, list) and data_items:
        b64_json = str((data_items[0] or {}).get("b64_json", "")).strip()
        if b64_json:
            return base64.b64decode(b64_json)
    return None


def _extract_image_url(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    data_items = payload.get("data")
    if isinstance(data_items, list) and data_items:
        return str((data_items[0] or {}).get("url", "")).strip()
    return ""


def _download_image(url: str, prompt: str) -> Path | None:
    try:
        with urlopen(Request(url=url, headers={"User-Agent": "JARVIS/1.0"}), timeout=60) as response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            image_bytes = response.read()
    except Exception:
        return None
    return _save_image_bytes(image_bytes, prompt, content_type=content_type)


def _save_image_bytes(image_bytes: bytes, prompt: str, content_type: str = "image/png") -> Path:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_for_content_type(content_type)
    timestamp = int(time.time())
    prompt_slug = _slugify(prompt)[:48] or "image"
    save_path = SCREENSHOTS_DIR / f"generated_{timestamp}_{prompt_slug}{suffix}"
    save_path.write_bytes(image_bytes)
    return save_path


def _success_payload(
    prompt: str,
    style: str,
    size: str,
    path: Path,
    *,
    provider: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "path": str(path),
        "url": f"/generated/{path.name}",
        "prompt": prompt,
        "style": style,
        "size": size,
        "provider": provider,
        "model": "black-forest-labs/flux.1-schnell",
    }


def _suffix_for_content_type(content_type: str) -> str:
    lowered = str(content_type or "").lower()
    if "jpeg" in lowered or "jpg" in lowered:
        return ".jpg"
    if "webp" in lowered:
        return ".webp"
    return ".png"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "image"
