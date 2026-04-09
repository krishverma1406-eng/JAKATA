"""GitHub device-flow and Copilot token exchange helpers."""

from __future__ import annotations

import json
import os
import re
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config.settings import CONFIG_DIR, SETTINGS


DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_DEVICE_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_TOKEN_PATH = CONFIG_DIR / "github_device_token.json"
COPILOT_TOKEN_PATH = CONFIG_DIR / "github_copilot_token.json"


def load_copilot_session() -> dict[str, str] | None:
    direct_token = os.getenv("COPILOT_API_KEY", "").strip()
    if direct_token:
        return {
            "token": direct_token,
            "base_url": _derive_copilot_base_url(direct_token),
        }

    cached_copilot = _load_cached_copilot_token(COPILOT_TOKEN_PATH)
    if cached_copilot:
        token = str(cached_copilot.get("token", "")).strip()
        if token:
            return {
                "token": token,
                "base_url": _derive_copilot_base_url(token),
            }

    github_token = _env_first("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
    if not github_token:
        github_token = _load_cached_github_token(GITHUB_TOKEN_PATH) or ""
    if not github_token:
        return None

    payload = _exchange_github_to_copilot_token(github_token, COPILOT_TOKEN_PATH)
    token = str(payload.get("token", "")).strip()
    if not token:
        return None
    return {
        "token": token,
        "base_url": _derive_copilot_base_url(token),
    }


def login_via_device_flow(force_reauth: bool = True) -> dict[str, Any]:
    github_token = get_github_token(force_reauth=force_reauth)
    copilot_payload = _exchange_github_to_copilot_token(github_token, COPILOT_TOKEN_PATH)
    token = str(copilot_payload.get("token", "")).strip()
    return {
        "github_token": github_token,
        "copilot_token": token,
        "copilot_base_url": _derive_copilot_base_url(token),
        "expires_at": str(copilot_payload.get("expires_at", "")),
    }


def get_github_token(force_reauth: bool = False) -> str:
    if not force_reauth:
        github_token = _env_first("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
        if github_token:
            return github_token
        cached = _load_cached_github_token(GITHUB_TOKEN_PATH)
        if cached:
            return cached

    flow_payload = _start_device_flow()
    device_code = str(flow_payload.get("device_code", "")).strip()
    user_code = str(flow_payload.get("user_code", "")).strip()
    verification_uri = str(flow_payload.get("verification_uri", "https://github.com/login/device")).strip()
    interval = int(flow_payload.get("interval", 5) or 5)
    expires_in = int(flow_payload.get("expires_in", 900) or 900)

    if not device_code or not user_code:
        raise RuntimeError(f"GitHub device flow returned invalid response: {flow_payload}")

    print(f"Open {verification_uri} and enter code: {user_code}")
    print("Waiting for GitHub authorization...")
    try:
        webbrowser.open(verification_uri)
    except Exception:
        pass

    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(max(1, interval))
        token_payload = _post_form(
            GITHUB_DEVICE_TOKEN_URL,
            {
                "client_id": GITHUB_COPILOT_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        access_token = str(token_payload.get("access_token", "")).strip()
        if access_token:
            _save_cached_github_token(GITHUB_TOKEN_PATH, access_token)
            return access_token

        error = str(token_payload.get("error", "")).strip()
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "expired_token":
            break
        if error == "access_denied":
            raise RuntimeError("GitHub device login was denied.")
        raise RuntimeError(f"GitHub device flow failed: {error or token_payload}")

    raise RuntimeError("GitHub device code expired. Run login again.")


def _start_device_flow() -> dict[str, Any]:
    return _post_form(
        GITHUB_DEVICE_CODE_URL,
        {
            "client_id": GITHUB_COPILOT_CLIENT_ID,
            "scope": "read:user",
        },
    )


def _exchange_github_to_copilot_token(github_token: str, cache_path: Path) -> dict[str, Any]:
    cached = _load_cached_copilot_token(cache_path)
    if cached:
        return cached

    request = Request(
        url=COPILOT_TOKEN_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_token}",
            "User-Agent": "GitHubCopilotChat/0.26.7",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404:
            raise RuntimeError("Copilot token exchange failed: this GitHub account likely has no Copilot access.") from exc
        raise RuntimeError(f"Copilot token exchange failed: HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc

    token = str(payload.get("token", "")).strip()
    expires_at = int(payload.get("expires_at", 0) or 0)
    if not token or not expires_at:
        raise RuntimeError(f"Copilot token exchange returned invalid payload: {payload}")
    _save_cached_copilot_token(cache_path, token, expires_at)
    return {"token": token, "expires_at": expires_at}


def _derive_copilot_base_url(token: str) -> str:
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", token, flags=re.IGNORECASE)
    if not match:
        return SETTINGS.copilot_base_url.rstrip("/")
    host = re.sub(r"^https?://", "", match.group(1).strip(), flags=re.IGNORECASE)
    host = re.sub(r"^proxy\.", "api.", host, flags=re.IGNORECASE)
    if not host:
        return SETTINGS.copilot_base_url.rstrip("/")
    return f"https://{host}"


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _load_cached_github_token(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = str(payload.get("token", "")).strip()
    return token or None


def _save_cached_github_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"token": token, "updated_at": int(time.time())}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_cached_copilot_token(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = str(payload.get("token", "")).strip()
    expires_at = int(payload.get("expires_at", 0) or 0)
    if not token or not expires_at:
        return None
    if expires_at - int(time.time()) <= 300:
        return None
    return {"token": token, "expires_at": expires_at}


def _save_cached_copilot_token(path: Path, token: str, expires_at: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"token": token, "expires_at": expires_at}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url=url,
        data=urlencode(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.26.7",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc
