"""Shared Google OAuth helpers for Gmail and Calendar services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def get_google_service(
    api_name: str,
    api_version: str,
    scopes: list[str],
    client_secret_file: str,
    token_file: str,
) -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "Google API dependencies are not installed. Install google-api-python-client, "
            "google-auth-httplib2, and google-auth-oauthlib."
        ) from exc

    client_secret_path = Path(client_secret_file)
    token_path = Path(token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    if not client_secret_path.exists():
        raise RuntimeError(f"Google client secret file not found: {client_secret_path}")

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    if credentials is None or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
        try:
            credentials = flow.run_local_server(
                port=0,
                open_browser=True,
                authorization_prompt_message=(
                    "Open this URL in your browser to authorize JARVIS:\n{url}\n"
                ),
                success_message="Authorization complete. You can close this tab and return to JARVIS.",
            )
        except OSError as exc:
            raise RuntimeError(
                "Google OAuth could not start the local callback server. "
                "Close any stuck login windows and try again."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Google OAuth did not complete. Finish the browser authorization and try the command again."
            ) from exc
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    return build(api_name, api_version, credentials=credentials)
