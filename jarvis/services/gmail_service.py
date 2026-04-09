"""Gmail integration helpers."""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any

from config.settings import SETTINGS
from services.google_oauth import get_google_service

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


class GmailService:
    def __init__(self) -> None:
        self._service: Any | None = None

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = get_google_service(
                "gmail",
                "v1",
                GMAIL_SCOPES,
                SETTINGS.gmail_client_secret_file,
                SETTINGS.gmail_token_file,
            )
        return self._service

    def unread(self, max_results: int = 10) -> list[dict[str, Any]]:
        return self.search("is:unread", max_results=max_results)

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        response = self.service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        messages = response.get("messages", [])
        results: list[dict[str, Any]] = []
        for item in messages:
            message = self.service.users().messages().get(userId="me", id=item["id"], format="metadata").execute()
            headers = {
                header.get("name", "").lower(): header.get("value", "")
                for header in message.get("payload", {}).get("headers", [])
            }
            results.append(
                {
                    "id": item["id"],
                    "thread_id": message.get("threadId"),
                    "subject": headers.get("subject", ""),
                    "from": headers.get("from", ""),
                    "date": headers.get("date", ""),
                    "snippet": message.get("snippet", ""),
                }
            )
        return results

    def send(self, to: str, subject: str, body: str) -> dict[str, Any]:
        email = EmailMessage()
        email["To"] = to
        email["Subject"] = subject
        email.set_content(body)
        encoded = base64.urlsafe_b64encode(email.as_bytes()).decode("utf-8")
        payload = {"raw": encoded}
        sent = self.service.users().messages().send(userId="me", body=payload).execute()
        return {"id": sent.get("id"), "thread_id": sent.get("threadId")}


_GMAIL_SERVICE: GmailService | None = None


def get_gmail_service() -> GmailService:
    global _GMAIL_SERVICE
    if _GMAIL_SERVICE is None:
        _GMAIL_SERVICE = GmailService()
    return _GMAIL_SERVICE
