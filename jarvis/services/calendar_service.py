"""Google Calendar integration helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from config.settings import SETTINGS
from services.google_oauth import get_google_service

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarService:
    def __init__(self) -> None:
        self._service: Any | None = None

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = get_google_service(
                "calendar",
                "v3",
                CALENDAR_SCOPES,
                SETTINGS.calendar_client_secret_file,
                SETTINGS.calendar_token_file,
            )
        return self._service

    def today_events(self, calendar_id: str = "primary", max_results: int = 10) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)
        response = self.service.events().list(
            calendarId=calendar_id,
            timeMin=now.isoformat(),
            timeMax=end_of_day.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = []
        for item in response.get("items", []):
            events.append(
                {
                    "id": item.get("id"),
                    "summary": item.get("summary", ""),
                    "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                    "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    "location": item.get("location", ""),
                }
            )
        return events

    def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        event = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": _iso_with_timezone(start_time)},
            "end": {"dateTime": _iso_with_timezone(end_time)},
        }
        created = self.service.events().insert(calendarId=calendar_id, body=event).execute()
        return {
            "id": created.get("id"),
            "summary": created.get("summary", ""),
            "html_link": created.get("htmlLink", ""),
        }


def _iso_with_timezone(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


_CALENDAR_SERVICE: CalendarService | None = None


def get_calendar_service() -> CalendarService:
    global _CALENDAR_SERVICE
    if _CALENDAR_SERVICE is None:
        _CALENDAR_SERVICE = CalendarService()
    return _CALENDAR_SERVICE
