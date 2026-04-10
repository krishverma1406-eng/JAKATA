"""Persistent browser automation through Amazon Nova Act."""

from __future__ import annotations

import atexit
import contextlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nova_act import NovaAct, NovaActError

from config.settings import SCREENSHOTS_DIR, SETTINGS, Settings


class BrowserAutomationError(RuntimeError):
    """Raised when browser automation cannot complete."""


@dataclass
class BrowserSession:
    session_id: str
    nova: NovaAct
    started_at: float


class BrowserAutomationService:
    """Drive browser sessions through the Nova Act SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._lock = threading.RLock()
        self._sessions: dict[str, BrowserSession] = {}
        self._screenshot_dir = SCREENSHOTS_DIR / "browser"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._silence_nova_logging()
        atexit.register(self.close)

    def run(self, action: str, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        action_name = action.strip().lower()
        session_key = self._session_key(session_id)
        with self._lock:
            try:
                handlers = {
                    "status": self.status,
                    "open": self.open_page,
                    "fetch": self.fetch_page,
                    "click": self.click,
                    "type": self.type_into,
                    "extract": self.extract_text,
                    "screenshot": self.capture_screenshot,
                    "scroll": self.scroll,
                    "wait": self.wait_for,
                    "evaluate": self.evaluate,
                    "fill_form": self.fill_form,
                    "act": self.act,
                }
                handler = handlers.get(action_name)
                if handler is None:
                    raise BrowserAutomationError(f"Unsupported browser action: {action}")
                return handler(params, session_key)
            except BrowserAutomationError:
                raise
            except NovaActError as exc:
                raise BrowserAutomationError(str(exc)) from exc
            except Exception as exc:
                raise BrowserAutomationError(str(exc)) from exc

    def close(self) -> None:
        with self._lock:
            for browser_session in list(self._sessions.values()):
                try:
                    self._quiet_call(browser_session.nova.stop)
                except Exception:
                    pass
            self._sessions.clear()

    def status(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._sessions.get(session_id)
        if browser_session is None:
            return {"ok": True, "action": "status", "started": False, "session_id": session_id}
        return {
            "ok": True,
            "action": "status",
            "started": True,
            "nova_session_id": browser_session.nova.get_session_id(),
            **self._page_snapshot(browser_session),
        }

    def open_page(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        url = str(params.get("url", "")).strip()
        if not url:
            raise BrowserAutomationError("Missing URL.")
        browser_session = self._ensure_session(session_id, starting_page="about:blank")
        page = self._page(browser_session)
        if str(getattr(page, "url", "") or "") != url:
            self._goto(page, url, timeout_ms=self._timeout_ms(params))
        return {"ok": True, "action": "open", **self._page_snapshot(browser_session)}

    def fetch_page(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        url = str(params.get("url", "")).strip()
        if url:
            self.open_page(params, session_id)
        return self.extract_text(params, session_id)

    def click(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        timeout = self._timeout_ms(params)
        selector = str(params.get("selector", "")).strip()
        text = str(params.get("text", "")).strip()
        if selector:
            page.locator(selector).first.click(timeout=timeout)
        elif text:
            page.get_by_text(text, exact=bool(params.get("exact_text", False))).first.click(timeout=timeout)
        else:
            raise BrowserAutomationError("Provide either selector or text for click.")
        if bool(params.get("wait_for_navigation", False)):
            page.wait_for_load_state(timeout=timeout)
        return {"ok": True, "action": "click", **self._page_snapshot(browser_session)}

    def type_into(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        timeout = self._timeout_ms(params)
        selector = str(params.get("selector", "")).strip()
        text = str(params.get("text", "")).strip()
        value = str(params.get("value", ""))
        clear = bool(params.get("clear", True))
        press_enter = bool(params.get("press_enter", False))
        if selector:
            locator = page.locator(selector).first
        elif text:
            locator = page.get_by_text(text, exact=bool(params.get("exact_text", False))).first
        else:
            raise BrowserAutomationError("Provide either selector or text for type.")
        locator.click(timeout=timeout)
        if clear:
            locator.fill(value, timeout=timeout)
        else:
            locator.press_sequentially(value, timeout=timeout)
        if press_enter:
            page.keyboard.press("Enter")
        return {"ok": True, "action": "type", **self._page_snapshot(browser_session)}

    def extract_text(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        timeout = self._timeout_ms(params)
        max_chars = max(
            500,
            int(params.get("max_chars", self.settings.browser_extract_max_chars) or self.settings.browser_extract_max_chars),
        )
        selector = str(params.get("selector", "")).strip()
        text = str(params.get("text", "")).strip()
        if selector:
            content = page.locator(selector).first.inner_text(timeout=timeout)
        elif text:
            content = page.get_by_text(text, exact=bool(params.get("exact_text", False))).first.inner_text(timeout=timeout)
        else:
            content = page.locator("body").inner_text(timeout=timeout)
        cleaned = str(content or "").strip()
        return {
            "ok": True,
            "action": "extract",
            "content": cleaned[:max_chars],
            **self._page_snapshot(browser_session),
        }

    def capture_screenshot(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        from tools.screenshot_tool import analyze_image_path

        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        path = Path(str(params.get("path", "")).strip()) if str(params.get("path", "")).strip() else self._default_screenshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=bool(params.get("full_page", True)), timeout=self._timeout_ms(params))
        payload: dict[str, Any] = {"ok": True, "action": "screenshot", "path": str(path), **self._page_snapshot(browser_session)}
        if bool(params.get("analyze", False)):
            prompt = str(params.get("prompt", "")).strip() or "Describe what is visible in this browser screenshot."
            payload["analysis"] = analyze_image_path(path, prompt)
        return payload

    def scroll(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        direction = str(params.get("direction", "down")).strip().lower()
        amount = int(params.get("amount", 900) or 900)
        delta = amount if direction != "up" else -amount
        page.evaluate("(delta) => window.scrollBy(0, delta)", delta)
        return {"ok": True, "action": "scroll", "delta": delta, **self._page_snapshot(browser_session)}

    def wait_for(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        timeout = self._timeout_ms(params)
        selector = str(params.get("selector", "")).strip()
        text = str(params.get("text", "")).strip()
        state = str(params.get("state", "visible")).strip() or "visible"
        if selector:
            page.locator(selector).first.wait_for(timeout=timeout, state=state)
        elif text:
            page.get_by_text(text, exact=bool(params.get("exact_text", False))).first.wait_for(timeout=timeout, state=state)
        else:
            time.sleep(max(0, int(params.get("milliseconds", 1000) or 1000)) / 1000.0)
        return {"ok": True, "action": "wait", **self._page_snapshot(browser_session)}

    def evaluate(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        script = str(params.get("script", "")).strip()
        if not script:
            raise BrowserAutomationError("Missing JavaScript to evaluate.")
        result = page.evaluate(script, params.get("arg"))
        return {"ok": True, "action": "evaluate", "result": result, **self._page_snapshot(browser_session)}

    def fill_form(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        browser_session = self._ensure_session(session_id)
        page = self._page(browser_session)
        timeout = self._timeout_ms(params)
        fields = params.get("fields", {})
        if not isinstance(fields, dict) or not fields:
            raise BrowserAutomationError("fill_form requires a non-empty fields object.")
        filled: list[str] = []
        for key, raw_value in fields.items():
            selector = str(key).strip()
            value = raw_value
            if isinstance(raw_value, dict):
                selector = str(raw_value.get("selector", "")).strip() or selector
                value = raw_value.get("value", "")
            if not selector:
                continue
            page.locator(selector).first.fill(str(value), timeout=timeout)
            filled.append(selector)
        if not filled:
            raise BrowserAutomationError("No valid selectors were provided for fill_form.")
        return {"ok": True, "action": "fill_form", "filled": filled, **self._page_snapshot(browser_session)}

    def act(self, params: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
        prompt = str(params.get("prompt", "")).strip() or str(params.get("task", "")).strip()
        if not prompt:
            raise BrowserAutomationError("Missing prompt for Nova Act.")
        url = str(params.get("url", "")).strip()
        browser_session = self._ensure_session(session_id, starting_page="about:blank")
        if url:
            self._goto(self._page(browser_session), url, timeout_ms=self._timeout_ms(params))
        result = self._quiet_call(
            browser_session.nova.act,
            prompt,
            timeout=self._timeout_seconds(params),
            max_steps=int(params.get("max_steps", 30) or 30),
        )
        return {
            "ok": True,
            "action": "act",
            "prompt": prompt,
            "replayable": bool(getattr(result, "replayable", False)),
            "trajectory_file_path": str(getattr(result, "trajectory_file_path", "") or ""),
            **self._page_snapshot(browser_session),
        }

    def _ensure_session(self, session_id: str, starting_page: str | None = None) -> BrowserSession:
        session_key = self._session_key(session_id)
        browser_session = self._sessions.get(session_key)
        if browser_session is not None:
            return browser_session
        api_key = self.settings.nova_act_api_key.strip()
        if not api_key:
            raise BrowserAutomationError("NOVA_ACT_API_KEY is not set.")
        initial_page = starting_page or "about:blank"
        nova = NovaAct(
            starting_page=initial_page,
            headless=self.settings.nova_act_headless,
            tty=False,
            chrome_channel=self.settings.nova_act_chrome_channel.strip() or None,
            logs_directory=self._logs_directory(session_key),
            nova_act_api_key=api_key,
        )
        self._quiet_call(nova.start)
        browser_session = BrowserSession(session_id=session_key, nova=nova, started_at=time.time())
        self._sessions[session_key] = browser_session
        return browser_session

    def _page_snapshot(self, browser_session: BrowserSession) -> dict[str, Any]:
        page = self._page(browser_session)
        title = ""
        try:
            title = str(page.title() or "")
        except Exception:
            title = ""
        return {
            "session_id": browser_session.session_id,
            "nova_session_id": browser_session.nova.get_session_id(),
            "url": str(getattr(page, "url", "") or ""),
            "title": title,
        }

    def _page(self, browser_session: BrowserSession) -> Any:
        page = browser_session.nova.get_page()
        if page is None:
            raise BrowserAutomationError("Nova Act session started but no browser page is available.")
        return page

    def _logs_directory(self, session_id: str) -> str:
        configured = self.settings.nova_act_logs_dir.strip()
        root = Path(configured) if configured else (SCREENSHOTS_DIR.parent / "nova_act_logs")
        root.mkdir(parents=True, exist_ok=True)
        session_dir = root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _session_key(self, session_id: str) -> str:
        return (session_id or "default").strip() or "default"

    def _timeout_ms(self, params: dict[str, Any]) -> int:
        return max(
            1000,
            int(params.get("timeout_ms", self.settings.browser_default_timeout_ms) or self.settings.browser_default_timeout_ms),
        )

    def _timeout_seconds(self, params: dict[str, Any]) -> int:
        return max(1, int(self._timeout_ms(params) / 1000))

    def _default_screenshot_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._screenshot_dir / f"browser_{stamp}.png"

    def _goto(self, page: Any, url: str, *, timeout_ms: int) -> None:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:
            raise BrowserAutomationError(f"Failed to open {url}: {exc}") from exc

    def _quiet_call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            return fn(*args, **kwargs)

    def _silence_nova_logging(self) -> None:
        for logger_name in (
            "nova_act",
            "nova_act.trace",
            "nova_act.nova_act",
            "nova_act.types.workflow",
            "nova_act.tools.browser.default.playwright",
        ):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False


_BROWSER_SERVICE: BrowserAutomationService | None = None


def get_browser_service(settings: Settings | None = None) -> BrowserAutomationService:
    global _BROWSER_SERVICE
    if _BROWSER_SERVICE is None:
        _BROWSER_SERVICE = BrowserAutomationService(settings=settings)
    return _BROWSER_SERVICE
