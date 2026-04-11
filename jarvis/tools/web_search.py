"""Structured web search with Tavily primary and Brave fallback."""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


TOOL_DEFINITION = {
    "name": "web_search",
    "description": "Search the web for current information and return structured citations.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to run on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
                "default": 5,
            },
            "fetch_full_page": {
                "type": "boolean",
                "description": "Fetch the full text of top results using browser_control fetch mode.",
                "default": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    query = str(params.get("query", "")).strip()
    max_results = max(1, min(int(params.get("max_results", 5) or 5), 10))
    fetch_full_page = bool(params.get("fetch_full_page", False))
    if not query:
        return {"ok": False, "error": "Missing query.", "results": []}

    provider_errors: list[str] = []
    tavily_payload = _search_tavily(query, max_results)
    if tavily_payload["ok"]:
        results = tavily_payload["results"]
        provider = "tavily"
    else:
        provider_errors.append(f"tavily: {tavily_payload['error']}")
        brave_payload = _search_brave(query, max_results)
        if brave_payload["ok"]:
            results = brave_payload["results"]
            provider = "brave"
        else:
            provider_errors.append(f"brave: {brave_payload['error']}")
            return {"ok": False, "error": " | ".join(provider_errors), "results": []}

    if fetch_full_page:
        for item in results:
            if not item.get("url"):
                continue
            full_page = _fetch_full_page(item["url"])
            if full_page:
                item["full_page_text"] = full_page

    return {
        "ok": True,
        "provider": provider,
        "query": query,
        "results": results,
    }


def _search_tavily(query: str, max_results: int) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY is not configured.", "results": []}

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
    }
    request = Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}", "results": []}
    except URLError as exc:
        return {"ok": False, "error": str(exc), "results": []}

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "published_date": item.get("published_date") or item.get("publishedTime") or "",
        }
        for item in raw.get("results", [])[:max_results]
    ]
    return {"ok": True, "results": results}


def _search_brave(query: str, max_results: int) -> dict[str, Any]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "BRAVE_SEARCH_API_KEY is not configured.", "results": []}

    request = Request(
        url=f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}&count={max_results}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}", "results": []}
    except URLError as exc:
        return {"ok": False, "error": str(exc), "results": []}

    web_results = raw.get("web", {}).get("results", [])[:max_results]
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "published_date": item.get("page_age", "") or "",
        }
        for item in web_results
    ]
    return {"ok": True, "results": results}


def _fetch_full_page(url: str) -> str:
    try:
        from tools import browser_control
    except Exception:
        return _direct_fetch(url)

    result = browser_control.execute({"action": "fetch", "url": url})
    if result.get("ok"):
        return str(result.get("content", "")).strip()
    return _direct_fetch(url)


def _direct_fetch(url: str) -> str:
    request = Request(url=url, headers={"User-Agent": "Jarvis/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    text = " ".join(body.split())
    return text[:12000]
