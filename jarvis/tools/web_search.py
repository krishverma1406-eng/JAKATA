"""Structured web search with Tavily primary and Brave fallback."""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlparse
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
    candidate_limit = max_results if max_results >= 10 else min(max_results * 2, 10)
    fetch_full_page = bool(params.get("fetch_full_page", False))
    if not query:
        return {"ok": False, "error": "Missing query.", "results": []}

    provider_errors: list[str] = []
    tavily_payload = _search_tavily(query, candidate_limit)
    if tavily_payload["ok"]:
        results = tavily_payload["results"]
        provider = "tavily"
    else:
        provider_errors.append(f"tavily: {tavily_payload['error']}")
        brave_payload = _search_brave(query, candidate_limit)
        if brave_payload["ok"]:
            results = brave_payload["results"]
            provider = "brave"
        else:
            provider_errors.append(f"brave: {brave_payload['error']}")
            return {"ok": False, "error": " | ".join(provider_errors), "results": []}

    results = _post_process_results(results, query)[:max_results]

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


def _post_process_results(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Score, deduplicate, and flag likely-answering search results."""
    query_tokens = _query_tokens(query)
    query_lower = query.strip().lower()
    seen_domains: set[str] = set()
    processed: list[dict[str, Any]] = []

    for result in results:
        item = dict(result)
        url = str(item.get("url", "")).strip()
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = url[:50].lower()
        domain = domain[4:] if domain.startswith("www.") else domain
        dedupe_key = domain or url[:50].lower()
        if dedupe_key and dedupe_key in seen_domains:
            continue
        if dedupe_key:
            seen_domains.add(dedupe_key)

        title = str(item.get("title", "")).strip().lower()
        snippet = str(item.get("snippet", "") or item.get("content", "")).strip().lower()
        haystack = f"{title} {snippet}".strip()
        hit_count = sum(1 for token in query_tokens if token in haystack)
        relevance = hit_count / max(len(query_tokens), 1)
        if query_lower and query_lower in haystack:
            relevance = max(relevance, 0.95)

        item["relevance_score"] = round(min(relevance, 1.0), 2)
        item["answers_query"] = bool(
            (query_lower and query_lower in haystack)
            or (query_tokens and hit_count >= max(1, len(query_tokens) // 2))
        )
        processed.append(item)

    processed.sort(
        key=lambda item: (
            int(bool(item.get("answers_query"))),
            float(item.get("relevance_score", 0.0)),
            len(str(item.get("title", ""))),
        ),
        reverse=True,
    )
    return processed


def _query_tokens(query: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 1]
    meaningful = [token for token in tokens if token not in {"the", "and", "for", "with", "from", "what", "when"}]
    return meaningful or tokens


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
