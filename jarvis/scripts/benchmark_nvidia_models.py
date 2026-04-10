"""Benchmark several NVIDIA hosted chat models without touching runtime routing."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import requests


DEFAULT_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "marin/marin-8b-instruct",
    "qwen/qwen3.5-122b-a10b",
]


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def benchmark_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    runs: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 32,
        "chat_template_kwargs": {"thinking": False},
    }

    attempts: list[dict[str, Any]] = []
    for run_index in range(runs):
        started = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = round(time.perf_counter() - started, 2)
            body_text = response.text
            if response.ok:
                parsed = response.json()
                content = (
                    parsed.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                attempts.append(
                    {
                        "run": run_index + 1,
                        "ok": True,
                        "status_code": response.status_code,
                        "seconds": elapsed,
                        "content": str(content).strip(),
                    }
                )
            else:
                attempts.append(
                    {
                        "run": run_index + 1,
                        "ok": False,
                        "status_code": response.status_code,
                        "seconds": elapsed,
                        "error": body_text[:400],
                    }
                )
        except requests.RequestException as exc:
            elapsed = round(time.perf_counter() - started, 2)
            attempts.append(
                {
                    "run": run_index + 1,
                    "ok": False,
                    "status_code": None,
                    "seconds": elapsed,
                    "error": str(exc),
                }
            )

    ok_attempts = [item for item in attempts if item["ok"]]
    summary: dict[str, Any] = {
        "model": model,
        "runs": attempts,
        "success_count": len(ok_attempts),
        "failure_count": len(attempts) - len(ok_attempts),
    }
    if ok_attempts:
        ok_times = [float(item["seconds"]) for item in ok_attempts]
        summary["avg_seconds"] = round(mean(ok_times), 2)
        summary["min_seconds"] = round(min(ok_times), 2)
        summary["max_seconds"] = round(max(ok_times), 2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="How many times to test each model.")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly ok",
        help="Prompt used for all model checks.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Explicit NVIDIA model IDs to benchmark.",
    )
    args = parser.parse_args()

    load_env(Path(__file__).resolve().parents[1] / "config" / ".env")

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is missing in jarvis/config/.env")

    results = []
    for model in args.models:
        results.append(
            benchmark_model(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=args.prompt,
                timeout=args.timeout,
                runs=args.runs,
            )
        )

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
