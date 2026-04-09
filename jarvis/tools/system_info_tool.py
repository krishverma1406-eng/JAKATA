"""System health and resource inspection."""

from __future__ import annotations

from typing import Any


TOOL_DEFINITION = {
    "name": "system_info_tool",
    "description": "Inspect CPU, memory, disk, battery, and running process information.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["overview", "processes", "battery", "disk"],
                "description": "System info action to perform.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum processes to return.",
                "default": 10,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("psutil is not installed.") from exc

    action = str(params.get("action", "")).strip().lower()
    max_results = max(1, min(int(params.get("max_results", 10) or 10), 30))

    if action == "overview":
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()
        return {
            "ok": True,
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "ram_percent": memory.percent,
            "ram_used_gb": round(memory.used / (1024 ** 3), 2),
            "ram_total_gb": round(memory.total / (1024 ** 3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "battery_percent": battery.percent if battery else None,
            "battery_plugged": battery.power_plugged if battery else None,
        }
    if action == "processes":
        processes = []
        for process in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = process.info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            processes.append(info)
        processes.sort(key=lambda item: (item.get("cpu_percent", 0), item.get("memory_percent", 0)), reverse=True)
        return {"ok": True, "processes": processes[:max_results]}
    if action == "battery":
        battery = psutil.sensors_battery()
        if battery is None:
            return {"ok": True, "battery": None}
        return {
            "ok": True,
            "battery": {
                "percent": battery.percent,
                "plugged": battery.power_plugged,
                "seconds_left": battery.secsleft,
            },
        }
    if action == "disk":
        partitions = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
            except PermissionError:
                continue
            partitions.append(
                {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "used_gb": round(usage.used / (1024 ** 3), 2),
                    "total_gb": round(usage.total / (1024 ** 3), 2),
                    "percent": usage.percent,
                }
            )
        return {"ok": True, "partitions": partitions}

    return {"ok": False, "error": f"Unsupported action: {action}"}
