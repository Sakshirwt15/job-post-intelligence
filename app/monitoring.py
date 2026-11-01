"""llm_logs table se dashboard numbers."""
from collections import defaultdict


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def _group(logs, key):
    groups = defaultdict(list)
    for r in logs:
        groups[r.get(key) or "-"].append(r)
    out = []
    for name, rows in groups.items():
        lat = [r["latency_ms"] or 0 for r in rows]
        out.append({
            key: name, "calls": len(rows),
            "error_rate": round(sum(1 for r in rows if not r["success"]) / len(rows), 3),
            "avg_latency_ms": round(sum(lat) / len(lat), 1),
            "tokens": sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows),
        })
    return sorted(out, key=lambda x: -x["calls"])


def summarize(logs: list[dict]) -> dict:
    if not logs:
        return {"total_calls": 0}
    lat = [r["latency_ms"] or 0 for r in logs]
    errors = [r for r in logs if not r["success"]]
    daily = defaultdict(lambda: {"calls": 0, "tokens": 0})
    for r in logs:
        d = (r["ts"] or "")[:10]
        daily[d]["calls"] += 1
        daily[d]["tokens"] += (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
    return {
        "total_calls": len(logs),
        "error_rate": round(len(errors) / len(logs), 3),
        "avg_latency_ms": round(sum(lat) / len(lat), 1),
        "p95_latency_ms": round(_p95(lat), 1),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in logs),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in logs),
        "by_purpose": _group(logs, "purpose"),
        "by_model": _group(logs, "model"),
        "by_prompt_version": _group(logs, "prompt_version"),
        "daily": [{"date": d, **v} for d, v in sorted(daily.items())],
        "recent_errors": [{"ts": r["ts"], "purpose": r["purpose"], "model": r["model"], "error": r["error"]}
                          for r in errors[:10]],
    }
