"""Extraction accuracy eval: prompt versions x models.

Prompts compare (ek model):
  python -m eval.run_extraction_eval --models groq:llama-3.3-70b-versatile --prompts extract_v1 extract_v2 extract_v3
Model compare (best prompt):
  python -m eval.run_extraction_eval --prompts extract_v2 \
      --models groq:llama-3.3-70b-versatile ollama:qwen2.5:3b gemini:gemini-2.5-flash
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from eval.metrics import FIELDS, aggregate, field_scores

HERE = Path(__file__).parent


def run_one(llm, prompt_version, items, sleep):
    from app.extraction import ExtractionError, extract_job

    scores, latencies, retries, failures, ptok, ctok = [], [], 0, 0, 0, 0
    for it in items:
        t0 = time.perf_counter()
        try:
            res = extract_job(llm, it["text"], prompt_version)
            pred = res.job.model_dump()
            retries += int(res.attempts > 1)
            ptok += res.prompt_tokens
            ctok += res.completion_tokens
        except ExtractionError as e:
            pred, failures = None, failures + 1
            ptok += e.usage.get("prompt_tokens", 0)
            ctok += e.usage.get("completion_tokens", 0)
        except Exception as e:  # network / auth errors
            print(f"  ! post {it['id']}: {type(e).__name__}: {e}")
            pred, failures = None, failures + 1
        latencies.append((time.perf_counter() - t0) * 1000)
        scores.append(field_scores(pred, it["labels"]))
        time.sleep(sleep)
    agg = aggregate(scores)
    return {**agg, "n": len(items), "failures": failures, "needed_retry": retries,
            "avg_latency_ms": round(statistics.mean(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
            "prompt_tokens": ptok, "completion_tokens": ctok}


def main():
    from app.llm import LLMClient, estimate_cost

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="provider:model, e.g. groq:llama-3.3-70b-versatile")
    ap.add_argument("--prompts", nargs="+", default=["extract_v1", "extract_v2", "extract_v3"])
    ap.add_argument("--dataset", default=str(HERE / "labelled_posts.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls (free-tier rate limits)")
    args = ap.parse_args()

    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    results = []
    for spec in args.models:
        provider, model = spec.split(":", 1)
        llm = LLMClient(provider, model, log=False)
        for pv in args.prompts:
            print(f"== {spec} | {pv} | {len(items)} posts")
            r = run_one(llm, pv, items, args.sleep)
            r.update(model=spec, prompt=pv,
                     est_cost_usd=round(estimate_cost(model, r["prompt_tokens"], r["completion_tokens"]), 5))
            results.append(r)
            print(f"   overall={r['overall']:.3f} exact={r['all_fields_correct']:.2f} "
                  f"latency={r['avg_latency_ms']}ms failures={r['failures']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"extraction_{ts}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    head = "| model | prompt | overall | all-correct | " + " | ".join(FIELDS) + " | latency ms | fails | retries | est $ |"
    sep = "|" + "---|" * (len(FIELDS) + 8)
    rows = [head, sep]
    for r in results:
        rows.append(f"| {r['model']} | {r['prompt']} | {r['overall']:.3f} | {r['all_fields_correct']:.2f} | "
                    + " | ".join(f"{r['per_field'][f]:.2f}" for f in FIELDS)
                    + f" | {r['avg_latency_ms']} | {r['failures']} | {r['needed_retry']} | {r['est_cost_usd']} |")
    md = "\n".join(rows)
    (HERE / "results" / "latest_extraction.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
