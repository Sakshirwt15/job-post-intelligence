"""Text-to-SQL eval on 40 questions (execution accuracy).

  python -m eval.run_text2sql_eval --provider groq --model llama-3.3-70b-versatile
"""
import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from eval.metrics import results_match

HERE = Path(__file__).parent


def main():
    from app.llm import LLMClient
    from app.text2sql import ask_sql
    from eval.seed_eval_db import build_db

    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    questions = json.loads((HERE / "text2sql_questions.json").read_text())
    db = build_db(Path(tempfile.mkdtemp()) / "eval.db")
    llm = LLMClient(args.provider, args.model, log=False)

    ok, by_cat, details = 0, defaultdict(lambda: [0, 0]), []
    for q in questions:
        _, gold_rows = db.run_select(q["gold_sql"])
        res = ask_sql(db, llm, q["question"], args.prompt)
        passed = res["success"] and results_match(gold_rows, res["rows"])
        ok += passed
        by_cat[q["category"]][0] += passed
        by_cat[q["category"]][1] += 1
        details.append({"id": q["id"], "question": q["question"], "passed": bool(passed), "attempts": res["attempts"],
                        "pred_sql": res["sql"], "gold_sql": q["gold_sql"], "errors": res["errors"]})
        print(f"[{'PASS' if passed else 'FAIL'}] #{q['id']} attempts={res['attempts']} {q['question']}")
        time.sleep(args.sleep)

    summary = {"success_rate": round(ok / len(questions), 3), "passed": ok, "total": len(questions),
               "needed_retry": sum(d["attempts"] > 1 for d in details),
               "by_category": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()},
               "model": f"{llm.provider}:{llm.model}"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "text2sql_latest.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2))
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
