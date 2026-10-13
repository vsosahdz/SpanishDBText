"""Run the evaluator ablation: 62 translations x 4 configurations x N repetitions.

It uses the production code (``agents/agente_wikitext.py``) for the model call,
the retries, the JSON parsing and the programmatic completeness check. Only the
prompt changes (see ``prompts.py``). Credentials are read from the ``.env`` at the repository root.

Results are written to ``results/raw/<config>.jsonl``, one line per call. The
run can be resumed: (article, repetition) pairs that are already evaluated are skipped.

Usage:
    python run_ablation.py --dry-run                 # estimate tokens and cost, no calls
    python run_ablation.py --limit 2 --repeats 1     # smoke test
    python run_ablation.py --repeats 3               # full run
"""
import argparse
import copy
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")  # before importing the agent, which reads the environment on import

import prompts as P  # noqa: E402
A = P.A

INPUTS = HERE / "data" / "inputs.jsonl"
RAW = HERE / "results" / "raw"
SEED = 42
EVAL_MAX_TOKENS = 1024  # same as stage_3_evaluate in production
BACKOFF = [20, 40, 80, 160, 320]  # seconds between retry rounds
_lock = threading.Lock()


def load_inputs(limit: int) -> list[dict]:
    rows = [json.loads(l) for l in INPUTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        # The smoke test always includes a truncation
        trunc = [r for r in rows if r["known_truncated"]][:1]
        rest = [r for r in rows if not r["known_truncated"]][: max(limit - len(trunc), 0)]
        rows = trunc + rest
    return rows


def done_keys(config: str) -> set[tuple[int, int]]:
    path = RAW / f"{config}.jsonl"
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("llm_eval") is not None:
            keys.add((r["article_id"], r["repeat"]))
    return keys


def evaluate(config: str, row: dict, repeat: int, evaluator) -> dict:
    system = P.system_prompt(config)
    user = P.user_prompt(config, row["original"], row["translation"])
    # call_with_retries retries without waiting, so a 429 (tokens-per-minute quota)
    # exhausts it at once. We wrap it in rounds with exponential backoff.
    for wait in BACKOFF + [None]:
        resp = A.call_with_retries(system, user,
                                   f"ablation {config} art.{row['article_id']} rep.{repeat}",
                                   config=evaluator, num_predict=EVAL_MAX_TOKENS)
        if resp.content or getattr(resp, "content_filter", False) or wait is None:
            break
        time.sleep(wait)
    llm_eval = A.parse_evaluation(resp.content) if resp.content else None
    adjusted = None
    if llm_eval is not None:
        adjusted = A._adjust_evaluation_by_completeness(
            copy.deepcopy(llm_eval), row["original"], row["translation"], row["article_id"])
    return {
        "config": config,
        "article_id": row["article_id"],
        "repeat": repeat,
        "known_truncated": row["known_truncated"],
        "char_ratio": row["char_ratio"],
        "llm_eval": llm_eval,
        "adjusted_eval": adjusted,
        "raw_response": resp.content,
        "content_filter": getattr(resp, "content_filter", False),
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "cost_usd": round(A.calculate_cost(evaluator.name, resp.prompt_tokens, resp.completion_tokens), 6),
        "model": evaluator.name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


def append(result: dict) -> None:
    with _lock:
        with (RAW / f"{result['config']}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def dry_run(rows: list[dict], configs: list[str], repeats: int, evaluator_name: str) -> None:
    total = 0.0
    print(f"Estimate (≈4 characters per token; output ≈200 tokens per call), evaluator {evaluator_name}")
    for c in configs:
        pt = sum((len(P.system_prompt(c)) + len(P.user_prompt(c, r["original"], r["translation"]))) / 4
                 for r in rows) * repeats
        calls = len(rows) * repeats
        cost = A.calculate_cost(evaluator_name, int(pt), 200 * calls)
        total += cost
        print(f"  {c}: {calls} calls, ~{pt/1e6:.2f} M input tokens, ~US${cost:.2f}")
    print(f"  Estimated total: ~US${total:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=P.CONFIGS, choices=P.CONFIGS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only N articles (includes one truncation)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--evaluator", default=None,
                    help="provider:model spec; defaults to EVALUATOR_MODEL from .env")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not INPUTS.exists():
        sys.exit("data/inputs.jsonl is missing: run  python prepare_inputs.py  first")
    import os
    spec = args.evaluator or os.getenv("EVALUATOR_MODEL")
    if not spec:
        sys.exit("No evaluator: set EVALUATOR_MODEL in .env or use --evaluator")
    evaluator = A.parse_model_spec(spec)
    rows = load_inputs(args.limit)

    if args.dry_run:
        dry_run(rows, args.configs, args.repeats, evaluator.name)
        return

    if not A.verify_model(evaluator):
        sys.exit(f"Could not verify the evaluator {evaluator.name}")

    RAW.mkdir(parents=True, exist_ok=True)
    tasks = []
    for c in args.configs:
        done = done_keys(c)
        tasks += [(c, r, k) for r in rows for k in range(1, args.repeats + 1)
                  if (r["article_id"], k) not in done]
    # Fixed random order: the configurations are interleaved in time
    random.Random(SEED).shuffle(tasks)
    print(f"{len(tasks)} pending calls with {evaluator.name}, {args.workers} in parallel")

    cost, failed = 0.0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, c, r, k, evaluator) for c, r, k in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            append(res)
            cost += res["cost_usd"]
            failed += res["llm_eval"] is None
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  cumulative cost US${cost:.3f}  failed {failed}")
    print(f"Done. Cost of this run: US${cost:.3f}. Failed: {failed} (they are retried on the next run).")


if __name__ == "__main__":
    main()
