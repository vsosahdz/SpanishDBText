"""Summarise the ablation: write results/summary.md and results/summary.json.

For each configuration it reports the score distribution (on the complete
translations, where inflation is measured), what the evaluator did with the three known
truncations, and the paired difference against production (G1L1) with a bootstrap CI.
The repetitions of the same article are averaged before comparing.

Usage:
    python analyze.py
"""
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import prompts as P

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT_MD = HERE / "results" / "summary.md"
OUT_JSON = HERE / "results" / "summary.json"

# English labels for summary.md. summary.json keeps the Spanish descriptions of
# prompts.DESCRIPTION, which are part of the published results.
DESCRIPTION_EN = {
    "G1L1": "production: guide + length evidence",
    "G0L1": "no guide, with length evidence",
    "G1L0": "with guide, no length evidence",
    "G0L0": "no guide and no length evidence",
}
TRUNC_CAP = 70  # the guide and the completeness block set this cap for a truncation
BOOT = 10_000


def load() -> dict[str, list[dict]]:
    data = {}
    for c in P.CONFIGS:
        path = RAW / f"{c}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            # One row per (article, repetition): the successful one if there is one. Earlier
            # failed attempts (e.g. from a 429 quota error) are dropped.
            best = {}
            for r in rows:
                k = (r["article_id"], r["repeat"])
                if k not in best or (best[k].get("llm_eval") is None and r.get("llm_eval") is not None):
                    best[k] = r
            data[c] = list(best.values())
            data[c + "_cost"] = round(sum(r.get("cost_usd", 0) for r in rows), 4)
    return data


def dist(values: list[float]) -> dict:
    if not values:
        return {}
    mode, mode_n = Counter(values).most_common(1)[0]
    return {
        "n": len(values),
        "mean": round(st.mean(values), 2),
        "median": st.median(values),
        "sd": round(st.pstdev(values), 2),
        "min": min(values),
        "max": max(values),
        "pct_ge95": round(100 * sum(v >= 95 for v in values) / len(values), 1),
        "pct_ge90": round(100 * sum(v >= 90 for v in values) / len(values), 1),
        "mode": mode,
        "pct_mode": round(100 * mode_n / len(values), 1),
        "distinct": len(set(values)),
    }


def mentions_truncation(ev: dict) -> bool:
    errs = ev.get("errors") or []
    text = " ".join(map(str, errs)).lower()
    return any(k in text for k in ("trunc", "omis", "omit", "incomplet", "falta", "missing"))


def per_article_mean(rows: list[dict], key: str, field: str) -> dict[int, float]:
    acc = defaultdict(list)
    for r in rows:
        ev = r.get(key)
        if ev is not None and isinstance(ev.get(field), (int, float)):
            acc[r["article_id"]].append(ev[field])
    return {a: st.mean(v) for a, v in acc.items()}


def paired_ci(a: dict[int, float], b: dict[int, float]) -> dict | None:
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(42)
    boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOT))
    return {"n": len(ids), "mean_diff": round(st.mean(diffs), 2),
            "ci95": [round(boots[int(0.025 * BOOT)], 2), round(boots[int(0.975 * BOOT)], 2)]}


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("No results in results/raw/. Run run_ablation.py first")
    summary = {}
    for c, rows in [(k, v) for k, v in data.items() if not k.endswith("_cost")]:
        ok = [r for r in rows if r.get("llm_eval") is not None]
        full = [r for r in ok if not r["known_truncated"]]
        trunc = [r for r in ok if r["known_truncated"]]
        s = {
            "description": P.DESCRIPTION[c],
            "keys": len(rows),
            "parse_failures": len(rows) - len(ok),
            "cost_usd": data[c + "_cost"],
            "overall_complete": dist([r["llm_eval"]["overall_score"] for r in full]),
            "completeness_complete": dist([r["llm_eval"]["completeness"] for r in full]),
            "truncated": {},
        }
        by_id = defaultdict(list)
        for r in trunc:
            by_id[r["article_id"]].append(r)
        for aid, rs in sorted(by_id.items()):
            s["truncated"][aid] = {
                "char_ratio": rs[0]["char_ratio"],
                "llm_completeness": [r["llm_eval"]["completeness"] for r in rs],
                "llm_overall": [r["llm_eval"]["overall_score"] for r in rs],
                "adjusted_overall": [r["adjusted_eval"]["overall_score"] for r in rs],
                "detected_by_llm": sum(r["llm_eval"]["completeness"] <= TRUNC_CAP for r in rs),
                "mentioned_in_errors": sum(mentions_truncation(r["llm_eval"]) for r in rs),
                "repeats": len(rs),
            }
        summary[c] = s

    if "G1L1" in data:
        base = per_article_mean([r for r in data["G1L1"] if not r["known_truncated"]], "llm_eval", "overall_score")
        for c in [k for k in data if not k.endswith("_cost")]:
            if c != "G1L1":
                other = per_article_mean([r for r in data[c] if not r["known_truncated"]], "llm_eval", "overall_score")
                summary[c]["paired_vs_G1L1_overall"] = paired_ci(other, base)

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Evaluator ablation — summary", "",
             "LLM scores before the programmatic check. Distribution on the complete",
             "translations (without the truncations 7, 24 and 41).", "",
             "| Config | Description | n | Mean | Median | SD | % ≥95 | % ≥90 | Mode (%) | Distinct values | Cost US$ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in P.CONFIGS:
        if c not in summary:
            continue
        d, s = summary[c]["overall_complete"], summary[c]
        if d:
            lines.append(f"| {c} | {DESCRIPTION_EN[c]} | {d['n']} | {d['mean']} | {d['median']} | {d['sd']} | "
                         f"{d['pct_ge95']} | {d['pct_ge90']} | {d['mode']} ({d['pct_mode']}) | {d['distinct']} | {s['cost_usd']} |")
    lines += ["", "## Known truncations", "",
              f"*Detected* = LLM completeness ≤ {TRUNC_CAP}. *Mentioned* = the truncation or omission appears in the list of errors.", "",
              "| Config | ID | Ratio | LLM completeness | LLM overall | Adjusted overall | Detected | Mentioned |",
              "|---|---|---|---|---|---|---|---|"]
    for c in P.CONFIGS:
        for aid, t in summary.get(c, {}).get("truncated", {}).items():
            lines.append(f"| {c} | {aid} | {t['char_ratio']} | {t['llm_completeness']} | {t['llm_overall']} | "
                         f"{t['adjusted_overall']} | {t['detected_by_llm']}/{t['repeats']} | {t['mentioned_in_errors']}/{t['repeats']} |")
    lines += ["", "## Paired difference against production (overall, complete translations)", "",
              "| Config | Articles | Mean difference (config − G1L1) | 95% bootstrap CI |", "|---|---|---|---|"]
    for c in P.CONFIGS:
        p = summary.get(c, {}).get("paired_vs_G1L1_overall")
        if p:
            lines.append(f"| {c} | {p['n']} | {p['mean_diff']} | [{p['ci95'][0]}, {p['ci95'][1]}] |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
