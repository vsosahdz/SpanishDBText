# -*- coding: utf-8 -*-
"""Ablation of Stage 4: the consensus, compared against its own candidates.

QUESTION
    The article presents consensus fusion as its main contribution, and this
    stage uses 77% of the budget, but it is never compared against anything. A
    reviewer will ask: "better, compared with what?".

DESIGN
    The three candidates are already saved in the pipeline JSON files, so there
    is NO need to translate anything again. Each candidate is only evaluated
    with the SAME evaluator as Stage 5: same model, same prompt, same scale,
    same programmatic completeness adjustment. This is why this script IMPORTS
    the agent's functions instead of writing them again; any reimplementation
    would make the comparison invalid.

TWO BASELINES, NOT ONE
    (a) EACH FIXED MODEL: "if you had simply used Mistral Large for
        everything". This is the realistic baseline, the one someone could
        adopt.
    (b) BEST CANDIDATE PER ENTRY: an ORACLE. In a real pipeline you do not
        know in advance which model will be the best for each entry, so this
        is an upper bound, not an available option. If the consensus loses
        to the oracle but beats every fixed model, the correct reading is
        that the consensus is as good as a perfect selector.

    Reporting only (b) would be unfair to the method; reporting only (a)
    would be too lenient. This is why both are reported.

POSSIBLE UNCOMFORTABLE RESULT
    If (b) matches the consensus, the conclusion would be that SELECTING is
    enough and FUSING is not needed, which would remove the most expensive
    stage of the pipeline. That is a valuable finding, and it is better to
    know it before submission than during review.

RECOVERY FROM FAILURES
    The process makes thousands of API calls, so one line is saved per
    FINISHED entry in metrics/output/ablation_progress.jsonl, with an
    immediate flush. If the script dies -- network outage, quota limit,
    Ctrl-C, reboot -- just run THE SAME COMMAND again: entries that were
    already evaluated are skipped and are not paid for again. Tokens spent in
    earlier runs are added up, so the cost reported at the end is the real
    total and not only the cost of the last batch.
    An entry where EVERYTHING failed is not marked as done, so the next
    attempt retries it. With --restart the run starts from scratch.

USAGE
    python metrics/ablation_consensus.py --sample 50 --dry-run
    python metrics/ablation_consensus.py --sample 20          # calibrate speed
    python metrics/ablation_consensus.py --sample 200         # full run
    python metrics/ablation_consensus.py --sample 200         # resumes if it was cut off
    python metrics/ablation_consensus.py --datasets fever --sample 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

CORPORA = {
    "wikitext": (BASE / "output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json",
                 BASE / "output/wikitext/wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json"),
    "fever": (BASE / "output/FEVER/fever_selected_260712_212751.json",
              BASE / "output/FEVER/fever_selected_pipeline_260712_212751.json"),
    "europarl": (BASE / "output/Europarl/selected_260710_084528.json",
                 BASE / "output/Europarl/selected_pipeline_260710_084528.json"),
}
SEED = 42
OUTDIR = BASE / "metrics" / "output"

# GPT-4.1 Mini prices on Azure, USD per million tokens
PRECIO_IN, PRECIO_OUT = 0.40, 1.60


def cargar(corp_path: Path, pipe_path: Path):
    corpus = {e["id"]: e for e in
              json.loads(corp_path.read_text(encoding="utf-8"))["payload"][0]["data"]}
    pipe = json.loads(pipe_path.read_text(encoding="utf-8"))
    pipe = pipe["pipeline"] if isinstance(pipe, dict) and "pipeline" in pipe else pipe
    return corpus, pipe


def id_de(p):
    return p.get("article_id", p.get("id"))


def pares(corpus, pipe):
    """Entries with 3 candidates and an available consensus."""
    out = []
    for p in pipe:
        pid = id_de(p)
        ent = corpus.get(pid)
        cands = [c for c in (p.get("candidates") or []) if (c.get("text") or "").strip()]
        if not ent or len(cands) < 2:
            continue
        original = (ent.get("english") or {}).get("text", "") or ""
        consenso = (ent.get("spanish") or {}).get("text", "") or ""
        if not original or not consenso:
            continue
        out.append({"id": pid, "original": original, "consenso": consenso,
                    "candidatos": [{"model": c.get("model"), "text": c["text"]} for c in cands]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CORPORA), choices=list(CORPORA))
    ap.add_argument("--sample", type=int, default=200, help="entries per dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="does not call the model: only estimates cost and number of calls")
    ap.add_argument("--reuse-consensus-eval", action="store_true",
                    help="reuses the consensus evaluation already saved in the pipeline JSON "
                         "instead of computing it again (saves 1 call in 4, but mixes "
                         "evaluations from different dates)")
    ap.add_argument("--progress", default=None,
                    help="JSONL progress file (default "
                         "metrics/output/ablation_progress.jsonl). One line is written "
                         "per finished entry; if the process is cut off, the next "
                         "run resumes where it stopped.")
    ap.add_argument("--restart", action="store_true",
                    help="ignores previous progress and starts from scratch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(SEED)
    trabajo = {}
    for ds in args.datasets:
        cp, pp = CORPORA[ds]
        if not cp.exists() or not pp.exists():
            print("  !! files missing for %s, skipped" % ds, file=sys.stderr)
            continue
        corpus, pipe = cargar(cp, pp)
        items = pares(corpus, pipe)
        if args.sample and len(items) > args.sample:
            items = rng.sample(items, args.sample)
        trabajo[ds] = items

    # ── progress: loaded BEFORE reporting, so that the count and
    #    --dry-run show what is really left to do ──
    prog_path = Path(args.progress) if args.progress else OUTDIR / "ablation_progress.jsonl"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.restart and prog_path.exists():
        prog_path.rename(prog_path.with_suffix(".jsonl.bak"))
        print("previous progress moved to %s\n" % prog_path.with_suffix(".jsonl.bak").name)

    resultados, uso_total = {ds: [] for ds in trabajo}, {"in": 0, "out": 0}
    hechas = set()
    if prog_path.exists():
        for linea in prog_path.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea:
                continue
            try:
                r = json.loads(linea)
            except json.JSONDecodeError:
                continue          # line truncated by a cut-off: discarded
            ds = r.get("ds")
            if ds is None or "id" not in r:
                continue
            hechas.add((ds, r["id"]))
            resultados.setdefault(ds, []).append(
                {k: v for k, v in r.items() if k not in ("ds", "tokens")})
            uso_total["in"] += (r.get("tokens") or {}).get("in", 0)
            uso_total["out"] += (r.get("tokens") or {}).get("out", 0)
        if hechas:
            print("progress recovered from %s: %s entries already evaluated\n"
                  % (prog_path.name, f"{len(hechas):,}"))

    pendientes = {ds: [i for i in items if (ds, i["id"]) not in hechas]
                  for ds, items in trabajo.items()}

    for ds, items in trabajo.items():
        pend = pendientes[ds]
        n_cand = sum(len(i["candidatos"]) for i in pend)
        n_cons = 0 if args.reuse_consensus_eval else len(pend)
        print("%-9s %4d entries | %4d already done | %4d left = %4d calls"
              % (ds, len(items), len(items) - len(pend), len(pend), n_cand + n_cons))

    total = sum(len(i["candidatos"]) + (0 if args.reuse_consensus_eval else 1)
                for v in pendientes.values() for i in v)
    n_pend = sum(len(v) for v in pendientes.values())
    print("\npending calls to the evaluator: %s" % f"{total:,}")

    if args.dry_run:
        print("\n--dry-run: no model was called.")
        print("To estimate the cost, the script uses the real usage per call that")
        print("is already recorded in the pipeline JSON files. Run without --dry-run to measure.")
        return 0

    # import the agent ONLY when the run is real
    try:
        import agente_wikitext as ag
    except ImportError as exc:
        print("Could not import agente_wikitext: %s" % exc, file=sys.stderr)
        return 1

    cfg = ag.load_pipeline_config()["evaluator_model"]
    if cfg is None:
        print("EVALUATOR_MODEL is not defined in the .env", file=sys.stderr)
        return 1
    print("evaluator: %s\n" % cfg.name)

    def evaluar(original, texto, etiqueta):
        """Same code path as Stage 5 of the pipeline."""
        resp = ag.call_with_retries(
            ag.EVALUATOR_SYSTEM,
            ag.build_evaluator_prompt(original, texto),
            etiqueta, config=cfg, num_predict=1024,
        )
        uso = {"in": resp.prompt_tokens, "out": resp.completion_tokens}
        if resp.content is None:
            return None, uso
        ev = ag.parse_evaluation(resp.content)
        if ev is None:
            return None, uso
        ev = ag._adjust_evaluation_by_completeness(ev, original, texto, 0)
        return ev, uso

    if not n_pend:
        print("nothing left to evaluate; the summary is built from what is already saved.\n")

    t0 = time.time()
    nuevas = 0            # entries evaluated in THIS run, for the pace
    prog = prog_path.open("a", encoding="utf-8")
    try:
        for ds, items in pendientes.items():
            for k, it in enumerate(items, 1):
                fila = {"id": it["id"], "candidatos": {}}
                uso_fila = {"in": 0, "out": 0}
                for c in it["candidatos"]:
                    ev, u = evaluar(it["original"], c["text"], "abl cand %s" % it["id"])
                    uso_fila["in"] += u["in"]; uso_fila["out"] += u["out"]
                    if ev:
                        fila["candidatos"][c["model"]] = ev.get("overall_score")
                if args.reuse_consensus_eval:
                    fila["consenso"] = None
                else:
                    ev, u = evaluar(it["original"], it["consenso"], "abl cons %s" % it["id"])
                    uso_fila["in"] += u["in"]; uso_fila["out"] += u["out"]
                    fila["consenso"] = ev.get("overall_score") if ev else None

                uso_total["in"] += uso_fila["in"]; uso_total["out"] += uso_fila["out"]

                # If EVERYTHING failed for this entry it is not marked as done:
                # this way the next run retries it instead of treating it as lost.
                if not fila["candidatos"] and fila["consenso"] is None:
                    print("\n  !! %s id=%s has no result, it will stay pending" % (ds, it["id"]))
                    continue

                resultados.setdefault(ds, []).append(fila)
                # the line is written and the buffer flushed BEFORE going on: if
                # the process dies here, this entry is not paid for again
                prog.write(json.dumps({"ds": ds, **fila, "tokens": uso_fila},
                                      ensure_ascii=False) + "\n")
                prog.flush()

                # ── on-screen progress ──
                # The pace is computed with a GLOBAL counter of new entries,
                # not with the dataset index: when moving to the second corpus
                # the index restarts while the clock keeps running, and the
                # estimate would be ruined.
                nuevas += 1
                seg = time.time() - t0
                por_entrada = seg / nuevas
                faltan = n_pend - nuevas
                eta = faltan * por_entrada / 60

                vals = [f for f in resultados[ds] if f["consenso"] is not None and f["candidatos"]]
                if vals:
                    m_cons = sum(f["consenso"] for f in vals) / len(vals)
                    m_mejor = sum(max(f["candidatos"].values()) for f in vals) / len(vals)
                    marcador = "cons %.1f  best cand %.1f" % (m_cons, m_mejor)
                else:
                    marcador = "no data yet"
                print("  %-9s %4d/%-4d  %s  |  %.1f s/entry  %d left (~%.0f min)   "
                      % (ds, k, len(items), marcador, por_entrada, faltan, max(eta, 0)),
                      end="\r", flush=True)
            if items:
                print()   # keeps the last line of the dataset before moving to the next one
    except KeyboardInterrupt:
        print("\n\ninterrupted. Progress was saved in %s" % prog_path)
        print("Run the same command again to resume where it stopped.")
        return 130
    finally:
        prog.close()

    coste = (uso_total["in"] * PRECIO_IN + uso_total["out"] * PRECIO_OUT) / 1e6
    payload = {"seed": SEED, "evaluator": cfg.name, "sample": args.sample,
               "tokens": uso_total, "cost_usd": round(coste, 4), "results": resultados}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUTDIR / "ablation_consensus.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── summary ──
    print("\n" + "=" * 72)
    print("%-10s %10s %10s %10s %10s" % ("dataset", "consensus", "best model", "oracle", "n"))
    print("=" * 72)
    for ds, filas in resultados.items():
        val = [f for f in filas if f["consenso"] is not None and f["candidatos"]]
        if not val:
            continue
        cons = sum(f["consenso"] for f in val) / len(val)
        modelos = sorted({m for f in val for m in f["candidatos"]})
        medias = {m: sum(f["candidatos"][m] for f in val if m in f["candidatos"])
                     / max(sum(1 for f in val if m in f["candidatos"]), 1) for m in modelos}
        mejor_fijo = max(medias.values()) if medias else 0
        oraculo = sum(max(f["candidatos"].values()) for f in val) / len(val)
        print("%-10s %10.2f %10.2f %10.2f %10d" % (ds, cons, mejor_fijo, oraculo, len(val)))
        for m, v in sorted(medias.items(), key=lambda x: -x[1]):
            print("             %-52s %6.2f" % (str(m)[-50:], v))
    print()
    print("tokens: %s in / %s out   cost: $%.2f USD"
          % (f"{uso_total['in']:,}", f"{uso_total['out']:,}", coste))
    print("saved: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
