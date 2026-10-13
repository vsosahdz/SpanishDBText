#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the five WikiText entries translated again into the corpus and the trace.

For each ID in results/, it replaces the `spanish` block of the corpus entry (the
`english` block, including its metadata, is kept) and the whole trace entry
(candidates, review, evaluation, tokens) in every copy of the WikiText files:

  corpus/final/...json.gz       the released corpus
  corpus/pipeline/...json.gz    the released trace
  output/wikitext/...json       the expanded copies, if scripts/unpack_corpus.py made them

Each new trace entry carries a `rerun` block (date, reason, replaced score), and
the trace summary a `corrections` block with the cost of the correction run.
That cost is not added to `total_cost_usd`, which stays the cost of the
production run; the paper and the README report the sum of the two (US$454.30).

The script can be run again safely: a trace entry that already has a `rerun`
block is left as it is, so the replaced score is never overwritten, and the
original entries are saved to results/replaced_entries.json only once. The
released corpus already contains the corrected entries.

Usage (from the repository root):
    python3 scripts/correction/apply_retranslation.py
"""
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAME = "wikitext-103-raw.wiki.train.selected_1000_260620_235838.json"
PNAME = "wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json"
TARGETS_MAIN = [REPO / "corpus/final" / (NAME + ".gz"), REPO / "output/wikitext" / NAME]
TARGETS_PIPE = [REPO / "corpus/pipeline" / (PNAME + ".gz"), REPO / "output/wikitext" / PNAME]
BACKUP = HERE / "results" / "replaced_entries.json"


def read(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    reruns = {d["article_id"]: d for d in
              (json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "results").glob("wikitext_*.json")))}
    assert sorted(reruns) == [6, 52, 264, 308, 338], sorted(reruns)
    replaced = {"main": {}, "pipeline": {}}

    for path in TARGETS_MAIN:
        if not path.exists():
            print(f"  (not found) {path.relative_to(REPO)}")
            continue
        obj = read(path)
        data = obj["payload"][0]["data"]
        changed = 0
        for i, e in enumerate(data):
            if e["id"] in reruns:
                new = dict(reruns[e["id"]]["result"])
                assert new["english"]["text"] == e["english"]["text"], f"the source changed in {e['id']}"
                new["english"] = e["english"]  # only the Spanish side changes; the English metadata is kept
                if new == e:
                    continue
                replaced["main"].setdefault(e["id"], e)
                data[i] = new
                changed += 1
        if changed:
            write(path, obj)
        print(f"  corpus {path.relative_to(REPO)}: {changed} entries replaced")

    for path in TARGETS_PIPE:
        if not path.exists():
            print(f"  (not found) {path.relative_to(REPO)}")
            continue
        obj = read(path)
        entries = obj["pipeline"]
        changed = 0
        for i, e in enumerate(entries):
            if e["article_id"] in reruns and "rerun" not in e:
                replaced["pipeline"].setdefault(e["article_id"], e)
                d = reruns[e["article_id"]]
                new = dict(d["pipeline_entry"])
                old_ev = e.get("evaluation") or {}
                new["rerun"] = {"rerun_at": d["rerun_at"], "reason": d["reason"], "attempt": d["attempt"],
                                "replaced_overall_score": old_ev.get("overall_score")}
                entries[i] = new
                changed += 1
        cost = round(sum(d["pipeline_entry"]["token_usage"]["cost_usd"]["total_usd"] for d in reruns.values()), 4)
        if changed or "corrections" not in obj["summary"]:
            obj["summary"]["corrections"] = {
                "retranslated_ids": sorted(reruns),
                "date": max(d["rerun_at"] for d in reruns.values())[:10],
                "reason": "spanish side left untranslated in the production run",
                "cost_usd": cost,
                "note": "not included in total_cost_usd, which is the cost of the production run",
            }
            write(path, obj)
        print(f"  trace  {path.relative_to(REPO)}: {changed} entries replaced (correction run US${cost})")

    if not BACKUP.exists() and (replaced["main"] or replaced["pipeline"]):
        BACKUP.write_text(json.dumps(replaced, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  replaced entries saved to {BACKUP.relative_to(REPO)}")


if __name__ == "__main__":
    main()
