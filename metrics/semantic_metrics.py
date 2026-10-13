# -*- coding: utf-8 -*-
"""6. Cross-lingual semantic metrics with LaBSE.

Measures how well meaning is preserved between each entry and its
translation, without the need for a human reference translation.

WHY ONE MULTILINGUAL MODEL AND NOT TWO MONOLINGUAL ONES
    Comparing aggregate outputs of two models trained separately (for
    example, counting entities with spaCy in each language) measures the
    difference between the models, not the text. LaBSE maps both languages
    into the same vector space, so the comparison is direct.

WHY SENTENCE ALIGNMENT AND NOT A DOCUMENT AVERAGE
    Averaging the vectors of all the sentences in a document and comparing
    the two averages does NOT detect omissions: a document that is missing
    its last third still has an average very close to that of the original.
    Instead, for each source sentence we look for its best match in the
    target:

      fidelity  = mean of the maximum similarities  -> how well it is translated
      omission  = fraction of source sentences whose best match falls
                  below the threshold               -> how much is lost

BEWARE OF SUBSAMPLING (bug found in the 2026-09-09 pilot)
    If the number of sentences is capped on BOTH sides separately, each side
    keeps sentences from different positions in the document, and many
    source sentences are left with no counterpart available. In the pilot
    this produced 28-30% false omission in the documents that reached the
    cap, against 1-6% in those that did not. This is why the cap is applied
    only to the source side and the target is always kept complete.

OUTPUTS
    - per entry   : fidelity, omission, number of sentences, centroid similarity
    - per dataset : distribution of the above
    - between datasets: cosine distance between centroids, which turns the
                  complementarity claim from lexical into semantic

USAGE
    python metrics/semantic_metrics.py --sample 200
    python metrics/semantic_metrics.py --full --batch 128
    python metrics/semantic_metrics.py --datasets fever --sample 50 --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "metrics" / "output"

CORPORA = {
    "wikitext": BASE / "output" / "wikitext" / "wikitext-103-raw.wiki.train.selected_1000_260620_235838.json",
    "fever": BASE / "output" / "FEVER" / "fever_selected_260712_212751.json",
    "europarl": BASE / "output" / "Europarl" / "selected_260710_084528.json",
}

MODEL_NAME = "sentence-transformers/LaBSE"
SEED = 42                 # the same one used by the rest of the project
OMISSION_THRESHOLD = 0.60  # below this, a sentence is counted as having no match
MIN_SENT_CHARS = 15        # drops fragments that are not sentences

TAG = re.compile(r"<[^>\n]{0,200}>")          # Europarl XML markup

# ─── source detokenization ───────────────────────────────────────────────────
# The English side of FEVER (99.8% of its entries) and of WikiText is tokenized
# in Penn Treebank style: punctuation marks are separated by spaces and the
# Wikipedia delimiters are encoded as five-character sequences.
# The Spanish side is natural text. Without normalization, this asymmetry breaks
# the segmentation: "-LRB- e.g. , . co.uk -RRB-" produces false cuts where there
# is no end of sentence, and an entry with 3 sentences is split into 19.
# It is the same cause that lowers the character ratio of 31 FEVER entries
# (see the expansion-ratio anomaly in docs/metrics-results.md).
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
       "-LCB-": "{", "-RCB-": "}"}
DOT_NAME = re.compile(r"\s+\.\s*([a-z]{2,4})\b")   # ". co" -> ".co"
# In upper case the previous rule would be dangerous: ". However" IS an end of
# sentence. The few cases that never start a sentence are listed explicitly.
DOT_UPPER = re.compile(r"\s+\.\s*(NET|COM|ORG|GOV|EDU|EXE|PDF)\b")
# WikiText marks section headings with equal signs:
#   "= = History = ="  ->  "History"
# They are 6.31% of the sentences on the English side and appear in all 1,000
# documents, while the Spanish side keeps only 0.56%. If they are not removed,
# each heading has no match and is counted as an omission. The TEXT of the
# title is kept (the line is not deleted) so that it can match its
# translation when the Spanish side does have it.
HEADING = re.compile(r"^[ \t]*=[ =\t]*(.*?)[ =\t]*=[ \t]*$", re.M)
SPACE_BEFORE = re.compile(r"[ \t]+([,.;:!?%)\]}])")
SPACE_AFTER = re.compile(r"([(\[{])[ \t]+")
CLITIC = re.compile(r"[ \t]+('s|'t|'re|'ve|'ll|'d|n't)\b")


def detokenize(text: str) -> str:
    """Undoes the PTB tokenization so that both sides are comparable."""
    text = HEADING.sub(r"\1", text)
    for k, v in PTB.items():
        text = text.replace(k, v)
    text = DOT_NAME.sub(r".\1", text)
    text = DOT_UPPER.sub(r".\1", text)
    text = SPACE_BEFORE.sub(r"\1", text)
    text = SPACE_AFTER.sub(r"\1", text)
    text = CLITIC.sub(r"\1", text)
    return re.sub(r"[ \t]+", " ", text)
# Splits at final punctuation AND at line breaks: Europarl separates speeches
# and titles with line breaks and no period, and without this we get "sentences"
# of hundreds of words that LaBSE would truncate to 256 word-pieces, silently
# dropping content.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SENT_WORDS = 120                       # last resort: splits whatever is still huge


# ─── loading and segmentation ────────────────────────────────────────────────

def load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["payload"][0]["data"]


def sentences(text: str, cap: int | None = None) -> list[str]:
    """Splits into sentences. With cap, samples evenly while keeping the
    start and the end, so that truncation at the end of the document is not missed."""
    text = detokenize(TAG.sub(" ", text or ""))
    out = []
    for s in SENT_SPLIT.split(text):
        s = (s or "").strip()
        if len(s) < MIN_SENT_CHARS:
            continue
        w = s.split()
        if len(w) <= MAX_SENT_WORDS:
            out.append(s)
        else:   # chunks of the same size, so the tail is not lost to truncation
            for k in range(0, len(w), MAX_SENT_WORDS):
                out.append(" ".join(w[k:k + MAX_SENT_WORDS]))
    if cap and len(out) > cap:
        step = len(out) / cap
        idx = sorted({min(int(i * step), len(out) - 1) for i in range(cap)} | {0, len(out) - 1})
        out = [out[i] for i in idx]
    return out


# ─── per-entry metrics ───────────────────────────────────────────────────────

def entry_scores(sim, threshold: float) -> dict:
    """sim: matrix (n_source x n_target) of cosine similarities."""
    best = sim.max(axis=1)                      # best match of each source sentence
    return {
        "fidelity": float(best.mean()),
        "fidelity_min": float(best.min()),
        "omission_rate": float((best < threshold).mean()),
        "unmatched": int((best < threshold).sum()),
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CORPORA),
                    choices=list(CORPORA), help="datasets to process")
    ap.add_argument("--sample", type=int, default=200,
                    help="entries per dataset (pilot). 0 or --full = all")
    ap.add_argument("--full", action="store_true", help="processes the full corpus")
    ap.add_argument("--cap-sentences", type=int, default=0,
                    help="maximum number of SOURCE sentences per document (0 = no limit). "
                         "The target side is always kept complete: capping both "
                         "sides misaligns them and creates false omissions.")
    ap.add_argument("--batch", type=int, default=64, help="encoder batch size")
    ap.add_argument("--threshold", type=float, default=OMISSION_THRESHOLD,
                    help="similarity threshold for counting a sentence as omitted")
    ap.add_argument("--ids", default=None,
                    help="comma-separated list of ids: processes only those entries. "
                         "Useful to validate the measure against known cases.")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: automatic)")
    ap.add_argument("--out", default=None, help="path of the output JSON")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("A dependency is missing: %s" % exc, file=sys.stderr)
        print("Install with:  pip install -r requirements.txt", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device      : %s" % device)
    if device == "cuda":
        print("GPU         : %s" % torch.cuda.get_device_name(0))
    print("model       : %s  (the first run downloads ~1.8 GB)" % MODEL_NAME)

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("model loaded in %.1f s\n" % (time.time() - t0))

    cap = args.cap_sentences or None
    rng = random.Random(SEED)
    results, centroids = {}, {}

    for ds in args.datasets:
        path = CORPORA[ds]
        if not path.exists():
            print("  !! %s does not exist, skipped" % path, file=sys.stderr)
            continue
        entries = load_entries(path)
        pairs = [e for e in entries
                 if (e.get("english") or {}).get("text") and (e.get("spanish") or {}).get("text")]
        if args.ids:
            want = {s.strip() for s in args.ids.split(",") if s.strip()}
            pairs = [e for e in pairs if str(e.get("id")) in want]
            print("  filtered by --ids: %d of %d found" % (len(pairs), len(want)))
        elif not args.full and args.sample:
            pairs = rng.sample(pairs, min(args.sample, len(pairs)))

        print("%s: %d entries" % (ds, len(pairs)))
        per_entry, doc_vecs = [], []
        t1 = time.time()

        for i, e in enumerate(pairs, 1):
            # The cap is applied ONLY to the source side. The target is kept
            # complete so that every source sentence has its real
            # counterpart available. Subsampling both sides separately
            # misaligns them and produces false omissions (see docstring).
            en = sentences(e["english"]["text"], cap)
            es = sentences(e["spanish"]["text"])
            if not en or not es:
                continue
            emb = model.encode(en + es, batch_size=args.batch,
                               convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
            a, b = emb[:len(en)], emb[len(en):]
            sc = entry_scores(a @ b.T, args.threshold)
            sc.update(id=e.get("id"), n_src=len(en), n_tgt=len(es))
            per_entry.append(sc)
            doc_vecs.append(np.concatenate([a.mean(0), b.mean(0)]))
            if i % 25 == 0 or i == len(pairs):
                print("   %4d/%d  %.1f s" % (i, len(pairs), time.time() - t1), end="\r")

        print()
        fid = np.array([s["fidelity"] for s in per_entry])
        omi = np.array([s["omission_rate"] for s in per_entry])
        results[ds] = {
            "n_entries": len(per_entry),
            "fidelity": {"mean": float(fid.mean()), "median": float(np.median(fid)),
                         "std": float(fid.std()), "min": float(fid.min()),
                         "p05": float(np.percentile(fid, 5))},
            "omission_rate": {"mean": float(omi.mean()), "median": float(np.median(omi)),
                              "max": float(omi.max()),
                              "entries_with_any": int((omi > 0).sum())},
            "threshold": args.threshold,
            "worst": sorted(per_entry, key=lambda s: s["fidelity"])[:10],
            "per_entry": per_entry,
        }
        centroids[ds] = np.stack(doc_vecs).mean(0)
        print("   mean fidelity %.4f | mean omission %.4f | %.1f s\n"
              % (fid.mean(), omi.mean(), time.time() - t1))

    # semantic distance between datasets (complementarity)
    inter = {}
    keys = list(centroids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u, v = centroids[keys[i]], centroids[keys[j]]
            cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            inter["%s vs %s" % (keys[i], keys[j])] = {"cosine": cos, "distance": 1 - cos}

    payload = {
        "model": MODEL_NAME,
        "device": device,
        "seed": SEED,
        "mode": "full" if args.full else "sample=%d" % args.sample,
        "cap_sentences": cap,
        "omission_threshold": args.threshold,
        "datasets": results,
        "inter_dataset": inter,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUTDIR / (
        "semantic_%s.json" % ("full" if args.full else "pilot"))
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 64)
    print("%-10s %10s %10s %10s" % ("dataset", "fidelity", "omission", "entries"))
    for ds, r in results.items():
        print("%-10s %10.4f %10.4f %10d"
              % (ds, r["fidelity"]["mean"], r["omission_rate"]["mean"], r["n_entries"]))
    print()
    print("SEMANTIC DISTANCE BETWEEN DATASETS (1 - cosine of centroids)")
    for k, v in inter.items():
        print("  %-24s %.4f" % (k, v["distance"]))
    print()
    print("saved: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
