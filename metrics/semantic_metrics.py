# -*- coding: utf-8 -*-
"""6. Metricas semanticas cross-lingues con LaBSE.

Mide preservacion de significado entre cada entrada y su traduccion, sin
necesidad de una traduccion humana de referencia.

POR QUE UN MODELO MULTILINGUE Y NO DOS MONOLINGUES
    Comparar salidas agregadas de dos modelos entrenados por separado (por
    ejemplo, contar entidades con spaCy en cada idioma) mide la diferencia
    entre los modelos, no el texto. LaBSE proyecta ambos idiomas en un mismo
    espacio vectorial, asi que la comparacion es directa.

POR QUE ALINEACION POR ORACION Y NO PROMEDIO DEL DOCUMENTO
    Promediar los vectores de todas las oraciones de un documento y comparar
    los dos promedios NO detecta omisiones: a un documento al que le falta el
    ultimo tercio le queda un promedio muy parecido al del original. En su
    lugar, para cada oracion del origen se busca su mejor pareja en el destino:

      fidelity  = media de las similitudes maximas  -> que tan bien se traduce
      omission  = fraccion de oraciones origen cuya mejor pareja queda por
                  debajo del umbral                 -> que tanto se pierde

CUIDADO CON EL SUBMUESTREO (error detectado en el piloto del 2026-09-09)
    Si se topa el numero de oraciones en AMBOS lados por separado, cada lado
    queda con oraciones de posiciones distintas del documento y muchas
    oraciones origen se quedan sin contraparte disponible. En el piloto eso
    produjo 28-30% de omision falsa en los documentos que alcanzaban el tope,
    frente a 1-6% en los que no lo alcanzaban. Por eso el tope se aplica solo
    al lado origen y el destino se conserva siempre completo.

SALIDAS
    - por entrada : fidelity, omission, n de oraciones, similitud del centroide
    - por dataset : distribucion de las anteriores
    - entre datasets: distancia coseno entre centroides, que convierte la
                  afirmacion de complementariedad de lexica a semantica

USO
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
SEED = 42                 # el mismo que usa el resto del proyecto
OMISSION_THRESHOLD = 0.60  # por debajo de esto, una oracion se cuenta como sin pareja
MIN_SENT_CHARS = 15        # descarta fragmentos que no son oraciones

TAG = re.compile(r"<[^>\n]{0,200}>")          # marcado XML de Europarl

# ─── destokenizacion de la fuente ────────────────────────────────────────────
# El lado ingles de FEVER (99.8% de sus entradas) y de WikiText viene tokenizado
# al estilo Penn Treebank: los signos van separados por espacios y los
# delimitadores de Wikipedia se codifican como secuencias de cinco caracteres.
# El lado espanol es texto natural. Sin normalizar, esa asimetria rompe la
# segmentacion: "-LRB- e.g. , . co.uk -RRB-" produce cortes falsos donde no hay
# fin de oracion, y una entrada de 3 oraciones queda troceada en 19.
# Es la misma causa que distorsiona la razon de caracteres (ver observaciones 1.9).
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
       "-LCB-": "{", "-RCB-": "}"}
DOT_NAME = re.compile(r"\s+\.\s*([a-z]{2,4})\b")   # ". co" -> ".co"
# En mayusculas la regla anterior seria peligrosa: ". However" SI es fin de
# oracion. Se listan explicitamente los pocos casos que nunca inician oracion.
DOT_UPPER = re.compile(r"\s+\.\s*(NET|COM|ORG|GOV|EDU|EXE|PDF)\b")
# WikiText marca los encabezados de seccion con signos de igual:
#   "= = History = ="  ->  "History"
# Son el 6.31% de las oraciones del lado ingles y aparecen en los 1,000
# documentos, mientras que el espanol solo conserva el 0.56%. Sin quitarlos,
# cada encabezado queda sin pareja y se cuenta como omision. Se conserva el
# TEXTO del titulo (no se borra la linea) para que pueda emparejar con su
# traduccion cuando el lado espanol si la tiene.
HEADING = re.compile(r"^[ \t]*=[ =\t]*(.*?)[ =\t]*=[ \t]*$", re.M)
SPACE_BEFORE = re.compile(r"[ \t]+([,.;:!?%)\]}])")
SPACE_AFTER = re.compile(r"([(\[{])[ \t]+")
CLITIC = re.compile(r"[ \t]+('s|'t|'re|'ve|'ll|'d|n't)\b")


def detokenize(text: str) -> str:
    """Deshace la tokenizacion PTB para que ambos lados sean comparables."""
    text = HEADING.sub(r"\1", text)
    for k, v in PTB.items():
        text = text.replace(k, v)
    text = DOT_NAME.sub(r".\1", text)
    text = DOT_UPPER.sub(r".\1", text)
    text = SPACE_BEFORE.sub(r"\1", text)
    text = SPACE_AFTER.sub(r"\1", text)
    text = CLITIC.sub(r"\1", text)
    return re.sub(r"[ \t]+", " ", text)
# Corta en puntuacion final Y en salto de linea: Europarl separa intervenciones
# y titulos con saltos de linea sin punto, y sin esto quedan "oraciones" de
# cientos de palabras que LaBSE truncaria a 256 word-pieces, descartando
# contenido en silencio.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SENT_WORDS = 120                       # ultimo recurso: parte lo que siga siendo enorme


# ─── carga y segmentacion ────────────────────────────────────────────────────

def load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["payload"][0]["data"]


def sentences(text: str, cap: int | None = None) -> list[str]:
    """Divide en oraciones. Con cap, muestrea uniformemente conservando el
    inicio y el final, para no perder el truncamiento al final del documento."""
    text = detokenize(TAG.sub(" ", text or ""))
    out = []
    for s in SENT_SPLIT.split(text):
        s = (s or "").strip()
        if len(s) < MIN_SENT_CHARS:
            continue
        w = s.split()
        if len(w) <= MAX_SENT_WORDS:
            out.append(s)
        else:   # trozos del mismo tamano, para no perder la cola por truncamiento
            for k in range(0, len(w), MAX_SENT_WORDS):
                out.append(" ".join(w[k:k + MAX_SENT_WORDS]))
    if cap and len(out) > cap:
        step = len(out) / cap
        idx = sorted({min(int(i * step), len(out) - 1) for i in range(cap)} | {0, len(out) - 1})
        out = [out[i] for i in idx]
    return out


# ─── metricas por entrada ────────────────────────────────────────────────────

def entry_scores(sim, threshold: float) -> dict:
    """sim: matriz (n_origen x n_destino) de similitudes coseno."""
    best = sim.max(axis=1)                      # mejor pareja de cada oracion origen
    return {
        "fidelity": float(best.mean()),
        "fidelity_min": float(best.min()),
        "omission_rate": float((best < threshold).mean()),
        "unmatched": int((best < threshold).sum()),
    }


# ─── principal ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CORPORA),
                    choices=list(CORPORA), help="datasets a procesar")
    ap.add_argument("--sample", type=int, default=200,
                    help="entradas por dataset (piloto). 0 o --full = todas")
    ap.add_argument("--full", action="store_true", help="procesa el corpus completo")
    ap.add_argument("--cap-sentences", type=int, default=0,
                    help="maximo de oraciones ORIGEN por documento (0 = sin limite). "
                         "El lado destino siempre se conserva completo: topar ambos "
                         "lados los desalinea y genera omisiones falsas.")
    ap.add_argument("--batch", type=int, default=64, help="tamano de lote del encoder")
    ap.add_argument("--threshold", type=float, default=OMISSION_THRESHOLD,
                    help="umbral de similitud para contar una oracion como omitida")
    ap.add_argument("--ids", default=None,
                    help="lista de ids separados por coma: procesa solo esas entradas. "
                         "Sirve para validar la medida contra casos conocidos.")
    ap.add_argument("--device", default=None, help="cuda | cpu (por defecto, automatico)")
    ap.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("Falta una dependencia: %s" % exc, file=sys.stderr)
        print("Instala con:  pip install -r requirements.txt", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("dispositivo : %s" % device)
    if device == "cuda":
        print("GPU         : %s" % torch.cuda.get_device_name(0))
    print("modelo      : %s  (la primera vez descarga ~1.8 GB)" % MODEL_NAME)

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("modelo cargado en %.1f s\n" % (time.time() - t0))

    cap = args.cap_sentences or None
    rng = random.Random(SEED)
    results, centroids = {}, {}

    for ds in args.datasets:
        path = CORPORA[ds]
        if not path.exists():
            print("  !! no existe %s, se omite" % path, file=sys.stderr)
            continue
        entries = load_entries(path)
        pairs = [e for e in entries
                 if (e.get("english") or {}).get("text") and (e.get("spanish") or {}).get("text")]
        if args.ids:
            want = {s.strip() for s in args.ids.split(",") if s.strip()}
            pairs = [e for e in pairs if str(e.get("id")) in want]
            print("  filtrado por --ids: %d de %d encontradas" % (len(pairs), len(want)))
        elif not args.full and args.sample:
            pairs = rng.sample(pairs, min(args.sample, len(pairs)))

        print("%s: %d entradas" % (ds, len(pairs)))
        per_entry, doc_vecs = [], []
        t1 = time.time()

        for i, e in enumerate(pairs, 1):
            # El tope se aplica SOLO al lado origen. El destino se conserva
            # completo para que toda oracion origen tenga disponible su
            # contraparte real. Submuestrear ambos lados por separado los
            # desalinea y produce omisiones falsas (ver docstring).
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
        print("   fidelidad media %.4f | omision media %.4f | %.1f s\n"
              % (fid.mean(), omi.mean(), time.time() - t1))

    # distancia semantica entre datasets (complementariedad)
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
    print("%-10s %10s %10s %10s" % ("dataset", "fidelidad", "omision", "entradas"))
    for ds, r in results.items():
        print("%-10s %10.4f %10.4f %10d"
              % (ds, r["fidelity"]["mean"], r["omission_rate"]["mean"], r["n_entries"]))
    print()
    print("DISTANCIA SEMANTICA ENTRE DATASETS (1 - coseno de centroides)")
    for k, v in inter.items():
        print("  %-24s %.4f" % (k, v["distance"]))
    print()
    print("guardado: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
