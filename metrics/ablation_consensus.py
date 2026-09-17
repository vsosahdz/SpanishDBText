# -*- coding: utf-8 -*-
"""Ablacion de la Etapa 4: el consenso, comparado contra sus propios candidatos.

PREGUNTA
    El articulo presenta la fusion por consenso como su aportacion central y esa
    etapa consume el 77% del presupuesto, pero nunca se compara contra nada. Un
    revisor preguntara: "mejor, comparado con que?".

DISENO
    Los tres candidatos ya estan guardados en los pipeline JSON, asi que NO hay
    que retraducir nada. Solo se evalua cada candidato con el MISMO evaluador de
    la Etapa 5: mismo modelo, mismo prompt, misma escala, mismo ajuste
    programatico de completitud. Por eso este script IMPORTA las funciones del
    agente en lugar de reimplementarlas; cualquier reimplementacion invalidaria
    la comparacion.

DOS BASELINES, NO UNO
    (a) CADA MODELO FIJO: "si simplemente hubieras usado Mistral Large para
        todo". Es el baseline realista, el que alguien podria adoptar.
    (b) MEJOR CANDIDATO POR ENTRADA: un ORACULO. En un pipeline real no se sabe
        de antemano cual modelo sera el mejor en cada entrada, asi que es una
        cota superior, no una alternativa disponible. Si el consenso pierde
        contra el oraculo pero gana a todos los modelos fijos, la lectura
        correcta es que el consenso equivale a un selector perfecto.

    Reportar solo (b) seria injusto con el metodo; reportar solo (a) seria
    complaciente. Por eso van los dos.

RESULTADO POSIBLE INCOMODO
    Si (b) iguala al consenso, la conclusion seria que basta SELECCIONAR y no
    hace falta FUSIONAR, lo que eliminaria la etapa mas cara del pipeline. Es un
    hallazgo valioso, y conviene conocerlo antes del envio y no en la revision.

RECUPERACION ANTE FALLOS
    El proceso son miles de llamadas a una API, asi que se guarda una linea por
    entrada TERMINADA en metrics/output/ablation_progress.jsonl, con flush
    inmediato. Si el script muere -- corte de red, limite de cuota, Ctrl-C,
    reinicio -- basta volver a lanzar EL MISMO COMANDO: las entradas ya
    evaluadas se saltan y no se vuelven a pagar. Los tokens gastados en
    ejecuciones previas se acumulan, de modo que el coste que reporta al final
    es el total real y no el de la ultima tanda.
    Una entrada en la que fallo TODO no se marca como hecha, para que el
    siguiente intento la reintente. Con --restart se empieza de cero.

USO
    python metrics/ablation_consensus.py --sample 50 --dry-run
    python metrics/ablation_consensus.py --sample 20          # calibrar velocidad
    python metrics/ablation_consensus.py --sample 200         # corrida completa
    python metrics/ablation_consensus.py --sample 200         # retoma si se corto
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

# tarifas de GPT-4.1 Mini en Azure, USD por millon de tokens
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
    """Entradas con 3 candidatos y consenso disponible."""
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
    ap.add_argument("--sample", type=int, default=200, help="entradas por dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="no llama al modelo: solo estima coste y numero de llamadas")
    ap.add_argument("--reuse-consensus-eval", action="store_true",
                    help="reutiliza la evaluacion del consenso ya guardada en el pipeline JSON "
                         "en vez de recalcularla (ahorra 1 de cada 4 llamadas, pero mezcla "
                         "evaluaciones de fechas distintas)")
    ap.add_argument("--progress", default=None,
                    help="archivo de progreso JSONL (por defecto "
                         "metrics/output/ablation_progress.jsonl). Se escribe una linea "
                         "por entrada terminada; si el proceso se corta, la siguiente "
                         "ejecucion retoma donde quedo.")
    ap.add_argument("--restart", action="store_true",
                    help="ignora el progreso previo y empieza de cero")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(SEED)
    trabajo = {}
    for ds in args.datasets:
        cp, pp = CORPORA[ds]
        if not cp.exists() or not pp.exists():
            print("  !! faltan archivos de %s, se omite" % ds, file=sys.stderr)
            continue
        corpus, pipe = cargar(cp, pp)
        items = pares(corpus, pipe)
        if args.sample and len(items) > args.sample:
            items = rng.sample(items, args.sample)
        trabajo[ds] = items

    # ── progreso: se carga ANTES de informar, para que el recuento y el
    #    --dry-run reflejen lo que realmente falta por hacer ──
    prog_path = Path(args.progress) if args.progress else OUTDIR / "ablation_progress.jsonl"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.restart and prog_path.exists():
        prog_path.rename(prog_path.with_suffix(".jsonl.bak"))
        print("progreso previo movido a %s\n" % prog_path.with_suffix(".jsonl.bak").name)

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
                continue          # linea truncada por un corte: se descarta
            ds = r.get("ds")
            if ds is None or "id" not in r:
                continue
            hechas.add((ds, r["id"]))
            resultados.setdefault(ds, []).append(
                {k: v for k, v in r.items() if k not in ("ds", "tokens")})
            uso_total["in"] += (r.get("tokens") or {}).get("in", 0)
            uso_total["out"] += (r.get("tokens") or {}).get("out", 0)
        if hechas:
            print("progreso recuperado de %s: %s entradas ya evaluadas\n"
                  % (prog_path.name, f"{len(hechas):,}"))

    pendientes = {ds: [i for i in items if (ds, i["id"]) not in hechas]
                  for ds, items in trabajo.items()}

    for ds, items in trabajo.items():
        pend = pendientes[ds]
        n_cand = sum(len(i["candidatos"]) for i in pend)
        n_cons = 0 if args.reuse_consensus_eval else len(pend)
        print("%-9s %4d entradas | %4d ya hechas | faltan %4d = %4d llamadas"
              % (ds, len(items), len(items) - len(pend), len(pend), n_cand + n_cons))

    total = sum(len(i["candidatos"]) + (0 if args.reuse_consensus_eval else 1)
                for v in pendientes.values() for i in v)
    n_pend = sum(len(v) for v in pendientes.values())
    print("\nllamadas pendientes al evaluador: %s" % f"{total:,}")

    if args.dry_run:
        print("\n--dry-run: no se llamo a ningun modelo.")
        print("Para estimar el coste, el script usa el consumo real por llamada que")
        print("ya esta registrado en los pipeline JSON. Ejecuta sin --dry-run para medir.")
        return 0

    # importar el agente SOLO cuando se va a ejecutar de verdad
    try:
        import agente_wikitext as ag
    except ImportError as exc:
        print("No se pudo importar agente_wikitext: %s" % exc, file=sys.stderr)
        return 1

    cfg = ag.load_pipeline_config()["evaluator_model"]
    if cfg is None:
        print("EVALUATOR_MODEL no esta definido en el .env", file=sys.stderr)
        return 1
    print("evaluador: %s\n" % cfg.name)

    def evaluar(original, texto, etiqueta):
        """Misma ruta de codigo que la Etapa 5 del pipeline."""
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
        print("no queda nada por evaluar; se genera el resumen con lo ya guardado.\n")

    t0 = time.time()
    nuevas = 0            # entradas evaluadas en ESTA ejecucion, para el ritmo
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

                # Si TODO fallo en esta entrada no se marca como hecha: asi la
                # proxima ejecucion la reintenta en vez de darla por perdida.
                if not fila["candidatos"] and fila["consenso"] is None:
                    print("\n  !! %s id=%s sin resultado, quedara pendiente" % (ds, it["id"]))
                    continue

                resultados.setdefault(ds, []).append(fila)
                # se escribe y se vacia el buffer ANTES de seguir: si el proceso
                # muere aqui, esta entrada no se vuelve a pagar
                prog.write(json.dumps({"ds": ds, **fila, "tokens": uso_fila},
                                      ensure_ascii=False) + "\n")
                prog.flush()

                # ── avance en pantalla ──
                # El ritmo se calcula con un contador GLOBAL de entradas nuevas,
                # no con el indice del dataset: al pasar al segundo corpus el
                # indice reinicia mientras el reloj sigue, y la estimacion se
                # iria al traste.
                nuevas += 1
                seg = time.time() - t0
                por_entrada = seg / nuevas
                faltan = n_pend - nuevas
                eta = faltan * por_entrada / 60

                vals = [f for f in resultados[ds] if f["consenso"] is not None and f["candidatos"]]
                if vals:
                    m_cons = sum(f["consenso"] for f in vals) / len(vals)
                    m_mejor = sum(max(f["candidatos"].values()) for f in vals) / len(vals)
                    marcador = "cons %.1f  mejor cand %.1f" % (m_cons, m_mejor)
                else:
                    marcador = "sin datos aun"
                print("  %-9s %4d/%-4d  %s  |  %.1f s/entrada  faltan %d (~%.0f min)   "
                      % (ds, k, len(items), marcador, por_entrada, faltan, max(eta, 0)),
                      end="\r", flush=True)
            if items:
                print()   # fija la ultima linea del dataset antes de pasar al siguiente
    except KeyboardInterrupt:
        print("\n\ninterrumpido. El progreso quedo guardado en %s" % prog_path)
        print("Vuelve a ejecutar el mismo comando para retomar donde quedo.")
        return 130
    finally:
        prog.close()

    coste = (uso_total["in"] * PRECIO_IN + uso_total["out"] * PRECIO_OUT) / 1e6
    payload = {"seed": SEED, "evaluator": cfg.name, "sample": args.sample,
               "tokens": uso_total, "cost_usd": round(coste, 4), "results": resultados}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUTDIR / "ablation_consensus.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── resumen ──
    print("\n" + "=" * 72)
    print("%-10s %10s %10s %10s %10s" % ("dataset", "consenso", "mejor modelo", "oraculo", "n"))
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
    print("tokens: %s in / %s out   coste: $%.2f USD"
          % (f"{uso_total['in']:,}", f"{uso_total['out']:,}", coste))
    print("guardado: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
