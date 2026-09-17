"""
Agente de traducción para corpus FEVER (Fact Extraction and VERification).

Trabaja con archivos .jsonl donde cada línea es un extracto de Wikipedia:
    {"id": "1996_BellSouth_Open", "text": "The 1996 BellSouth Open...", "lines": "..."}

Estrategia:
    - Cada línea JSON = un artículo independiente
    - El campo `id` (título Wikipedia con guiones bajos) sirve como identificador
    - El campo `text` contiene el texto plano a traducir
    - Los artículos son cortos (mediana ~50 palabras) → siempre 1 chunk
    - Acepta el archivo JSONL completo o el subset seleccionado por el notebook

Uso:
    # Subset seleccionado (recomendado):
    python3 agente_fever.py -i data/FEVER/selected/fever_selected.jsonl

    # Con límite para prueba:
    python3 agente_fever.py -i data/FEVER/selected/fever_selected.jsonl --limit 10

    # Retomar desde artículo específico:
    python3 agente_fever.py -i data/FEVER/selected/fever_selected.jsonl --start-from 50
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from agente_wikitext import (
    MAX_RETRIES,
    ModelConfig,
    LLMResponse,
    load_pipeline_config,
    verify_all_models,
    detect_language,
    call_with_retries,
    clean_llm_response,
    calculate_cost,
    parse_metadata,
    parse_evaluation,
    load_progress,
    save_progress,
    TRANSLATION_SYSTEM,
    METADATA_SYSTEM_ES,
    METADATA_SYSTEM_EN,
    REVIEWER_CRITIQUE_SYSTEM,
    REVIEWER_MERGE_SYSTEM,
    EVALUATOR_SYSTEM,
    TRANSLATION_LANGUAGE_RETRIES,
    build_translation_retry_prompt,
    build_evaluator_prompt,
    _adjust_evaluation_by_completeness,
    _extract_error_detail,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/FEVER")


# ─── Prompts adaptados para FEVER ────────────────────────────────────────────

FEVER_TRANSLATION_SYSTEM = """Eres un traductor profesional especializado en enciclopedias y artículos factuales.

Reglas:
1. Fidelidad semántica: NO inventes, NO omitas, NO agregues nada que no esté en el original.
2. Naturalidad: evita traducciones literales robóticas.
3. Terminología: conserva términos técnicos correctos. No traduzcas nombres propios (personas, equipos, lugares, torneos).
4. Estilo: respeta el tono enciclopédico neutral del original.
5. Estructura: conserva párrafos y puntuación importante.
6. Calidad: español neutro profesional / inglés neutro.
7. PROHIBIDO: resumir, simplificar, reinterpretar, inferir datos faltantes o censurar.
8. NO agregues formato markdown (negritas **, cursivas *, encabezados #) a menos que esté en el original.
9. Anglicismos deportivos: usa el término establecido en español cuando exista (ej. "golpe de aproximación" en lugar de "approach", "ponchar" en lugar de "strikeout", "tiro libre" en lugar de "free throw"). Si no hay equivalente establecido, mantén el término inglés.

Responde SOLO con la traducción, sin explicaciones ni prefijos."""


def build_fever_translation_prompt(text: str) -> str:
    return f"""Traduce el siguiente extracto enciclopédico al español. Si ya está en español, tradúcelo al inglés.
Responde SOLO con la traducción, sin explicaciones ni prefijos.

{text}"""


def build_fever_metadata_prompt(text: str, title: str, lang: str = "en") -> str:
    """Genera prompt de metadata para artículos FEVER (extractos cortos).

    Para metadata en inglés: usa el título original (nombre propio Wikipedia).
    Para metadata en español: traduce el título cuando sea posible, conservando
    nombres propios que no tienen equivalente establecido en español.
    """
    sample = text[:2000]
    if lang == "en":
        return f"""Analyze the following encyclopedia excerpt and respond EXACTLY with this format (one line per field):

TITLE: [clean article title (use: {title})]
KEYWORDS: [between 3 and 6 keywords separated by comma]
CLASSIFICATION: [main category: sports, music, film, history, politics, science, geography, biography, technology, religion, military, transport, or other appropriate]
SUMMARY: [brief informative summary, maximum 60 words]

Respond ONLY with these 4 fields.

=== EXCERPT ===
{sample}"""
    return f"""Analiza el siguiente extracto enciclopédico (ya traducido al español) y responde EXACTAMENTE con este formato (una línea por campo):

TITLE: [título en español: traduce el título "{title}" al español cuando sea posible (ej. "Temporada 1999 de los Detroit Lions", "Serie Mundial de 1980"); conserva en inglés solo los nombres propios sin equivalente establecido]
KEYWORDS: [entre 3 y 6 keywords en español, separadas por coma]
CLASSIFICATION: [categoría principal: sports, music, film, history, politics, science, geography, biography, technology, religion, military, transport, u otra apropiada]
SUMMARY: [resumen informativo breve en español, máximo 60 palabras]

Responde SOLO con estos 4 campos.

=== EXTRACTO ===
{sample}"""


# ─── Parsing de entradas FEVER ───────────────────────────────────────────────

def clean_fever_id(raw_id: str) -> str:
    """Convierte el id FEVER a título legible.

    Ejemplos:
        '1996_BellSouth_Open'       → '1996 BellSouth Open'
        'Battle_of_Hastings'        → 'Battle of Hastings'
        'Albert_Einstein_-LRB-disambiguation-RRB-' → 'Albert Einstein (disambiguation)'
    """
    s = raw_id.replace('-LRB-', '(').replace('-RRB-', ')')
    s = s.replace('_', ' ')
    return s.strip()


def load_fever_entries(jsonl_path: Path, min_words: int = 0) -> list[dict]:
    """Carga artículos del JSONL. Retorna lista de dicts con id, titulo, text."""
    entries = []
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "").strip()
            if not text:
                skipped += 1
                continue
            words = len(text.split())
            if min_words > 0 and words < min_words:
                skipped += 1
                continue
            entries.append({
                "fever_id": obj["id"],
                "titulo": clean_fever_id(obj["id"]),
                "text": text,
                "words": words,
            })
    log.info(f"Artículos cargados: {len(entries):,} (descartados: {skipped})")
    return entries


# ─── Stage 1: Traducción ────────────────────────────────────────────────────

def _translate_article(text: str, source_lang: str,
                       config: ModelConfig, label: str) -> tuple[str | None, dict, bool]:
    """Traduce un extracto FEVER. Siempre 1 chunk (artículos cortos)."""
    target_lang = "es" if source_lang == "en" else "en"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    resp = call_with_retries(
        FEVER_TRANSLATION_SYSTEM,
        build_fever_translation_prompt(text),
        label,
        config=config,
    )
    if resp.content is None:
        return None, usage, resp.content_filter
    usage["prompt_tokens"] += resp.prompt_tokens
    usage["completion_tokens"] += resp.completion_tokens
    usage["total_tokens"] += resp.total_tokens

    translated = clean_llm_response(resp.content)

    # Validar idioma
    detected = detect_language(translated)
    if detected == target_lang:
        return translated, usage, False

    # Reintento si idioma incorrecto
    for retry in range(1, TRANSLATION_LANGUAGE_RETRIES + 1):
        log.warning(f"      [{config.name}] {label} — output en '{detected}' (esperado '{target_lang}'), retry {retry}")
        resp = call_with_retries(
            FEVER_TRANSLATION_SYSTEM,
            build_translation_retry_prompt(text, target_lang),
            f"{label} (lang retry {retry})",
            config=config,
        )
        if resp.content is None:
            continue
        usage["prompt_tokens"] += resp.prompt_tokens
        usage["completion_tokens"] += resp.completion_tokens
        usage["total_tokens"] += resp.total_tokens
        translated = clean_llm_response(resp.content)
        detected = detect_language(translated)
        if detected == target_lang:
            return translated, usage, False

    return translated, usage, False


def stage_1_generate(text: str, article_id: int,
                     models: list[ModelConfig]) -> tuple[list[dict], dict, list[str]]:
    """Genera traducciones candidatas en paralelo."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates = []
    stage_usage = {}
    cf_models = []

    def _run(config):
        try:
            log.info(f"    [Stage 1] Traduciendo con {config.name}...")
            source_lang = detect_language(text)
            translated, usage, cf = _translate_article(
                text, source_lang, config,
                f"translation article {article_id} [{config.name}]",
            )
            return config, (translated, [translated], usage, cf), False
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            detail, is_cf = _extract_error_detail(e)
            if is_cf:
                log.error(f"    [Stage 1] {config.name} — CONTENT FILTER: {detail[:200]}")
            else:
                log.error(f"    [Stage 1] {config.name} — error: {type(e).__name__}: {e}")
            return config, None, is_cf

    def _process(config, result, cf_flag):
        if result:
            text_out, chunks, usage, cf = result
            if text_out:
                candidates.append({
                    "model": config.name, "text": text_out, "chunks": chunks,
                    "translated_at": datetime.now().isoformat(),
                })
                stage_usage[config.name] = usage
                if cf:
                    cf_models.append(config.name)
                log.info(f"    [Stage 1] {config.name} — OK ({len(text_out):,} chars, {usage['total_tokens']} tok)")
            else:
                log.warning(f"    [Stage 1] {config.name} — traducción vacía")
        else:
            if cf_flag:
                cf_models.append(config.name)
            log.warning(f"    [Stage 1] {config.name} — failed")

    if len(models) <= 1:
        for config in models:
            c, result, cf = _run(config)
            _process(c, result, cf)
    else:
        log.info(f"    [Stage 1] Translating with {len(models)} models in parallel...")
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(_run, c): c for c in models}
            for future in as_completed(futures):
                c, result, cf = future.result()
                _process(c, result, cf)

    return candidates, stage_usage, cf_models


# ─── Stage 2: Revisión ──────────────────────────────────────────────────────

def stage_2_review(original: str, candidates: list[dict],
                   article_id: int, reviewer_config: ModelConfig) -> tuple[dict | None, dict]:
    """Revisión por consenso (siempre 1 sección — artículos cortos)."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not candidates:
        return None, usage

    log.info(f"    [Stage 2] Revisando con {reviewer_config.name}...")

    # Critique
    parts = [f"=== TEXTO ORIGINAL ===\n{original[:2000]}\n"]
    for i, c in enumerate(candidates, 1):
        parts.append(f"=== CANDIDATO {i} (modelo: {c['model']}, {len(c['text']):,} chars) ===\n{c['text'][:2000]}\n")
    parts.append("Evalúa cada candidato y decide cuál usar como base. Responde solo con JSON.")

    resp = call_with_retries(
        REVIEWER_CRITIQUE_SYSTEM, "\n".join(parts),
        f"critique article {article_id}", config=reviewer_config, num_predict=1024,
    )
    usage["prompt_tokens"] += resp.prompt_tokens
    usage["completion_tokens"] += resp.completion_tokens
    usage["total_tokens"] += resp.total_tokens

    if resp.content is None:
        log.warning("    [Stage 2] Critique falló — fallback a candidato 1")
        return None, usage

    json_match = re.search(r'\{.*\}', resp.content, flags=re.DOTALL)
    critique = {}
    if json_match:
        try:
            critique = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            critique = {"raw": resp.content[:500]}
    else:
        critique = {"raw": resp.content[:500]}

    log.info("    [Stage 2] Critique completada")

    # Merge (1 sola sección)
    best_model = None
    if isinstance(critique, dict) and "candidates_analysis" in critique:
        analyses = critique["candidates_analysis"]
        best = max(analyses, key=lambda x: x.get("score", 0))
        best_model = best.get("model", "")

    merge_parts = [f"=== SECCIÓN ORIGINAL ===\n{original}\n"]
    for i, c in enumerate(candidates, 1):
        label = "(MEJOR)" if best_model and best_model in c["model"] else ""
        merge_parts.append(f"=== CANDIDATO {i} {label} ({c['model']}) ===\n{c['text']}\n")
    merge_parts.append(
        f"Decisión: {critique.get('decision', 'Usar el mejor candidato como base.')}\n"
        "Genera la traducción final mejorada. Responde SOLO con el texto traducido."
    )

    merge_resp = call_with_retries(
        REVIEWER_MERGE_SYSTEM, "\n".join(merge_parts),
        f"merge article {article_id}", config=reviewer_config,
    )
    usage["prompt_tokens"] += merge_resp.prompt_tokens
    usage["completion_tokens"] += merge_resp.completion_tokens
    usage["total_tokens"] += merge_resp.total_tokens

    if merge_resp.content is None:
        log.warning("    [Stage 2] Merge falló")
        return None, usage

    consensus = clean_llm_response(merge_resp.content)
    review = {
        "critique": critique, "consensus_text": consensus,
        "model": reviewer_config.name, "reviewed_at": datetime.now().isoformat(),
    }
    log.info(f"    [Stage 2] Revisión completada ({len(consensus):,} chars, {usage['total_tokens']} tok)")
    return review, usage


# ─── Stage 3: Evaluación ────────────────────────────────────────────────────

def stage_3_evaluate(original: str, translation: str,
                     article_id: int, evaluator_config: ModelConfig) -> tuple[dict | None, dict]:
    """Evalúa la traducción."""
    log.info(f"    [Stage 3] Evaluando con {evaluator_config.name}...")
    resp = call_with_retries(
        EVALUATOR_SYSTEM, build_evaluator_prompt(original, translation),
        f"evaluación article {article_id}", config=evaluator_config, num_predict=1024,
    )
    usage = {
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
    }
    if resp.content is None:
        return None, usage

    evaluation = parse_evaluation(resp.content)
    if evaluation is None:
        log.warning("    [Stage 3] No se pudo parsear la evaluación")
        return None, usage

    evaluation = _adjust_evaluation_by_completeness(evaluation, original, translation, article_id)
    evaluation["model"] = evaluator_config.name
    evaluation["evaluated_at"] = datetime.now().isoformat()
    log.info(f"    [Stage 3] Score: {evaluation.get('overall_score', '?')}/100")
    return evaluation, usage


# ─── Pipeline completo por artículo ─────────────────────────────────────────

def process_article(entry: dict, article_id: int,
                    pipeline_config: dict) -> tuple[dict, dict] | None:
    """Procesa un artículo FEVER completo: Stage 1→2→3 + metadata."""
    text = entry["text"]
    titulo = entry["titulo"]
    language = detect_language(text)
    log.info(f"    Language detected: {language}")

    translation_models = pipeline_config["translation_models"]
    reviewer_config = pipeline_config["reviewer_model"]
    evaluator_config = pipeline_config["evaluator_model"]
    metadata_config = pipeline_config["metadata_model"]
    is_consensus = len(translation_models) > 1 or reviewer_config is not None

    token_usage = {
        "stage_1_translation": {},
        "stage_2_review":    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "stage_3_evaluation": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata":           {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # ── Stage 1 ──
    log.info(f"    [Stage 1] Generando {len(translation_models)} traducción(es)...")
    candidates, s1_usage, cf_models = stage_1_generate(text, article_id, translation_models)
    token_usage["stage_1_translation"] = s1_usage
    if not candidates:
        log.error("    Ninguna traducción candidata exitosa")
        return None

    # ── Stage 2 ──
    review = None
    if reviewer_config and candidates:
        review, s2_usage = stage_2_review(text, candidates, article_id, reviewer_config)
        token_usage["stage_2_review"] = s2_usage

    final_translation = review["consensus_text"] if review and review.get("consensus_text") else candidates[0]["text"]

    # ── Stage 3 ──
    evaluation = None
    if evaluator_config:
        evaluation, s3_usage = stage_3_evaluate(text, final_translation, article_id, evaluator_config)
        token_usage["stage_3_evaluation"] = s3_usage

    # ── Metadata ──
    log.info("    Metadata original...")
    meta_resp = call_with_retries(
        METADATA_SYSTEM_EN if language == "en" else METADATA_SYSTEM_ES,
        build_fever_metadata_prompt(text, titulo, lang=language),
        f"metadata article {article_id}", config=metadata_config, num_predict=512,
    )
    token_usage["metadata"]["prompt_tokens"] += meta_resp.prompt_tokens
    token_usage["metadata"]["completion_tokens"] += meta_resp.completion_tokens
    token_usage["metadata"]["total_tokens"] += meta_resp.total_tokens
    if meta_resp.content is None:
        return None
    metadata = parse_metadata(meta_resp.content)
    if metadata is None:
        # Fallback: construir metadata mínima desde el id
        log.warning("    Metadata falló — usando fallback")
        metadata = {
            "title": titulo,
            "keywords": [titulo],
            "classification": "other",
            "summary": text[:150],
        }

    log.info("    Metadata traducción...")
    meta_trans_resp = call_with_retries(
        METADATA_SYSTEM_ES if language == "en" else METADATA_SYSTEM_EN,
        build_fever_metadata_prompt(
            final_translation, titulo,
            lang="es" if language == "en" else "en",
        ),
        f"metadata traducida article {article_id}", config=metadata_config, num_predict=512,
    )
    token_usage["metadata"]["prompt_tokens"] += meta_trans_resp.prompt_tokens
    token_usage["metadata"]["completion_tokens"] += meta_trans_resp.completion_tokens
    token_usage["metadata"]["total_tokens"] += meta_trans_resp.total_tokens
    translated_metadata = parse_metadata(meta_trans_resp.content) if meta_trans_resp.content else None
    if translated_metadata is None:
        log.warning("    Metadata traducida falló — fallback")
        translated_metadata = metadata

    # ── Costos ──
    cost = {}
    total_cost = 0.0

    s1_cost = {}
    for model_name, u in token_usage["stage_1_translation"].items():
        c = calculate_cost(model_name, u["prompt_tokens"], u["completion_tokens"])
        s1_cost[model_name] = round(c, 6)
        total_cost += c
    cost["stage_1_translation"] = s1_cost

    s2 = token_usage["stage_2_review"]
    s2_cost = calculate_cost(reviewer_config.name if reviewer_config else "", s2["prompt_tokens"], s2["completion_tokens"])
    cost["stage_2_review"] = round(s2_cost, 6)
    total_cost += s2_cost

    s3 = token_usage["stage_3_evaluation"]
    s3_cost = calculate_cost(evaluator_config.name if evaluator_config else "", s3["prompt_tokens"], s3["completion_tokens"])
    cost["stage_3_evaluation"] = round(s3_cost, 6)
    total_cost += s3_cost

    meta = token_usage["metadata"]
    meta_cost = calculate_cost(metadata_config.name, meta["prompt_tokens"], meta["completion_tokens"])
    cost["metadata"] = round(meta_cost, 6)
    total_cost += meta_cost
    cost["total_usd"] = round(total_cost, 6)
    token_usage["cost_usd"] = cost

    total_tok = (
        sum(u.get("total_tokens", 0) for u in token_usage["stage_1_translation"].values())
        + token_usage["stage_2_review"]["total_tokens"]
        + token_usage["stage_3_evaluation"]["total_tokens"]
        + token_usage["metadata"]["total_tokens"]
    )
    log.info(f"    Tokens article {article_id}: {total_tok:,} | Costo: ${total_cost:.4f} USD")

    # ── Ensamblar (mismo formato que wikitext / europarl) ──
    timestamp = datetime.now().isoformat()
    translated_by = review["model"] if review else candidates[0]["model"]

    if language == "en":
        result = {
            "id": article_id,
            "source_language": "en",
            "english": {
                "title": metadata["title"],
                "keywords": metadata["keywords"],
                "summary": metadata["summary"],
                "classification": metadata["classification"],
                "text": text,
            },
            "spanish": {
                "title": translated_metadata["title"],
                "keywords": translated_metadata["keywords"],
                "summary": translated_metadata["summary"],
                "classification": translated_metadata["classification"],
                "text": final_translation,
                "translated_at": timestamp,
                "translated_by": translated_by,
            },
        }
    else:
        result = {
            "id": article_id,
            "source_language": "es",
            "english": {
                "title": translated_metadata["title"],
                "keywords": translated_metadata["keywords"],
                "summary": translated_metadata["summary"],
                "classification": translated_metadata["classification"],
                "text": final_translation,
                "translated_at": timestamp,
                "translated_by": translated_by,
            },
            "spanish": {
                "title": metadata["title"],
                "keywords": metadata["keywords"],
                "summary": metadata["summary"],
                "classification": metadata["classification"],
                "text": text,
            },
        }

    review_block = None
    if review:
        review_block = {
            "model": review["model"],
            "critique": review["critique"],
            "reviewed_at": review["reviewed_at"],
        }

    pipeline_entry = {
        "article_id": article_id,
        "fever_id": entry["fever_id"],
        "source_language": language,
        "mode": "consensus" if is_consensus else "single",
        "candidates": [
            {
                "model": c["model"],
                "translated_at": c["translated_at"],
                "char_count": len(c["text"]),
                "text": c["text"],
            }
            for c in candidates
        ],
        "review": review_block,
        "evaluation": evaluation,
        "token_usage": token_usage,
    }
    if cf_models:
        pipeline_entry["content_filter"] = cf_models

    return result, pipeline_entry


# ─── Procesamiento principal ─────────────────────────────────────────────────

def process_entries(entries: list[dict], pipeline_config: dict,
                    start_from: int = 1, limit: int = 0,
                    run_name: str = "fever_selected") -> tuple[list, list]:
    """Procesa la lista de entradas FEVER con persistencia incremental.

    El progreso se indexa por article_id (numérico) para ser consistente
    con los demás agentes (wikitext, europarl).
    """
    progress_path = OUTPUT_DIR / f".progress_{run_name}.json"
    progress = load_progress(progress_path)

    data = progress["data"]
    all_candidates = progress.get("candidates", [])
    completed_ids = set(progress.get("completed_ids", []))

    processed_count = 0
    total = len(entries)

    for i, entry in enumerate(entries, start=1):
        article_id = i
        fever_id = entry["fever_id"]

        if article_id < start_from:
            continue
        if limit > 0 and processed_count >= limit:
            log.info(f"Limit reached ({limit} articles)")
            break
        if article_id in completed_ids:
            log.info(f"  [{i}/{total}] #{article_id} {fever_id} — ya procesado, skipping")
            continue

        title = entry["titulo"][:60]
        words = entry["words"]
        log.info(f"  [{i}/{total}] #{article_id} — {title} ({words} palabras)")

        result = process_article(entry, article_id, pipeline_config)
        if result:
            article_data, pipeline_entry = result
            data.append(article_data)
            all_candidates.append(pipeline_entry)
            completed_ids.add(article_id)
            progress["completed_ids"] = sorted(completed_ids)
            progress["data"] = data
            progress["candidates"] = all_candidates
            save_progress(progress_path, progress)
            processed_count += 1
            log.info(f"  OK (total: {len(data)})")
        else:
            log.warning(f"  FAILED — skipped")

    # Borrar progress solo si completó normalmente
    if progress_path.exists():
        all_done = (limit == 0 or processed_count >= limit)
        if all_done:
            progress_path.unlink()
            log.info("Temporary progress removed (run completed)")
        else:
            log.info(f"Progress saved at: {progress_path}")

    return data, all_candidates


# ─── Entry point ─────────────────────────────────────────────────────────────

def run(input_file: Path, max_retries: int = MAX_RETRIES,
        start_from: int = 1, limit: int = 0, min_words: int = 10) -> None:
    import agente_wikitext
    agente_wikitext.MAX_RETRIES = max_retries

    if not input_file.exists():
        log.error(f"El archivo '{input_file}' no existe.")
        return
    if input_file.suffix not in {".jsonl", ".json"}:
        log.error(f"Extensión '{input_file.suffix}' no soportada. Se espera .jsonl")
        return

    entries = load_fever_entries(input_file, min_words=min_words)
    if not entries:
        log.error("No se encontraron entradas válidas.")
        return

    log.info(f"Artículos a procesar: {len(entries):,}")

    pipeline_config = load_pipeline_config()
    models_info = [m.name for m in pipeline_config["translation_models"]]
    reviewer = pipeline_config["reviewer_model"]
    evaluator = pipeline_config["evaluator_model"]

    log.info(f"Pipeline: {len(models_info)} traductor(es) {models_info}")
    if reviewer:
        log.info(f"  Revisor: {reviewer.name}")
    if evaluator:
        log.info(f"  Evaluador: {evaluator.name}")

    if not verify_all_models(pipeline_config):
        return

    translation_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if limit > 0:
        log.info(f"Límite: {limit} artículos")

    run_name = input_file.stem  # ej. "fever_selected"

    data, all_candidates = process_entries(
        entries, pipeline_config,
        start_from=start_from, limit=limit, run_name=run_name,
    )

    if not data:
        log.warning("Sin datos — no se genera JSON.")
        return

    date_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    file_size_mb = round(input_file.stat().st_size / (1024 * 1024), 2)

    # ── Archivo principal ──
    output = {
        "source_name": "fever",
        "url": "https://fever.ai/dataset/fever.html",
        "creation_date": "2018",
        "authors_name": ["Thorne et al."],
        "payload": [
            {
                "file_name": input_file.name,
                "file_size": f"{file_size_mb} MB",
                "file_format": "jsonl",
                "description": (
                    "FEVER (Fact Extraction and VERification) dataset. "
                    "Wikipedia article excerpts used for fact-checking research. "
                    "Each entry is an independent factual passage."
                ),
                "translation_date": translation_date,
                "llm_used": models_info,
                "data": data,
            }
        ],
    }

    output_path = OUTPUT_DIR / f"{run_name}_{date_stamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info(f"Guardado: {output_path}")

    # ── Archivo de pipeline ──
    if all_candidates:
        total_summary = {
            "articles_processed": len(all_candidates),
            "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "total_cost_usd": {
                "stage_1_translation": {},
                "stage_2_review": 0.0,
                "stage_3_evaluation": 0.0,
                "metadata": 0.0,
                "total": 0.0,
            },
        }
        for entry in all_candidates:
            tu = entry.get("token_usage", {})
            for stage_key in ["stage_2_review", "stage_3_evaluation", "metadata"]:
                stage = tu.get(stage_key, {})
                for tk in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    total_summary["total_tokens"][tk] += stage.get(tk, 0)
            for model_name, u in tu.get("stage_1_translation", {}).items():
                for tk in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    total_summary["total_tokens"][tk] += u.get(tk, 0)
            cost = tu.get("cost_usd", {})
            for model_name, c in cost.get("stage_1_translation", {}).items():
                total_summary["total_cost_usd"]["stage_1_translation"][model_name] = (
                    total_summary["total_cost_usd"]["stage_1_translation"].get(model_name, 0.0) + c
                )
            total_summary["total_cost_usd"]["stage_2_review"]    += cost.get("stage_2_review", 0.0)
            total_summary["total_cost_usd"]["stage_3_evaluation"] += cost.get("stage_3_evaluation", 0.0)
            total_summary["total_cost_usd"]["metadata"]           += cost.get("metadata", 0.0)
            total_summary["total_cost_usd"]["total"]              += cost.get("total_usd", 0.0)

        for model_name in total_summary["total_cost_usd"]["stage_1_translation"]:
            total_summary["total_cost_usd"]["stage_1_translation"][model_name] = round(
                total_summary["total_cost_usd"]["stage_1_translation"][model_name], 4)
        for key in ["stage_2_review", "stage_3_evaluation", "metadata", "total"]:
            total_summary["total_cost_usd"][key] = round(total_summary["total_cost_usd"][key], 4)

        pipeline_output = {
            "source_file": input_file.name,
            "generated_at": translation_date,
            "models_used": models_info,
            "reviewer": reviewer.name if reviewer else None,
            "evaluator": evaluator.name if evaluator else None,
            "summary": total_summary,
            "pipeline": all_candidates,
        }

        pipeline_path = OUTPUT_DIR / f"{run_name}_pipeline_{date_stamp}.json"
        with open(pipeline_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_output, f, ensure_ascii=False, indent=2)
        log.info(f"Pipeline: {pipeline_path}")
        log.info(f"Costo total: ${total_summary['total_cost_usd']['total']:.4f} USD")

    log.info(f"Artículos procesados: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traduce artículos FEVER (extractos Wikipedia en formato JSONL)",
    )
    parser.add_argument(
        "--input", "-i", type=Path,
        default=Path("data/FEVER/selected/fever_selected.jsonl"),
        help="Archivo .jsonl de entrada (default: data/FEVER/selected/fever_selected.jsonl)",
    )
    parser.add_argument(
        "--retries", "-r", type=int, default=MAX_RETRIES,
        help=f"Máximo de intentos por artículo (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--start-from", "-s", type=int, default=1,
        help="Empezar desde este número de artículo (default: 1)",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=0,
        help="Máximo de artículos a procesar (default: 0 = todos)",
    )
    parser.add_argument(
        "--min-words", type=int, default=10,
        help="Mínimo de palabras por artículo para incluirlo (default: 10)",
    )
    args = parser.parse_args()

    run(
        args.input,
        max_retries=args.retries,
        start_from=args.start_from,
        limit=args.limit,
        min_words=args.min_words,
    )
