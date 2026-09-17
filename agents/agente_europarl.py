"""
Agente de traducción para corpus Europarl (formato XML con CHAPTER/SPEAKER/P).

Trabaja con archivos .txt de la carpeta data/Europarl/en/ que usan tags XML:
    <CHAPTER ID="011">
    Título del debate
    <SPEAKER ID="047" NAME="President" LANGUAGE="EN">
    Texto del discurso...
    <P>
    Más párrafos...

Estrategia:
    - Cada archivo .txt = un debate independiente (un "artículo")
    - Acepta un directorio (procesa todos los .txt) o un archivo individual
    - Extrae texto de cada <SPEAKER>, traduce preservando los tags
    - Chunking por speakers cuando el debate excede el umbral de tokens
    - Los tags <CHAPTER>, <SPEAKER> y <P> se preservan intactos en la salida

Uso:
    # Directorio completo:
    python3 agente_europarl.py --input data/Europarl/en/

    # Archivo individual:
    python3 agente_europarl.py --input data/Europarl/en/ep-10-10-06-011.txt

    # Con límites:
    python3 agente_europarl.py --input data/Europarl/en/ --start-from 5 --limit 10

    # Solo archivos con contenido sustancial (>500 chars):
    python3 agente_europarl.py --input data/Europarl/en/ --min-chars 500
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from agente_wikitext import (
    MAX_RETRIES,
    MAX_CHARS_PER_CHUNK,
    ModelConfig,
    LLMResponse,
    load_pipeline_config,
    verify_all_models,
    read_file,
    detect_language,
    call_with_retries,
    clean_llm_response,
    clean_equal_signs,
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
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/Europarl")
EXTENSIONS = {".txt", ".raw"}


# ─── Prompts adaptados para Europarl ─────────────────────────────────────────

EUROPARL_TRANSLATION_SYSTEM = """Eres un traductor profesional especializado en traducción de actas parlamentarias.

Reglas:
1. Fidelidad semántica: NO inventes, NO omitas, NO agregues nada que no esté en el original.
2. Naturalidad: evita traducciones literales robóticas.
3. Terminología: conserva términos técnicos, nombres propios, referencias a resoluciones y directivas.
4. Estilo: respeta el tono formal parlamentario del original.
5. Estructura: conserva párrafos.
6. Calidad: español neutro profesional / inglés neutro.
7. PROHIBIDO: resumir, simplificar, reinterpretar o censurar.
8. PRESERVA INTACTOS todos los tags XML: <CHAPTER ...>, <SPEAKER ...>, <P>. No los traduzcas ni modifiques.
9. NO agregues formato markdown (negritas **, cursivas *, encabezados #).

Responde SOLO con la traducción, sin explicaciones ni prefijos."""


def build_europarl_translation_prompt(text: str) -> str:
    return f"""Traduce el siguiente texto parlamentario al español. Si ya está en español, tradúcelo al inglés.
Preserva INTACTOS todos los tags XML (<CHAPTER>, <SPEAKER>, <P>).
Responde SOLO con la traducción.

{text}"""


def build_europarl_metadata_prompt(text: str, lang: str = "en") -> str:
    """Genera prompt de metadata adaptado para debates parlamentarios."""
    # Strip tags for metadata analysis
    clean = re.sub(r'<[^>]+>', '', text).strip()
    # Use first 3000 chars for metadata
    sample = clean[:3000]

    if lang == "en":
        return f"""Analyze the following parliamentary debate and respond EXACTLY with this format (one line per field):

TITLE: [clean debate title]
KEYWORDS: [between 3 and 8 keywords separated by comma]
CLASSIFICATION: [main category: politics, legislation, economy, environment, human rights, agriculture, defense, trade, health, technology, or other appropriate]
SUMMARY: [informative summary of the debate, maximum 100 words]

Respond ONLY with these 4 fields.

=== DEBATE ===
{sample}"""
    return f"""Analiza el siguiente debate parlamentario y responde EXACTAMENTE con este formato (una línea por campo):

TITLE: [título limpio del debate]
KEYWORDS: [entre 3 y 8 keywords separadas por coma]
CLASSIFICATION: [categoría principal: politics, legislation, economy, environment, human rights, agriculture, defense, trade, health, technology, u otra apropiada]
SUMMARY: [resumen informativo del debate, máximo 100 palabras]

Responde SOLO con estos 4 campos.

=== DEBATE ===
{sample}"""


# ─── Parsing de estructura Europarl ──────────────────────────────────────────

def parse_europarl_file(text: str) -> dict:
    """Parsea un archivo Europarl XML y extrae su estructura.

    Returns:
        {
            "chapter_id": str,
            "chapter_title": str,
            "speakers": [{"tag": str, "name": str, "text": str}, ...],
            "full_text": str,  # texto completo con tags
        }
    """
    # Extraer CHAPTER
    chapter_match = re.search(r'<CHAPTER\s+ID="([^"]*)">', text)
    chapter_id = chapter_match.group(1) if chapter_match else ""

    # Extraer título (línea después de <CHAPTER>)
    title = ""
    chapter_end = text.find('>', text.find('<CHAPTER')) + 1 if '<CHAPTER' in text else 0
    next_tag = text.find('<', chapter_end)
    if next_tag > chapter_end:
        title = text[chapter_end:next_tag].strip()

    # Extraer speakers
    speakers = []
    speaker_pattern = re.compile(
        r'(<SPEAKER\s+[^>]*>)(.*?)(?=<SPEAKER|$)',
        re.DOTALL,
    )
    for m in speaker_pattern.finditer(text):
        tag = m.group(1)
        body = m.group(2).strip()
        name_match = re.search(r'NAME="([^"]*)"', tag)
        name = name_match.group(1) if name_match else ""
        speakers.append({"tag": tag, "name": name, "text": body})

    return {
        "chapter_id": chapter_id,
        "chapter_title": title,
        "speakers": speakers,
        "full_text": text.strip(),
    }


def get_text_only(text: str) -> str:
    """Extrae solo el texto sin tags XML, para estimación de tokens y detección de idioma."""
    clean = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\n{3,}', '\n\n', clean).strip()


# ─── Chunking por speakers ───────────────────────────────────────────────────

def split_by_speakers(text: str) -> list[str]:
    """Divide el texto por bloques <SPEAKER>, manteniendo los tags."""
    pattern = r'(?=<SPEAKER\s)'
    parts = re.split(pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [text]
    return parts


def group_speaker_chunks(sections: list[str],
                         max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Agrupa bloques de speakers consecutivos hasta max_chars."""
    chunks = []
    current = ""
    for section in sections:
        if len(section) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(section.strip())
        elif len(current) + len(section) > max_chars:
            chunks.append(current.strip())
            current = section
        else:
            current += "\n" + section if current else section
    if current:
        chunks.append(current.strip())
    return chunks


# ─── Stage 1: Traducción ────────────────────────────────────────────────────

def _translate_chunk(chunk: str, source_lang: str,
                     config: ModelConfig, chunk_label: str) -> tuple[str | None, dict, bool]:
    """Traduce un chunk preservando tags XML."""
    target_lang = "es" if source_lang == "en" else "en"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    resp = call_with_retries(
        EUROPARL_TRANSLATION_SYSTEM,
        build_europarl_translation_prompt(chunk),
        chunk_label,
        config=config,
    )
    if resp.content is None:
        return None, usage, resp.content_filter
    usage["prompt_tokens"] += resp.prompt_tokens
    usage["completion_tokens"] += resp.completion_tokens
    usage["total_tokens"] += resp.total_tokens

    text = resp.content.strip()
    # No aplicar clean_llm_response (borraría tags); solo limpiar prefijos comunes
    text = re.sub(r'^(Traducci[oó]n|Translation)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    # Validar idioma (solo sobre texto sin tags)
    detected = detect_language(get_text_only(text))
    if detected == target_lang:
        return text, usage, False

    for retry in range(1, TRANSLATION_LANGUAGE_RETRIES + 1):
        log.warning(f"      [{config.name}] {chunk_label} — output in '{detected}' instead of '{target_lang}', retry {retry}")
        resp = call_with_retries(
            EUROPARL_TRANSLATION_SYSTEM,
            build_translation_retry_prompt(chunk, target_lang),
            f"{chunk_label} (lang retry {retry})",
            config=config,
        )
        if resp.content is None:
            continue
        usage["prompt_tokens"] += resp.prompt_tokens
        usage["completion_tokens"] += resp.completion_tokens
        usage["total_tokens"] += resp.total_tokens
        text = resp.content.strip()
        detected = detect_language(get_text_only(text))
        if detected == target_lang:
            return text, usage, False

    return text, usage, False


def translate_debate_with_model(debate_text: str, article_id: int,
                                config: ModelConfig) -> tuple[str, list[str], dict, bool] | None:
    """Traduce un debate completo. Retorna (texto, chunks, usage, content_filter) o None."""
    text_only = get_text_only(debate_text)
    source_lang = detect_language(text_only)
    est_tokens = len(text_only) // 4
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if est_tokens <= 6000:
        log.info(f"      [{config.name}] Debate (~{est_tokens} tok) → 1 chunk")
        text, chunk_usage, cf = _translate_chunk(
            debate_text, source_lang, config,
            f"translation debate {article_id} [{config.name}]",
        )
        for k in usage:
            usage[k] += chunk_usage[k]
        if text is None:
            return None
        return text, [text], usage, cf

    # Debate largo: partir por speakers
    sections = split_by_speakers(debate_text)
    chunks = group_speaker_chunks(sections)
    log.info(f"      [{config.name}] Debate largo (~{est_tokens} tok) → {len(chunks)} chunks")

    translations = []
    any_cf = False
    for j, chunk in enumerate(chunks, 1):
        log.info(f"      [{config.name}] Chunk {j}/{len(chunks)} ({len(chunk):,} chars)")
        text, chunk_usage, cf = _translate_chunk(
            chunk, source_lang, config,
            f"translation debate {article_id} [{config.name}] chunk {j}/{len(chunks)}",
        )
        for k in usage:
            usage[k] += chunk_usage[k]
        if cf:
            any_cf = True
        if text is None:
            log.warning(f"      [{config.name}] Chunk {j} failed")
            return None
        translations.append(text)

    return "\n".join(translations), translations, usage, any_cf


# ─── Stage 1: Candidatos en paralelo ────────────────────────────────────────

def stage_1_generate(debate_text: str, article_id: int,
                     models: list[ModelConfig]) -> tuple[list[dict], dict, list[str]]:
    """Genera traducciones candidatas. Retorna (candidates, usage, cf_models)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates = []
    stage_usage = {}
    cf_models = []

    def _run(config):
        try:
            log.info(f"    [Stage 1] Traduciendo con {config.name}...")
            result = translate_debate_with_model(debate_text, article_id, config)
            return config, result, False
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            from agente_wikitext import _extract_error_detail
            detail, is_cf = _extract_error_detail(e)
            if is_cf:
                log.error(f"    [Stage 1] {config.name} — CONTENT FILTER: {detail[:200]}")
            else:
                log.error(f"    [Stage 1] {config.name} — error: {type(e).__name__}: {e}")
            return config, None, is_cf

    def _process(config, result, cf_flag):
        if result:
            text, chunks, usage, cf = result
            candidates.append({
                "model": config.name, "text": text, "chunks": chunks,
                "translated_at": datetime.now().isoformat(),
            })
            stage_usage[config.name] = usage
            if cf:
                cf_models.append(config.name)
            log.info(f"    [Stage 1] {config.name} — OK ({len(text):,} chars, {len(chunks)} chunks, {usage['total_tokens']} tok)")
        else:
            if cf_flag:
                cf_models.append(config.name)
            log.warning(f"    [Stage 1] {config.name} — failed")

    if len(models) <= 1:
        for config in models:
            config, result, cf = _run(config)
            _process(config, result, cf)
    else:
        log.info(f"    [Stage 1] Translating with {len(models)} models in parallel...")
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(_run, c): c for c in models}
            for future in as_completed(futures):
                config, result, cf = future.result()
                _process(config, result, cf)

    return candidates, stage_usage, cf_models


# ─── Stage 2: Revisión ──────────────────────────────────────────────────────

def _reviewer_critique(original: str, candidates: list[dict],
                       article_id: int, reviewer_config: ModelConfig) -> tuple[dict | None, dict]:
    """Análisis de candidatos por el reviewer."""
    # Use text-only for critique (less noise from tags)
    orig_clean = get_text_only(original)[:2000]
    parts = [f"=== TEXTO ORIGINAL (primeros 2000 chars) ===\n{orig_clean}\n"]
    for i, c in enumerate(candidates, 1):
        c_clean = get_text_only(c["text"])[:2000]
        parts.append(f"=== CANDIDATO {i} (modelo: {c['model']}, {len(c['text']):,} chars) ===\n{c_clean}\n")
    parts.append("Evalúa cada candidato y decide cuál usar como base. Responde solo con JSON.")

    resp = call_with_retries(
        REVIEWER_CRITIQUE_SYSTEM, "\n".join(parts),
        f"critique debate {article_id}", config=reviewer_config, num_predict=1024,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    if resp.content is None:
        return None, usage

    json_match = re.search(r'\{.*\}', resp.content, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0)), usage
        except json.JSONDecodeError:
            return {"raw": resp.content[:500]}, usage
    return {"raw": resp.content[:500]}, usage


def _reviewer_merge_section(original_section: str, candidates: list[dict],
                            critique: dict, article_id: int, section_idx: int,
                            total_sections: int, reviewer_config: ModelConfig) -> tuple[str | None, dict]:
    """Genera traducción final para una sección."""
    best_model = None
    if isinstance(critique, dict) and "candidates_analysis" in critique:
        analyses = critique["candidates_analysis"]
        best = max(analyses, key=lambda x: x.get("score", 0))
        best_model = best.get("model", "")

    parts = [f"=== SECCIÓN ORIGINAL ({section_idx}/{total_sections}) ===\n{original_section}\n"]
    for i, c in enumerate(candidates, 1):
        label = "(MEJOR)" if best_model and best_model in c["model"] else ""
        parts.append(f"=== CANDIDATO {i} {label} ({c['model']}) ===\n{c['section_text']}\n")
    parts.append(
        f"Decisión: {critique.get('decision', 'Usar el mejor candidato como base.')}\n"
        "Genera la traducción final mejorada. Preserva INTACTOS todos los tags XML (<CHAPTER>, <SPEAKER>, <P>). "
        "Responde SOLO con el texto traducido."
    )

    resp = call_with_retries(
        REVIEWER_MERGE_SYSTEM, "\n".join(parts),
        f"merge debate {article_id} sección {section_idx}/{total_sections}",
        config=reviewer_config,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    return resp.content, usage


def _split_text_aligned(original: str, candidates: list[dict]) -> list[dict]:
    """Divide original y candidatos en secciones alineadas por speakers."""
    sections_orig = split_by_speakers(original)
    chunks_orig = group_speaker_chunks(sections_orig)

    if len(chunks_orig) <= 1:
        return [{"original": original,
                 "candidates": [{"model": c["model"], "section_text": c["text"]} for c in candidates]}]

    n_chunks = len(chunks_orig)
    all_have_chunks = all("chunks" in c and len(c["chunks"]) == n_chunks for c in candidates)

    sections = []
    for j, chunk_orig in enumerate(chunks_orig):
        section = {"original": chunk_orig, "candidates": []}
        for c in candidates:
            if all_have_chunks:
                section_text = c["chunks"][j]
            else:
                ratio_start = sum(len(ch) for ch in chunks_orig[:j]) / max(len(original), 1)
                ratio_end = sum(len(ch) for ch in chunks_orig[:j + 1]) / max(len(original), 1)
                start = int(len(c["text"]) * ratio_start)
                end = int(len(c["text"]) * ratio_end)
                section_text = c["text"][start:end]
            section["candidates"].append({"model": c["model"], "section_text": section_text})
        sections.append(section)

    if not all_have_chunks:
        log.warning("    [Stage 2] Chunks no alineados — fallback proporcional")
    return sections


def stage_2_review(original: str, candidates: list[dict],
                   article_id: int, reviewer_config: ModelConfig) -> tuple[dict | None, dict]:
    """Revisión por consenso."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not candidates:
        return None, usage

    text_only = get_text_only(original)
    est_tokens = len(text_only) // 4
    log.info(f"    [Stage 2] Revisando con {reviewer_config.name}...")

    critique, cr_usage = _reviewer_critique(original, candidates, article_id, reviewer_config)
    for k in usage:
        usage[k] += cr_usage[k]
    if critique is None:
        log.warning("    [Stage 2] Critique falló — fallback a candidato 1")
        return None, usage
    log.info("    [Stage 2] Critique completada")

    if est_tokens <= 6000:
        sections = [{"original": original,
                     "candidates": [{"model": c["model"], "section_text": c["text"]} for c in candidates]}]
    else:
        sections = _split_text_aligned(original, candidates)
        log.info(f"    [Stage 2] Debate largo → {len(sections)} secciones")

    merged_parts = []
    for j, section in enumerate(sections, 1):
        if len(sections) > 1:
            log.info(f"    [Stage 2] Merging sección {j}/{len(sections)}")
        result, merge_usage = _reviewer_merge_section(
            section["original"], section["candidates"],
            critique, article_id, j, len(sections), reviewer_config,
        )
        for k in usage:
            usage[k] += merge_usage[k]
        if result is None:
            log.warning(f"    [Stage 2] Sección {j} falló")
            return None, usage
        merged_parts.append(result.strip())

    consensus = "\n".join(merged_parts)
    review = {
        "critique": critique, "consensus_text": consensus,
        "model": reviewer_config.name, "reviewed_at": datetime.now().isoformat(),
    }
    log.info(f"    [Stage 2] Revisión completada ({len(consensus):,} chars, {usage['total_tokens']} tok)")
    return review, usage


# ─── Stage 3: Evaluación ────────────────────────────────────────────────────

def stage_3_evaluate(original: str, translation: str,
                     article_id: int, evaluator_config: ModelConfig) -> tuple[dict | None, dict]:
    """Evalúa la traducción comparada con el original (texto sin tags)."""
    from agente_wikitext import _adjust_evaluation_by_completeness

    orig_clean = get_text_only(original)
    trans_clean = get_text_only(translation)
    log.info(f"    [Stage 3] Evaluando con {evaluator_config.name}...")

    resp = call_with_retries(
        EVALUATOR_SYSTEM, build_evaluator_prompt(orig_clean, trans_clean),
        f"evaluación debate {article_id}", config=evaluator_config, num_predict=1024,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    if resp.content is None:
        return None, usage

    evaluation = parse_evaluation(resp.content)
    if evaluation is None:
        log.warning("    [Stage 3] No se pudo parsear la evaluación")
        return None, usage

    evaluation = _adjust_evaluation_by_completeness(evaluation, orig_clean, trans_clean, article_id)
    evaluation["model"] = evaluator_config.name
    evaluation["evaluated_at"] = datetime.now().isoformat()
    log.info(f"    [Stage 3] Score: {evaluation.get('overall_score', '?')}/100")
    return evaluation, usage


# ─── Pipeline completo por debate ────────────────────────────────────────────

def process_debate(debate_text: str, article_id: int,
                   pipeline_config: dict) -> tuple[dict, dict] | None:
    """Procesa un debate Europarl completo: Stage 1→2→3 + metadata."""
    text_only = get_text_only(debate_text)
    language = detect_language(text_only)
    log.info(f"    Language detected: {language}")

    translation_models = pipeline_config["translation_models"]
    reviewer_config = pipeline_config["reviewer_model"]
    evaluator_config = pipeline_config["evaluator_model"]
    metadata_config = pipeline_config["metadata_model"]
    is_consensus = len(translation_models) > 1 or reviewer_config is not None

    token_usage = {
        "stage_1_translation": {},
        "stage_2_review": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "stage_3_evaluation": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # ── Stage 1 ──
    log.info(f"    [Stage 1] Generando {len(translation_models)} traducción(es)...")
    candidates, s1_usage, cf_models = stage_1_generate(debate_text, article_id, translation_models)
    token_usage["stage_1_translation"] = s1_usage
    if not candidates:
        log.error("    Ninguna traducción candidata exitosa")
        return None

    # ── Stage 2 ──
    review = None
    if reviewer_config and candidates:
        review, s2_usage = stage_2_review(debate_text, candidates, article_id, reviewer_config)
        token_usage["stage_2_review"] = s2_usage

    final_translation = review["consensus_text"] if review and review.get("consensus_text") else candidates[0]["text"]

    # ── Stage 3 ──
    evaluation = None
    if evaluator_config:
        evaluation, s3_usage = stage_3_evaluate(debate_text, final_translation, article_id, evaluator_config)
        token_usage["stage_3_evaluation"] = s3_usage

    # ── Metadata ──
    log.info("    Metadata original...")
    meta_resp = call_with_retries(
        METADATA_SYSTEM_EN if language == "en" else METADATA_SYSTEM_ES,
        build_europarl_metadata_prompt(debate_text, lang=language),
        f"metadata debate {article_id}", config=metadata_config, num_predict=1024,
    )
    token_usage["metadata"]["prompt_tokens"] += meta_resp.prompt_tokens
    token_usage["metadata"]["completion_tokens"] += meta_resp.completion_tokens
    token_usage["metadata"]["total_tokens"] += meta_resp.total_tokens
    if meta_resp.content is None:
        return None
    metadata = parse_metadata(meta_resp.content)
    if metadata is None:
        return None

    log.info("    Metadata traducción...")
    meta_trans_resp = call_with_retries(
        METADATA_SYSTEM_ES if language == "en" else METADATA_SYSTEM_EN,
        build_europarl_metadata_prompt(final_translation, lang="es" if language == "en" else "en"),
        f"metadata traducida debate {article_id}", config=metadata_config, num_predict=1024,
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

    total_tok = sum(
        u.get("total_tokens", 0) if isinstance(u, dict) and "total_tokens" in u
        else sum(v.get("total_tokens", 0) for v in u.values()) if isinstance(u, dict) else 0
        for u in [token_usage["stage_1_translation"], token_usage["stage_2_review"],
                  token_usage["stage_3_evaluation"], token_usage["metadata"]]
    )
    log.info(f"    Tokens totales debate {article_id}: {total_tok:,} | Costo: ${total_cost:.4f} USD")

    # ── Ensamblar (mismo formato que agente_wikitext) ──
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
                "text": debate_text,
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
                "text": debate_text,
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


# ─── Recolección de archivos ─────────────────────────────────────────────────

def collect_files(input_path: Path, min_chars: int = 0) -> list[Path]:
    """Recolecta archivos .txt del directorio o retorna archivo individual."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
        if min_chars > 0:
            filtered = []
            for f in files:
                if f.stat().st_size >= min_chars:
                    filtered.append(f)
            log.info(f"Filtrado: {len(filtered)}/{len(files)} archivos (>= {min_chars} chars)")
            return filtered
        return files

    log.error(f"'{input_path}' no es un archivo ni directorio válido.")
    return []


# ─── Procesamiento principal ─────────────────────────────────────────────────

def process_files(files: list[Path], pipeline_config: dict,
                  id_offset: int = 0, start_from: int = 1,
                  limit: int = 0, run_name: str = "europarl") -> tuple[list, list]:
    """Procesa una lista de archivos Europarl con persistencia incremental."""
    progress_path = OUTPUT_DIR / f".progress_{run_name}.json"
    progress = load_progress(progress_path)

    data = progress["data"]
    all_candidates = progress.get("candidates", [])
    completed_ids = set(progress["completed_ids"])

    processed_count = 0
    for i, file_path in enumerate(files, start=1):
        article_id = id_offset + i

        if article_id < start_from:
            continue
        if limit > 0 and processed_count >= limit:
            log.info(f"Limit reached ({limit} debates)")
            break
        if article_id in completed_ids:
            log.info(f"  [{i}/{len(files)}] ID {article_id} — already processed, skipping")
            continue

        text = read_file(file_path)
        if not text:
            continue

        text_only = get_text_only(text)
        if len(text_only) < 50:
            log.info(f"  [{i}/{len(files)}] ID {article_id} — {file_path.name}: too short ({len(text_only)} chars), skipping")
            continue

        parsed = parse_europarl_file(text)
        title = parsed["chapter_title"][:60] or file_path.stem
        log.info(f"  [{i}/{len(files)}] ID {article_id} — {file_path.name}: {title} ({len(text_only):,} chars, {len(parsed['speakers'])} speakers)")

        result = process_debate(text, article_id, pipeline_config)
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

    # Solo borramos progress si terminó normalmente (no se interrumpió)
    if progress_path.exists():
        all_done = (limit == 0 or processed_count >= limit)
        if all_done:
            progress_path.unlink()
            log.info("Temporary progress removed (run completed)")
        else:
            log.info(f"Progress saved at: {progress_path}")

    return data, all_candidates


# ─── Entry point ─────────────────────────────────────────────────────────────

def run(input_path: Path, max_retries: int = MAX_RETRIES,
        start_from: int = 1, limit: int = 0, min_chars: int = 0) -> None:
    import agente_wikitext
    agente_wikitext.MAX_RETRIES = max_retries

    files = collect_files(input_path, min_chars=min_chars)
    if not files:
        log.error("No files found.")
        return

    log.info(f"Files to process: {len(files)}")

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
        log.info(f"Limite: {limit} debates")

    # Nombre para progress y archivos de salida
    if input_path.is_file():
        run_name = input_path.stem
    else:
        run_name = input_path.name

    data, all_candidates = process_files(
        files, pipeline_config, start_from=start_from, limit=limit,
        run_name=run_name,
    )

    if not data:
        log.warning("Sin datos -- no se genera JSON.")
        return

    date_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Calcular tamaño total de archivos procesados
    total_size = sum(f.stat().st_size for f in files if f.exists())
    total_size_mb = round(total_size / (1024 * 1024), 2)

    # ── Archivo principal (mismo formato que wikitext) ──
    output = {
        "source_name": "europarl",
        "url": "https://www.statmt.org/europarl/",
        "creation_date": "1996-2011",
        "authors_name": ["Philipp Koehn"],
        "payload": [
            {
                "file_name": run_name,
                "file_size": f"{total_size_mb} MB",
                "file_format": "txt",
                "description": "Proceedings of the European Parliament. "
                               "Each debate (CHAPTER) is translated independently "
                               "preserving XML tags (CHAPTER, SPEAKER, P).",
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

    # ── Archivo de pipeline (mismo formato que wikitext) ──
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
            total_summary["total_cost_usd"]["stage_2_review"] += cost.get("stage_2_review", 0.0)
            total_summary["total_cost_usd"]["stage_3_evaluation"] += cost.get("stage_3_evaluation", 0.0)
            total_summary["total_cost_usd"]["metadata"] += cost.get("metadata", 0.0)
            total_summary["total_cost_usd"]["total"] += cost.get("total_usd", 0.0)

        for model_name in total_summary["total_cost_usd"]["stage_1_translation"]:
            total_summary["total_cost_usd"]["stage_1_translation"][model_name] = round(
                total_summary["total_cost_usd"]["stage_1_translation"][model_name], 4)
        for key in ["stage_2_review", "stage_3_evaluation", "metadata", "total"]:
            total_summary["total_cost_usd"][key] = round(total_summary["total_cost_usd"][key], 4)

        pipeline_output = {
            "source_file": run_name,
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

    log.info(f"Debates procesados: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traduce debates Europarl (formato XML con CHAPTER/SPEAKER/P)",
    )
    parser.add_argument(
        "--input", "-i", type=Path,
        default=Path("data/Europarl/en/"),
        help="Archivo .txt o directorio con archivos Europarl",
    )
    parser.add_argument("--id-offset", type=int, default=0, help="Offset para IDs")
    parser.add_argument("--retries", "-r", type=int, default=MAX_RETRIES,
                        help=f"Máximo de intentos por debate (default: {MAX_RETRIES})")
    parser.add_argument("--start-from", "-s", type=int, default=1,
                        help="Empezar desde este número de debate (default: 1)")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Máximo de debates a procesar (default: 0 = todos)")
    parser.add_argument("--min-chars", type=int, default=0,
                        help="Tamaño mínimo de archivo en bytes (default: 0 = todos)")
    args = parser.parse_args()

    run(args.input, max_retries=args.retries, start_from=args.start_from,
        limit=args.limit, min_chars=args.min_chars)
