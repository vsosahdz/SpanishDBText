import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ─── Configuración base ─────────────────────────────────────────────────────

# Legacy (backward compat)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

# Azure OpenAI (GPT-4.1, GPT-4o, etc.)
AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")

# Azure AI Foundry - Modelos serverless
# Cada modelo serverless tiene su propio endpoint y key
FOUNDRY_MISTRAL_ENDPOINT = os.getenv("FOUNDRY_MISTRAL_ENDPOINT", "")
FOUNDRY_MISTRAL_KEY      = os.getenv("FOUNDRY_MISTRAL_KEY", "")
FOUNDRY_CLAUDE_ENDPOINT  = os.getenv("FOUNDRY_CLAUDE_ENDPOINT", "")
FOUNDRY_CLAUDE_KEY       = os.getenv("FOUNDRY_CLAUDE_KEY", "")
FOUNDRY_DEEPSEEK_ENDPOINT = os.getenv("FOUNDRY_DEEPSEEK_ENDPOINT", "")
FOUNDRY_DEEPSEEK_KEY      = os.getenv("FOUNDRY_DEEPSEEK_KEY", "")

# Parámetros compartidos
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_CTX     = int(os.getenv("NUM_CTX", "32768"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "8192"))

OUTPUT_DIR  = Path("output/wikitext")
EXTENSIONS = {".tokens", ".raw"}
MAX_RETRIES = 3
TIMEOUT     = 600
MAX_CHARS_PER_CHUNK = 12000

# Precios por 1M tokens (USD) — actualizar si cambian
# Formato: substring del nombre del modelo → (input_price, output_price)
# Nota: los deployment names de Azure pueden no tener guiones/puntos (ej. "gpt41mini")
MODEL_PRICING = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt41mini":    (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "gpt41":        (2.00, 8.00),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt4omini":    (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt4o":        (2.50, 10.00),
    "mistral":      (0.50, 1.50),
    "deepseek":     (0.50, 1.80),
    "ollama":       (0.00, 0.00),
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calcula costo USD basado en nombre del modelo y tokens usados."""
    name_lower = model_name.lower()
    # Buscar precio: primero el más específico (gpt-4.1-mini antes que gpt-4.1)
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if key in name_lower:
            input_price, output_price = MODEL_PRICING[key]
            return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── ModelConfig ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    provider: str           # "ollama" | "azure" | "foundry"
    model: str
    ollama_base_url: str = "http://localhost:11434"
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-12-01-preview"
    azure_deployment: str = ""
    foundry_endpoint: str = ""  # Para modelos serverless (Mistral, DeepSeek)
    foundry_key: str = ""
    temperature: float = 0.2
    num_ctx: int = 32768
    num_predict: int = 8192

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_model_spec(spec: str) -> ModelConfig:
    """Parsea 'provider:model' → ModelConfig.

    Formatos soportados:
      - ollama:qwen3.5:9b          → Ollama local
      - azure:gpt-4.1-mini         → Azure OpenAI (GPT)
      - foundry:mistral-large      → Azure AI Foundry serverless (Mistral, DeepSeek)
    """
    parts = spec.split(":", 1)
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else ""

    if provider == "azure":
        return ModelConfig(
            provider="azure",
            model=model,
            azure_endpoint=AZURE_ENDPOINT,
            azure_api_key=AZURE_API_KEY,
            azure_api_version=AZURE_API_VERSION,
            azure_deployment=model,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )
    elif provider == "foundry":
        # Determinar endpoint y key según el modelo
        endpoint, key = _get_foundry_credentials(model)
        return ModelConfig(
            provider="foundry",
            model=model,
            foundry_endpoint=endpoint,
            foundry_key=key,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )
    else:
        return ModelConfig(
            provider="ollama",
            model=model,
            ollama_base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )


def _get_foundry_credentials(model: str) -> tuple[str, str]:
    """Obtiene endpoint y key para un modelo serverless de Azure AI Foundry."""
    model_lower = model.lower()
    if "mistral" in model_lower:
        return FOUNDRY_MISTRAL_ENDPOINT, FOUNDRY_MISTRAL_KEY
    elif "claude" in model_lower:
        return FOUNDRY_CLAUDE_ENDPOINT, FOUNDRY_CLAUDE_KEY
    elif "llama" in model_lower:
        return (
            os.getenv("FOUNDRY_LLAMA_ENDPOINT", ""),
            os.getenv("FOUNDRY_LLAMA_KEY", ""),
        )
    elif "deepseek" in model_lower:
        return FOUNDRY_DEEPSEEK_ENDPOINT, FOUNDRY_DEEPSEEK_KEY
    else:
        # Fallback: usa variables genéricas con el nombre del modelo
        env_prefix = model.upper().replace("-", "_").replace(".", "_")
        return (
            os.getenv(f"FOUNDRY_{env_prefix}_ENDPOINT", ""),
            os.getenv(f"FOUNDRY_{env_prefix}_KEY", ""),
        )


def load_pipeline_config() -> dict:
    """Carga configuración del pipeline desde .env."""
    translation_models = []
    for i in range(1, 5):
        spec = os.getenv(f"TRANSLATION_MODEL_{i}")
        if spec:
            translation_models.append(parse_model_spec(spec))

    # Backward compat: sin TRANSLATION_MODEL_* usa legacy
    if not translation_models:
        if LLM_PROVIDER == "azure":
            translation_models.append(ModelConfig(
                provider="azure", model=AZURE_DEPLOYMENT,
                azure_endpoint=AZURE_ENDPOINT, azure_api_key=AZURE_API_KEY,
                azure_api_version=AZURE_API_VERSION, azure_deployment=AZURE_DEPLOYMENT,
                temperature=TEMPERATURE, num_ctx=NUM_CTX, num_predict=NUM_PREDICT,
            ))
        else:
            translation_models.append(ModelConfig(
                provider="ollama", model=OLLAMA_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE, num_ctx=NUM_CTX, num_predict=NUM_PREDICT,
            ))

    reviewer_spec = os.getenv("REVIEWER_MODEL")
    evaluator_spec = os.getenv("EVALUATOR_MODEL")

    return {
        "translation_models": translation_models,
        "reviewer_model": parse_model_spec(reviewer_spec) if reviewer_spec else None,
        "evaluator_model": parse_model_spec(evaluator_spec) if evaluator_spec else None,
        "metadata_model": translation_models[0],
    }


# ─── Verificación de modelos ────────────────────────────────────────────────

def verify_model(config: ModelConfig) -> bool:
    if config.provider == "azure":
        if not config.azure_endpoint or not config.azure_api_key:
            log.error(f"Azure: AZURE_ENDPOINT y AZURE_API_KEY requeridos para {config.name}")
            return False
        log.info(f"Verificado: {config.name}")
        return True

    if config.provider == "foundry":
        if not config.foundry_endpoint or not config.foundry_key:
            log.error(f"Foundry: endpoint y key requeridos para {config.name}")
            return False
        log.info(f"Verificado: {config.name}")
        return True

    try:
        resp = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        if config.model not in available:
            log.error(f"Model '{config.model}' not found. Available: {available}")
            return False
        log.info(f"Verificado: {config.name}")
        return True
    except httpx.ConnectError:
        log.error(f"No se pudo conectar a Ollama en {config.ollama_base_url}")
        return False
    except Exception as e:
        log.error(f"Error verificando {config.name}: {e}")
        return False


def verify_all_models(pipeline_config: dict) -> bool:
    seen = set()
    all_configs = list(pipeline_config["translation_models"])
    if pipeline_config["reviewer_model"]:
        all_configs.append(pipeline_config["reviewer_model"])
    if pipeline_config["evaluator_model"]:
        all_configs.append(pipeline_config["evaluator_model"])

    for config in all_configs:
        if config.name not in seen:
            seen.add(config.name)
            if not verify_model(config):
                return False
    return True


# ─── Llamadas al LLM ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Respuesta de un LLM con contenido y uso de tokens."""
    content: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    content_filter: bool = False


def call_llm(system: str, user: str, config: ModelConfig,
               num_predict: int | None = None) -> LLMResponse:
    np = num_predict if num_predict is not None else config.num_predict
    if config.provider == "azure":
        return _call_azure(system, user, config, np)
    elif config.provider == "foundry":
        return _call_foundry(system, user, config, np)
    return _call_ollama(system, user, config, np)


def _call_ollama(system: str, user: str, config: ModelConfig,
                   num_predict: int) -> LLMResponse:
    resp = httpx.post(
        f"{config.ollama_base_url}/api/chat",
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": config.temperature,
                "num_ctx": config.num_ctx,
                "num_predict": num_predict,
            },
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("done_reason") == "length":
        log.warning(f"[{config.name}] Respuesta truncada (num_predict={num_predict})")

    content = data.get("message", {}).get("content", "").strip()
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    return LLMResponse(
        content=content if content else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _call_azure(system: str, user: str, config: ModelConfig,
                  num_predict: int) -> LLMResponse:
    url = f"{config.azure_endpoint}/openai/deployments/{config.azure_deployment}/chat/completions?api-version={config.azure_api_version}"
    resp = httpx.post(
        url,
        headers={"api-key": config.azure_api_key, "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": config.temperature,
            "max_tokens": num_predict,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()

    if choice.get("finish_reason") == "length":
        log.warning(f"[{config.name}] Respuesta truncada (max_tokens={num_predict})")

    usage = data.get("usage", {})
    return LLMResponse(
        content=content if content else None,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _call_foundry(system: str, user: str, config: ModelConfig,
                    num_predict: int) -> LLMResponse:
    """Llama a modelos serverless de Azure AI Foundry (Mistral, DeepSeek, etc.)."""
    # Construir URL: si el endpoint ya tiene /chat/completions, usarlo tal cual
    endpoint = config.foundry_endpoint.rstrip("/")
    if "/chat/completions" in endpoint:
        url = endpoint
    else:
        url = f"{endpoint}/chat/completions"

    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {config.foundry_key}", "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": config.temperature,
            "max_tokens": num_predict,
            "model": config.model,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()

    if choice.get("finish_reason") == "length":
        log.warning(f"[{config.name}] Respuesta truncada (max_tokens={num_predict})")

    usage = data.get("usage", {})
    return LLMResponse(
        content=content if content else None,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _extract_error_detail(exc: Exception) -> tuple[str, bool]:
    """Extracts error detail from an HTTP exception.
    Returns (detail_message, is_content_filter)."""
    is_content_filter = False
    detail = ""
    if hasattr(exc, "response") and exc.response is not None:
        try:
            body = exc.response.json()
            error = body.get("error", {})
            detail = error.get("message", "") or error.get("detail", "") or str(body)
            code = error.get("code", "").lower()
            inner = error.get("innererror", {})
            inner_code = inner.get("code", "").lower()
            if "content_filter" in code or "content_filter" in inner_code or "content_filter" in detail.lower():
                is_content_filter = True
        except Exception:
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = ""
    return detail, is_content_filter


def call_with_retries(system: str, user: str, description: str,
                          config: ModelConfig,
                          num_predict: int | None = None) -> LLMResponse:
    """Retorna LLMResponse (content puede ser None si falla)."""
    content_filter_detected = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = call_llm(system, user, config, num_predict)
            if resp.content:
                if attempt > 1:
                    log.info(f"Success on attempt {attempt} for {description}")
                return resp
            log.warning(f"Empty response — {description} (attempt {attempt}/{MAX_RETRIES})")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            detail, is_cf = _extract_error_detail(e)
            if is_cf:
                content_filter_detected = True
                log.error(f"CONTENT FILTER — {description} (attempt {attempt}/{MAX_RETRIES}): {detail[:200]}")
            elif detail:
                log.error(f"Error — {description} (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e} | Detail: {detail[:200]}")
            else:
                log.error(f"Error — {description} (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
    log.error(f"{description} failed after {MAX_RETRIES} attempts")
    resp = LLMResponse(content=None)
    resp.content_filter = content_filter_detected
    return resp


# ─── Prompts: Traducción ────────────────────────────────────────────────────

TRANSLATION_SYSTEM = """Eres un traductor profesional especializado en traducción editorial y técnica.

Reglas:
1. Fidelidad semántica: NO inventes, NO omitas, NO agregues nada que no esté en el original.
2. Naturalidad: evita traducciones literales robóticas.
3. Terminología: conserva términos técnicos correctos. No traduzcas nombres propios.
4. Estilo: respeta el tono del original.
5. Estructura: conserva párrafos y puntuación importante.
6. Calidad: español neutro profesional / inglés neutro.
7. PROHIBIDO: resumir, simplificar, reinterpretar o censurar.
8. Limpia signos = de títulos y subtítulos.
9. Limpia tokens <unk>.
10. NO agregues formato markdown (negritas **, cursivas *, encabezados #) a menos que esté en el original.

Responde SOLO con la traducción, sin explicaciones ni prefijos."""

METADATA_SYSTEM_ES = """Eres un sistema experto en análisis y clasificación de textos.
Responde SIEMPRE en el formato exacto indicado, sin explicaciones adicionales."""

METADATA_SYSTEM_EN = """You are an expert text analysis and classification system.
ALWAYS respond in the exact format indicated, with no additional explanations."""


def build_translation_prompt(article: str) -> str:
    return f"""Traduce el siguiente artículo al español. Si ya está en español, tradúcelo al inglés.
Responde SOLO con la traducción, sin explicaciones ni prefijos.

{article}"""


def build_translation_retry_prompt(article: str, target_lang: str = "es") -> str:
    if target_lang == "es":
        return f"""IMPORTANTE: Tu respuesta anterior fue en inglés. DEBES traducir el siguiente texto al español.
No devuelvas el texto original. Responde ÚNICAMENTE con la traducción al español.

{article}"""
    return f"""IMPORTANT: Your previous response was in Spanish. You MUST translate the following text to English.
Do not return the original text. Respond ONLY with the English translation.

{article}"""


TRANSLATION_LANGUAGE_RETRIES = 2


def build_metadata_prompt(article: str, lang: str = "es") -> str:
    if lang == "en":
        return f"""Analyze the following article and respond EXACTLY with this format (one line per field):

TITLE: [clean article title, without = signs]
KEYWORDS: [between 3 and 8 keywords separated by comma]
CLASSIFICATION: [main category: video game, novel, biography, science, history, technology, politics, medicine, sports, entertainment, or other appropriate]
SUMMARY: [informative summary of the article, maximum 100 words]

Respond ONLY with these 4 fields.

=== ARTICLE ===
{article}"""
    return f"""Analiza el siguiente artículo y responde EXACTAMENTE con este formato (una línea por campo):

TITLE: [título limpio del artículo, sin signos =]
KEYWORDS: [entre 3 y 8 keywords separadas por coma]
CLASSIFICATION: [categoría principal: video game, novel, biography, science, history, technology, politics, medicine, sports, entertainment, u otra apropiada]
SUMMARY: [resumen informativo del artículo, máximo 100 palabras]

Responde SOLO con estos 4 campos.

=== ARTÍCULO ===
{article}"""


# ─── Prompts: Revisor (Stage 2) ─────────────────────────────────────────────

REVIEWER_SYSTEM = """Eres un editor y revisor profesional de traducciones.

Tu tarea es:
1. Evaluar cada traducción candidata individualmente (fortalezas, debilidades, puntuación).
2. Decidir qué candidato usar como base y qué mejoras aplicar.
3. Generar UNA traducción final mejorada combinando lo mejor de todos los candidatos.

Reglas:
- Fidelidad al original: NO inventes, NO omitas contenido.
- Naturalidad: la traducción final debe sonar fluida y profesional.
- Completitud: incluye TODO el contenido del original.
- Terminología: conserva términos técnicos y nombres propios.

Responde EXACTAMENTE con este formato (JSON + texto):

ANALYSIS:
[JSON con el análisis de cada candidato]
{"candidates_analysis": [{"model": "nombre", "strengths": ["..."], "weaknesses": ["..."], "score": 0-100}], "decision": "explicación breve de la decisión"}

FINAL TRANSLATION:
[La traducción final mejorada por consenso, texto completo]"""


def build_reviewer_prompt(original: str, candidates: list[dict]) -> str:
    parts = [f"=== TEXTO ORIGINAL ===\n{original}\n"]
    for i, c in enumerate(candidates, 1):
        parts.append(f"=== TRADUCCIÓN CANDIDATA {i} (modelo: {c['model']}) ===\n{c['text']}\n")
    parts.append(
        "Evalúa cada candidato (fortalezas, debilidades, puntuación 0-100), "
        "decide cuál usar como base y genera la traducción final mejorada. "
        "Responde con ANALYSIS: (JSON) y FINAL TRANSLATION: como se indica."
    )
    return "\n".join(parts)


# ─── Prompts: Evaluador (Stage 3) ───────────────────────────────────────────

EVALUATOR_SYSTEM = """Eres un evaluador ESTRICTO y CRÍTICO experto en calidad de traducción.

Evalúa la traducción proporcionada comparándola con el original. Sé exigente — una puntuación perfecta (100) debe ser extremadamente rara.

GUÍA IMPORTANTE DE PUNTUACIÓN:
- 95-100: IMPECABLE, calidad profesional, publicable sin ediciones. Debe ser raro.
- 90-94: EXCELENTE con solo problemas triviales.
- 85-89: MUY BUENA con problemas menores (1-2 errores pequeños).
- 80-84: BUENA pero con problemas notables que necesitan corrección.
- 70-79: ACEPTABLE pero con problemas significativos.
- Menor a 70: la traducción tiene problemas graves (truncamiento, omisiones importantes, significado incorrecto).

CRÍTICO — VERIFICACIÓN DE COMPLETITUD:
- Los metadatos abajo muestran el conteo de caracteres del original y la traducción.
- Si la traducción es significativamente más corta que el original (ratio < 0.85), probablemente está TRUNCADA o tiene OMISIONES.
- Una traducción truncada NO PUEDE obtener más de 70 en completitud, sin importar qué tan buenas sean las partes traducidas.
- Verifica que todas las secciones/párrafos del original estén presentes en la traducción.

Responde EXACTAMENTE con un JSON válido (sin markdown, sin bloques de código):

{
  "clarity": <0-100>,
  "fidelity": <0-100>,
  "completeness": <0-100>,
  "fluency": <0-100>,
  "errors": ["error 1", "error 2"],
  "overall_score": <0-100>
}

Criterios:
- Clarity: ¿La traducción es clara y fácil de entender?
- Fidelity: ¿Refleja fielmente el significado del original?
- Completeness: ¿TODA la información del original está presente? Verifica párrafos y secciones.
- Fluency: ¿Suena natural en el idioma destino?
- Errors: Lista específica de errores encontrados (vacía si no hay). Incluye truncamiento si se detecta.
- Overall score: Evaluación global ponderada. Problemas de completitud deben penalizar fuertemente esta puntuación."""


def build_evaluator_prompt(original: str, translation: str) -> str:
    orig_chars = len(original)
    trans_chars = len(translation)
    ratio = trans_chars / orig_chars if orig_chars > 0 else 0
    completeness_warning = ""
    if ratio < 0.85:
        completeness_warning = f"\n⚠️ ADVERTENCIA: La traducción es solo el {ratio:.0%} del largo del original. Esto sugiere fuertemente TRUNCAMIENTO u OMISIONES. Penaliza completitud en consecuencia."
    elif ratio < 0.92:
        completeness_warning = f"\n⚠️ NOTA: La traducción es el {ratio:.0%} del largo del original. Verifica posibles omisiones."

    return f"""Evalúa la siguiente traducción comparándola con el texto original.

=== METADATOS ===
Original: {orig_chars:,} caracteres
Traducción: {trans_chars:,} caracteres
Ratio: {ratio:.2f}{completeness_warning}

=== TEXTO ORIGINAL ===
{original}

=== TRADUCCIÓN ===
{translation}

Responde SOLO con el JSON de evaluación."""


# ─── Limpieza ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r' @,@ ', ',', text)
    text = re.sub(r' @\.@ ', '.', text)
    text = re.sub(r' @-@ ', '-', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_equal_signs(text: str) -> str:
    text = re.sub(r'^=+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*=+$', '', text, flags=re.MULTILINE)
    return text.strip()


def clean_llm_response(text: str) -> str:
    text = re.sub(r'^===.*===\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(Traducci[oó]n|Translation)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** → bold
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)  # *italic* → italic
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ### heading → heading
    return text.strip()


# ─── Parseo ──────────────────────────────────────────────────────────────────

def parse_metadata(raw: str) -> dict | None:
    result = {}
    for field, key in [("TITLE", "title"), ("CLASSIFICATION", "classification"),
                       ("SUMMARY", "summary")]:
        match = re.search(rf'^{field}:\s*(.+)', raw, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    match = re.search(r'^KEYWORDS:\s*(.+)', raw, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        result["keywords"] = [k.strip() for k in match.group(1).split(",") if k.strip()]

    missing = {"title", "keywords", "classification", "summary"} - set(result.keys())
    if missing:
        log.warning(f"Missing metadata fields: {missing}")
        return None
    return result


def parse_review(raw: str) -> dict | None:
    analysis_match = re.search(
        r'ANALYSIS:\s*\n(.+?)(?=FINAL TRANSLATION:)',
        raw, flags=re.DOTALL | re.IGNORECASE
    )
    translation_match = re.search(
        r'FINAL TRANSLATION:\s*\n(.+)',
        raw, flags=re.DOTALL | re.IGNORECASE
    )
    if not translation_match:
        return None

    critique = {}
    if analysis_match:
        analysis_text = analysis_match.group(1).strip()
        json_match = re.search(r'\{.*\}', analysis_text, flags=re.DOTALL)
        if json_match:
            try:
                critique = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                critique = {"raw": analysis_text}
        else:
            critique = {"raw": analysis_text}

    return {
        "critique": critique,
        "consensus_text": translation_match.group(1).strip(),
    }


def parse_evaluation(raw: str) -> dict | None:
    json_match = re.search(r'\{[^{}]*\}', raw, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group(0))
        required = {"clarity", "fidelity", "completeness", "fluency", "overall_score"}
        if not required.issubset(set(data.keys())):
            log.warning(f"Campos faltantes en evaluación: {required - set(data.keys())}")
            return None
        return data
    except json.JSONDecodeError as e:
        log.warning(f"JSON de evaluación inválido: {e}")
        return None


# ─── Lectura de archivo ──────────────────────────────────────────────────────

def read_file(path: Path) -> str | None:
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    log.error(f"No se pudo leer {path.name}")
    return None


# ─── División de artículos ───────────────────────────────────────────────────

def has_multiple_articles(text: str) -> bool:
    matches = re.findall(r'^ = [^=].+[^=] = $', text, flags=re.MULTILINE)
    return len(matches) > 1

def split_articles(text: str) -> list:
    pattern = r'(?=^ = [^=].+[^=] = $)'
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


# ─── Persistencia incremental ────────────────────────────────────────────────

def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        log.info(f"Progress loaded: {len(progress.get('completed_ids', []))} previous articles")
        return progress
    return {"completed_ids": [], "data": []}


def save_progress(progress_path: Path, progress: dict) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ─── Detección de idioma ─────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    spanish_words = {"de", "la", "el", "en", "los", "las", "del", "que", "por", "con", "una", "para", "como", "fue", "son"}
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return "en"
    count = sum(1 for w in words if w in spanish_words)
    ratio = count / len(words)
    return "es" if ratio > 0.08 else "en"


# ─── Chunking para artículos largos ──────────────────────────────────────────

def split_into_sections(text: str) -> list[str]:
    pattern = r'(?=^ = = .+? = = $)'
    parts = re.split(pattern, text, flags=re.MULTILINE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [text]
    return parts


def group_sections(sections: list[str], max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    chunks = []
    current_chunk = ""
    for section in sections:
        if len(section) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(section.strip())
        elif len(current_chunk) + len(section) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n\n" + section if current_chunk else section
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ─── Stage 1: Generación de traducciones candidatas ─────────────────────────

def _translate_chunk(chunk: str, source_lang: str,
                     config: ModelConfig, chunk_label: str) -> tuple[str | None, dict, bool]:
    """Translates a single chunk with language validation and retry.
    Returns (translated_text, token_usage, content_filter_flag)."""
    target_lang = "es" if source_lang == "en" else "en"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # First attempt with normal prompt
    resp = call_with_retries(
        TRANSLATION_SYSTEM,
        build_translation_prompt(chunk),
        f"{chunk_label}",
        config=config,
    )
    if resp.content is None:
        return None, usage, resp.content_filter
    usage["prompt_tokens"] += resp.prompt_tokens
    usage["completion_tokens"] += resp.completion_tokens
    usage["total_tokens"] += resp.total_tokens
    text = clean_llm_response(resp.content)

    # Validate output language
    detected = detect_language(text)
    if detected == target_lang:
        return text, usage, False

    # Output is in wrong language — retry with reinforced prompt
    for retry in range(1, TRANSLATION_LANGUAGE_RETRIES + 1):
        log.warning(f"      [{config.name}] {chunk_label} — output in '{detected}' instead of '{target_lang}', retry {retry}/{TRANSLATION_LANGUAGE_RETRIES}")
        resp = call_with_retries(
            TRANSLATION_SYSTEM,
            build_translation_retry_prompt(chunk, target_lang),
            f"{chunk_label} (lang retry {retry})",
            config=config,
        )
        if resp.content is None:
            continue
        usage["prompt_tokens"] += resp.prompt_tokens
        usage["completion_tokens"] += resp.completion_tokens
        usage["total_tokens"] += resp.total_tokens
        text = clean_llm_response(resp.content)
        detected = detect_language(text)
        if detected == target_lang:
            log.info(f"      [{config.name}] {chunk_label} — lang retry {retry} succeeded")
            return text, usage, False

    log.warning(f"      [{config.name}] {chunk_label} — failed to produce '{target_lang}' after {TRANSLATION_LANGUAGE_RETRIES} retries, using last output")
    return text, usage, False


def translate_article_with_model(article: str, article_id: int,
                                 config: ModelConfig) -> tuple[str, list[str], dict, bool] | None:
    """Retorna (texto_completo, chunks_traducidos, token_usage, content_filter) o None si falla."""
    source_lang = detect_language(article)
    est_tokens = len(article) // 4
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if est_tokens <= 6000:
        log.info(f"      [{config.name}] Article (~{est_tokens} tok) → 1 chunk")
        text, chunk_usage, cf = _translate_chunk(
            article, source_lang, config,
            f"translation article {article_id} [{config.name}]",
        )
        for k in usage:
            usage[k] += chunk_usage[k]
        if text is None:
            return None
        return text, [text], usage, cf

    sections = split_into_sections(article)
    chunks = group_sections(sections)
    log.info(f"      [{config.name}] Artículo largo (~{est_tokens} tok) → {len(chunks)} chunks")

    translations = []
    any_content_filter = False
    for j, chunk in enumerate(chunks, 1):
        log.info(f"      [{config.name}] Chunk {j}/{len(chunks)} ({len(chunk):,} chars)")
        text, chunk_usage, cf = _translate_chunk(
            chunk, source_lang, config,
            f"translation article {article_id} [{config.name}] chunk {j}/{len(chunks)}",
        )
        for k in usage:
            usage[k] += chunk_usage[k]
        if cf:
            any_content_filter = True
        if text is None:
            log.warning(f"      [{config.name}] Chunk {j} failed")
            return None
        translations.append(text)

    return "\n\n".join(translations), translations, usage, any_content_filter


def _translate_single_model(article: str, article_id: int,
                            config: ModelConfig) -> tuple[ModelConfig, tuple | None, bool]:
    """Translates with a single model. Returns (config, result_or_None, content_filter)."""
    try:
        log.info(f"    [Stage 1] Traduciendo con {config.name}...")
        result = translate_article_with_model(article, article_id, config)
        return config, result, False
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        detail, is_cf = _extract_error_detail(e)
        if is_cf:
            log.error(f"    [Stage 1] {config.name} — CONTENT FILTER: {detail[:200]}")
        else:
            log.error(f"    [Stage 1] {config.name} — unexpected error: {type(e).__name__}: {e}")
        return config, None, is_cf


def stage_1_generate_candidates(article: str, article_id: int,
                                models: list[ModelConfig]) -> tuple[list[dict], dict, list[str]]:
    """Retorna (candidates, token_usage_por_modelo, content_filter_models)."""
    candidates = []
    stage_usage = {}
    cf_models = []

    def _process_result(config, result, cf_from_caller=False):
        if result:
            text, chunks, usage, cf = result
            candidates.append({
                "model": config.name,
                "text": text,
                "chunks": chunks,
                "translated_at": datetime.now().isoformat(),
            })
            stage_usage[config.name] = usage
            if cf:
                cf_models.append(config.name)
                log.warning(f"    [Stage 1] {config.name} — CONTENT FILTER detected")
            log.info(f"    [Stage 1] {config.name} — OK ({len(text):,} chars, {len(chunks)} chunks, {usage['total_tokens']} tok)")
        else:
            if cf_from_caller:
                cf_models.append(config.name)
                log.warning(f"    [Stage 1] {config.name} — failed (CONTENT FILTER), excluded from consensus")
            else:
                log.warning(f"    [Stage 1] {config.name} — failed, excluded from consensus")

    if len(models) <= 1:
        for config in models:
            log.info(f"    [Stage 1] Traduciendo con {config.name}...")
            result = translate_article_with_model(article, article_id, config)
            _process_result(config, result)
        return candidates, stage_usage, cf_models

    # Multiple models — run in parallel
    log.info(f"    [Stage 1] Translating with {len(models)} models in parallel...")
    futures = {}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        for config in models:
            future = executor.submit(_translate_single_model, article, article_id, config)
            futures[future] = config

        for future in as_completed(futures):
            config, result, cf_flag = future.result()
            _process_result(config, result, cf_from_caller=cf_flag)

    return candidates, stage_usage, cf_models


# ─── Stage 2: Revisión por consenso ──────────────────────────────────────────

REVIEWER_CRITIQUE_SYSTEM = """Eres un editor y revisor profesional de traducciones.

Tu tarea es evaluar cada traducción candidata individualmente (fortalezas, debilidades, puntuación) y decidir cuál usar como base para la traducción final.

Responde EXACTAMENTE con este formato JSON (sin markdown, sin bloques de código):

{"candidates_analysis": [{"model": "nombre", "strengths": ["..."], "weaknesses": ["..."], "score": 0-100}], "decision": "explicación breve: qué candidato usar como base y qué mejoras aplicar"}"""


REVIEWER_MERGE_SYSTEM = """Eres un editor profesional de traducciones.

Ya analizaste las traducciones candidatas y decidiste cuál es la mejor base.
Ahora genera la traducción final MEJORADA para la siguiente sección.

Reglas:
- Fidelidad al original: NO inventes, NO omitas contenido.
- Naturalidad: la traducción final debe sonar fluida y profesional.
- Completitud: incluye TODO el contenido de la sección original.
- Terminología: conserva términos técnicos y nombres propios.
- NO agregues formato (negritas, cursivas, markdown) a menos que esté en el original.

Responde SOLO con el texto traducido, sin explicaciones ni prefijos."""


def _reviewer_critique(original: str, candidates: list[dict],
                       article_id: int, reviewer_config: ModelConfig) -> tuple[dict | None, dict]:
    """Stage 2a: Reviewer analyzes candidates and picks the best.
    Retorna (critique_dict, token_usage)."""
    # Build a shorter prompt with only first 2000 chars of each candidate for analysis
    parts = [f"=== TEXTO ORIGINAL (primeros 2000 chars) ===\n{original[:2000]}\n"]
    for i, c in enumerate(candidates, 1):
        parts.append(f"=== CANDIDATO {i} (modelo: {c['model']}, {len(c['text']):,} chars) ===\n{c['text'][:2000]}\n")
    parts.append("Evalúa cada candidato y decide cuál usar como base. Responde solo con JSON.")
    prompt = "\n".join(parts)

    resp = call_with_retries(
        REVIEWER_CRITIQUE_SYSTEM,
        prompt,
        f"critique artículo {article_id}",
        config=reviewer_config,
        num_predict=1024,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    if resp.content is None:
        return None, usage

    # Parse JSON from response
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
    """Stage 2b: Reviewer generates final translation for one section.
    Retorna (texto, token_usage)."""
    # Find the best candidate based on critique
    best_model = None
    if isinstance(critique, dict) and "candidates_analysis" in critique:
        analyses = critique["candidates_analysis"]
        best = max(analyses, key=lambda x: x.get("score", 0))
        best_model = best.get("model", "")

    # Build prompt with section from each candidate
    parts = [f"=== SECCIÓN ORIGINAL ({section_idx}/{total_sections}) ===\n{original_section}\n"]
    for i, c in enumerate(candidates, 1):
        label = "(MEJOR)" if best_model and best_model in c["model"] else ""
        parts.append(f"=== CANDIDATO {i} {label} ({c['model']}) ===\n{c['section_text']}\n")
    parts.append(
        f"Decisión: {critique.get('decision', 'Usar el mejor candidato como base.')}\n"
        "Genera la traducción final mejorada para esta sección. Responde SOLO con el texto traducido."
    )
    prompt = "\n".join(parts)

    resp = call_with_retries(
        REVIEWER_MERGE_SYSTEM,
        prompt,
        f"merge artículo {article_id} sección {section_idx}/{total_sections}",
        config=reviewer_config,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    return resp.content, usage


def _split_text_aligned(original: str, candidates: list[dict], max_chars: int = MAX_CHARS_PER_CHUNK) -> list[dict]:
    """Split original and candidate texts into aligned sections.

    Uses the chunk structure preserved from Stage 1 (each candidate has a
    'chunks' list that is 1:1 aligned with the original's section split).
    Falls back to proportional slicing only if chunks are missing or misaligned.
    """
    # Split original by level-2 headers
    sections_orig = split_into_sections(original)
    chunks_orig = group_sections(sections_orig, max_chars)

    if len(chunks_orig) <= 1:
        return [{
            "original": original,
            "candidates": [{"model": c["model"], "section_text": c["text"]} for c in candidates],
        }]

    n_chunks = len(chunks_orig)

    # Check if all candidates have aligned chunks from Stage 1
    all_have_chunks = all(
        "chunks" in c and len(c["chunks"]) == n_chunks
        for c in candidates
    )

    sections = []
    for j, chunk_orig in enumerate(chunks_orig):
        section = {"original": chunk_orig, "candidates": []}
        for c in candidates:
            if all_have_chunks:
                # Use the actual translated chunk (aligned 1:1 with original)
                section_text = c["chunks"][j]
            else:
                # Fallback: proportional slicing (legacy, may misalign)
                ratio_start = sum(len(ch) for ch in chunks_orig[:j]) / len(original)
                ratio_end = sum(len(ch) for ch in chunks_orig[:j+1]) / len(original)
                start = int(len(c["text"]) * ratio_start)
                end = int(len(c["text"]) * ratio_end)
                section_text = c["text"][start:end]
            section["candidates"].append({
                "model": c["model"],
                "section_text": section_text,
            })
        sections.append(section)

    if not all_have_chunks:
        log.warning(f"    [Stage 2] Chunks no alineados — usando fallback proporcional")

    return sections


def stage_2_review(original: str, candidates: list[dict],
                    article_id: int, reviewer_config: ModelConfig) -> tuple[dict | None, dict]:
    """Retorna (review_dict, token_usage)."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not candidates:
        return None, usage

    est_tokens = len(original) // 4
    log.info(f"    [Stage 2] Revisando con {reviewer_config.name}...")

    # Step 1: Critique (always on full article summary)
    critique, critique_usage = _reviewer_critique(original, candidates, article_id, reviewer_config)
    for k in usage:
        usage[k] += critique_usage[k]
    if critique is None:
        log.warning(f"    [Stage 2] Critique falló — fallback a candidato 1")
        return None, usage

    log.info(f"    [Stage 2] Critique completada")

    # Step 2: Generate final translation
    if est_tokens <= 6000:
        # Short article: single merge call
        sections = [{"original": original, "candidates": [
            {"model": c["model"], "section_text": c["text"]} for c in candidates
        ]}]
    else:
        # Long article: split and merge by sections
        sections = _split_text_aligned(original, candidates)
        log.info(f"    [Stage 2] Artículo largo (~{est_tokens} tok) → {len(sections)} secciones")

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
        merged_parts.append(clean_llm_response(result))

    consensus_text = "\n\n".join(merged_parts)

    review = {
        "critique": critique,
        "consensus_text": consensus_text,
        "model": reviewer_config.name,
        "reviewed_at": datetime.now().isoformat(),
    }
    log.info(f"    [Stage 2] Revisión completada ({len(consensus_text):,} chars, {usage['total_tokens']} tok)")
    return review, usage


# ─── Stage 3: Evaluación ────────────────────────────────────────────────────

def _adjust_evaluation_by_completeness(evaluation: dict, original: str,
                                        translation: str, article_id: int) -> dict:
    """Verificación programática: ajusta scores si el ratio de chars indica truncamiento.

    El LLM evaluador puede no detectar truncamiento (especialmente en textos largos).
    Esta función compara longitudes y penaliza completeness/overall si hay evidencia
    objetiva de contenido faltante.
    """
    orig_chars = len(original)
    trans_chars = len(translation)
    ratio = trans_chars / orig_chars if orig_chars > 0 else 0

    llm_completeness = evaluation.get("completeness", 100)
    llm_overall = evaluation.get("overall_score", 100)

    # Calcular completeness programático basado en ratio
    if ratio < 0.50:
        prog_completeness = max(int(ratio * 100), 10)  # 26% → 26, 36% → 36
    elif ratio < 0.75:
        prog_completeness = max(int(ratio * 100), 40)  # 79% → 79
    elif ratio < 0.85:
        prog_completeness = max(int(ratio * 100), 70)
    elif ratio < 0.92:
        prog_completeness = max(int(ratio * 105), 85)  # leve tolerancia
    else:
        prog_completeness = None  # ratio OK, confiar en el LLM

    if prog_completeness is not None and llm_completeness > prog_completeness + 10:
        # El LLM sobreestimó completeness — corregir
        old_completeness = evaluation["completeness"]
        evaluation["completeness"] = prog_completeness

        # Recalcular overall: completeness pesa ~30% del overall
        other_avg = (
            evaluation.get("clarity", 90)
            + evaluation.get("fidelity", 90)
            + evaluation.get("fluency", 90)
        ) / 3
        evaluation["overall_score"] = int(other_avg * 0.6 + prog_completeness * 0.4)

        # Agregar error de truncamiento si no existe
        errors = evaluation.get("errors", [])
        if not isinstance(errors, list):
            errors = [str(errors)]
        truncation_error = f"TRUNCADO: traducción es {ratio:.0%} del original ({trans_chars:,} vs {orig_chars:,} chars)"
        if not any("TRUNCADO" in e or "truncat" in e.lower() for e in errors):
            errors.insert(0, truncation_error)
        evaluation["errors"] = errors

        evaluation["programmatic_adjustment"] = {
            "ratio": round(ratio, 3),
            "llm_completeness": old_completeness,
            "adjusted_completeness": prog_completeness,
            "llm_overall": llm_overall,
            "adjusted_overall": evaluation["overall_score"],
        }
        log.warning(
            f"    [Stage 3] Ajuste programático art. {article_id}: "
            f"ratio={ratio:.2f}, completeness {old_completeness}→{prog_completeness}, "
            f"overall {llm_overall}→{evaluation['overall_score']}"
        )

    # Guardar ratio siempre para trazabilidad
    evaluation["char_ratio"] = round(ratio, 3)
    return evaluation


def stage_3_evaluate(original: str, translation: str,
                    article_id: int, evaluator_config: ModelConfig) -> tuple[dict | None, dict]:
    """Retorna (evaluation_dict, token_usage)."""
    log.info(f"    [Stage 3] Evaluando con {evaluator_config.name}...")

    resp = call_with_retries(
        EVALUATOR_SYSTEM,
        build_evaluator_prompt(original, translation),
        f"evaluación artículo {article_id}",
        config=evaluator_config,
        num_predict=1024,
    )
    usage = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens, "total_tokens": resp.total_tokens}
    if resp.content is None:
        return None, usage

    evaluation = parse_evaluation(resp.content)
    if evaluation is None:
        log.warning(f"    [Stage 3] No se pudo parsear la evaluación")
        return None, usage

    # Verificación programática de completitud
    evaluation = _adjust_evaluation_by_completeness(
        evaluation, original, translation, article_id
    )

    evaluation["model"] = evaluator_config.name
    evaluation["evaluated_at"] = datetime.now().isoformat()
    log.info(f"    [Stage 3] Score: {evaluation.get('overall_score', '?')}/100")
    return evaluation, usage


# ─── Procesamiento de un artículo (pipeline completo) ────────────────────────

def process_article(article: str, article_id: int,
                      pipeline_config: dict) -> dict | None:
    language = detect_language(article)
    original_text = clean_equal_signs(article)
    log.info(f"    Language detected: {language}")

    translation_models = pipeline_config["translation_models"]
    reviewer_config = pipeline_config["reviewer_model"]
    evaluator_config = pipeline_config["evaluator_model"]
    metadata_config = pipeline_config["metadata_model"]

    is_consensus = len(translation_models) > 1 or reviewer_config is not None

    # Token usage acumulado por stage
    token_usage = {
        "stage_1_translation": {},
        "stage_2_review": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "stage_3_evaluation": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # ── Stage 1: Generar candidatos ──
    log.info(f"    [Stage 1] Generando {len(translation_models)} traducción(es)...")
    candidates, s1_usage, cf_models = stage_1_generate_candidates(article, article_id, translation_models)
    token_usage["stage_1_translation"] = s1_usage
    if not candidates:
        log.error(f"    Ninguna traducción candidata exitosa")
        return None

    # ── Stage 2: Revisión (si hay reviewer configurado) ──
    review = None
    if reviewer_config and candidates:
        review, s2_usage = stage_2_review(original_text, candidates, article_id, reviewer_config)
        token_usage["stage_2_review"] = s2_usage

    # Determinar traducción final
    if review and review.get("consensus_text"):
        final_translation = review["consensus_text"]
    else:
        final_translation = candidates[0]["text"]

    # ── Stage 3: Evaluación (si hay evaluator configurado) ──
    evaluation = None
    if evaluator_config:
        evaluation, s3_usage = stage_3_evaluate(
            original_text, final_translation, article_id, evaluator_config
        )
        token_usage["stage_3_evaluation"] = s3_usage

    # ── Metadata (sobre la traducción final) ──
    log.info(f"    Metadata original...")
    meta_resp = call_with_retries(
        METADATA_SYSTEM_EN if language == "en" else METADATA_SYSTEM_ES,
        build_metadata_prompt(article, lang=language),
        f"metadata artículo {article_id}",
        config=metadata_config, num_predict=1024,
    )
    token_usage["metadata"]["prompt_tokens"] += meta_resp.prompt_tokens
    token_usage["metadata"]["completion_tokens"] += meta_resp.completion_tokens
    token_usage["metadata"]["total_tokens"] += meta_resp.total_tokens
    if meta_resp.content is None:
        return None
    metadata = parse_metadata(meta_resp.content)
    if metadata is None:
        return None

    log.info(f"    Metadata traducción...")
    meta_trans_resp = call_with_retries(
        METADATA_SYSTEM_ES if language == "en" else METADATA_SYSTEM_EN,
        build_metadata_prompt(final_translation, lang="es" if language == "en" else "en"),
        f"metadata traducida artículo {article_id}",
        config=metadata_config, num_predict=1024,
    )
    token_usage["metadata"]["prompt_tokens"] += meta_trans_resp.prompt_tokens
    token_usage["metadata"]["completion_tokens"] += meta_trans_resp.completion_tokens
    token_usage["metadata"]["total_tokens"] += meta_trans_resp.total_tokens
    translated_metadata = parse_metadata(meta_trans_resp.content) if meta_trans_resp.content else None
    if translated_metadata is None:
        log.warning(f"    Metadata traducida falló — usando original como fallback")
        translated_metadata = metadata

    # ── Calcular costos por stage ──
    cost = {}
    total_cost = 0.0

    # Stage 1: costo por modelo
    s1_cost = {}
    for model_name, u in token_usage["stage_1_translation"].items():
        c = calculate_cost(model_name, u["prompt_tokens"], u["completion_tokens"])
        s1_cost[model_name] = round(c, 6)
        total_cost += c
    cost["stage_1_translation"] = s1_cost

    # Stage 2: reviewer
    s2 = token_usage["stage_2_review"]
    reviewer_name = reviewer_config.name if reviewer_config else ""
    s2_cost = calculate_cost(reviewer_name, s2["prompt_tokens"], s2["completion_tokens"])
    cost["stage_2_review"] = round(s2_cost, 6)
    total_cost += s2_cost

    # Stage 3: evaluator
    s3 = token_usage["stage_3_evaluation"]
    evaluator_name = evaluator_config.name if evaluator_config else ""
    s3_cost = calculate_cost(evaluator_name, s3["prompt_tokens"], s3["completion_tokens"])
    cost["stage_3_evaluation"] = round(s3_cost, 6)
    total_cost += s3_cost

    # Metadata
    meta = token_usage["metadata"]
    meta_cost = calculate_cost(metadata_config.name, meta["prompt_tokens"], meta["completion_tokens"])
    cost["metadata"] = round(meta_cost, 6)
    total_cost += meta_cost

    cost["total_usd"] = round(total_cost, 6)
    token_usage["cost_usd"] = cost

    # ── Log de tokens y costo ──
    total_tok = sum(
        u.get("total_tokens", 0) if isinstance(u, dict) else sum(v.get("total_tokens", 0) for v in u.values())
        for u in [token_usage["stage_1_translation"], token_usage["stage_2_review"],
                  token_usage["stage_3_evaluation"], token_usage["metadata"]]
    )
    log.info(f"    Tokens totales artículo {article_id}: {total_tok:,} | Costo: ${total_cost:.4f} USD")

    # ── Ensamblar JSON ──
    timestamp = datetime.now().isoformat()
    translated_by = review["model"] if review else candidates[0]["model"]

    # Archivo principal: solo el resultado final (sin pipeline)
    if language == "en":
        result = {
            "id": article_id,
            "source_language": "en",
            "english": {
                "title": metadata["title"],
                "keywords": metadata["keywords"],
                "summary": metadata["summary"],
                "classification": metadata["classification"],
                "text": original_text,
            },
            "spanish": {
                "title": translated_metadata["title"],
                "keywords": translated_metadata["keywords"],
                "summary": translated_metadata["summary"],
                "classification": translated_metadata["classification"],
                "text": clean_equal_signs(final_translation),
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
                "text": clean_equal_signs(final_translation),
                "translated_at": timestamp,
                "translated_by": translated_by,
            },
            "spanish": {
                "title": metadata["title"],
                "keywords": metadata["keywords"],
                "summary": metadata["summary"],
                "classification": metadata["classification"],
                "text": original_text,
            },
        }

    # Archivo de candidates: pipeline completo (traducciones + review + evaluation)
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


# ─── Procesamiento de archivo ────────────────────────────────────────────────

def process_file(path: Path, pipeline_config: dict,
                     id_offset: int = 0, start_from: int = 1, limit: int = 0) -> tuple[list, list]:
    """Retorna (data_principal, candidates_full)."""
    text = read_file(path)
    if not text:
        return [], []

    text = clean_text(text)

    if has_multiple_articles(text):
        articles = split_articles(text)
        log.info(f"Multiple articles detected: {len(articles)}")
    else:
        articles = [text]
        log.info(f"Single article")

    progress_path = OUTPUT_DIR / f".progress_{path.stem}.json"
    progress = load_progress(progress_path)

    data = progress["data"]
    all_candidates = progress.get("candidates", [])
    completed_ids = set(progress["completed_ids"])

    processed_count = 0
    for i, article in enumerate(articles, start=1):
        article_id = id_offset + i

        if article_id < start_from:
            continue

        if limit > 0 and processed_count >= limit:
            log.info(f"Limit reached ({limit} articles)")
            break

        if article_id in completed_ids:
            log.info(f"  [{i}/{len(articles)}] ID {article_id} — already processed, skipping")
            continue

        title = article.splitlines()[0][:60].strip()
        log.info(f"  [{i}/{len(articles)}] ID {article_id} — {title}...")

        result = process_article(article, article_id, pipeline_config)
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

    if progress_path.exists():
        progress_path.unlink()
        log.info("Temporary progress removed")

    return data, all_candidates


# ─── Entry point ─────────────────────────────────────────────────────────────

def run(input_file: Path, max_retries: int = MAX_RETRIES, start_from: int = 1, limit: int = 0) -> None:
    global MAX_RETRIES
    MAX_RETRIES = max_retries

    if not input_file.exists():
        log.error(f"El archivo '{input_file}' no existe.")
        return

    if not input_file.is_file():
        log.error(f"'{input_file}' no es un archivo. Especifica un archivo, no una carpeta.")
        return

    if input_file.suffix not in EXTENSIONS:
        log.error(f"Extensión '{input_file.suffix}' no soportada. Extensiones válidas: {EXTENSIONS}")
        return

    # Cargar y verificar pipeline
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

    log.info(f"Procesando: {input_file}")
    if limit > 0:
        log.info(f"Límite: {limit} artículos")
    data, all_candidates = process_file(input_file, pipeline_config, start_from=start_from, limit=limit)

    if not data:
        log.warning(f"Sin datos — no se genera JSON.")
        return

    date_stamp = datetime.now().strftime("%y%m%d_%H%M%S")  # AAMMDD_HHMMSS

    # ── Archivo principal (sin textos de candidatos) ──
    output = {
        "source_name":   "wikitext",
        "url":           "https://www.kaggle.com/datasets/rohitgr/wikitext/data",
        "creation_date": "2023-01-01",
        "authors_name":  ["Rohit Gupta (Owner)"],
        "payload": [
            {
                "file_name":        input_file.name,
                "file_size":        f"{round(input_file.stat().st_size / (1024*1024), 2)} MB",
                "file_format":      input_file.suffix.lstrip("."),
                "description":      "Dataset derived from carefully selected Wikipedia articles. It is widely used to evaluate language models in document-level coherent text generation tasks. Its adaptation to Spanish allows us to analyze the model's capacity to capture discursive structure and thematic consistency.",
                "translation_date": translation_date,
                "llm_used":         models_info,
                "data":             data,
            }
        ]
    }

    output_path = OUTPUT_DIR / f"{input_file.stem}_{date_stamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info(f"Guardado: {output_path}")

    # ── Archivo de pipeline (textos completos de cada traducción) ──
    if all_candidates:
        # Calcular totales de tokens y costo de todos los artículos
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
            # Tokens por stage
            for stage_key in ["stage_2_review", "stage_3_evaluation", "metadata"]:
                stage = tu.get(stage_key, {})
                for tk in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    total_summary["total_tokens"][tk] += stage.get(tk, 0)
            # Stage 1: por modelo
            for model_name, u in tu.get("stage_1_translation", {}).items():
                for tk in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    total_summary["total_tokens"][tk] += u.get(tk, 0)
            # Costos
            cost = tu.get("cost_usd", {})
            for model_name, c in cost.get("stage_1_translation", {}).items():
                total_summary["total_cost_usd"]["stage_1_translation"][model_name] = (
                    total_summary["total_cost_usd"]["stage_1_translation"].get(model_name, 0.0) + c
                )
            total_summary["total_cost_usd"]["stage_2_review"] += cost.get("stage_2_review", 0.0)
            total_summary["total_cost_usd"]["stage_3_evaluation"] += cost.get("stage_3_evaluation", 0.0)
            total_summary["total_cost_usd"]["metadata"] += cost.get("metadata", 0.0)
            total_summary["total_cost_usd"]["total"] += cost.get("total_usd", 0.0)

        # Redondear costos
        for model_name in total_summary["total_cost_usd"]["stage_1_translation"]:
            total_summary["total_cost_usd"]["stage_1_translation"][model_name] = round(
                total_summary["total_cost_usd"]["stage_1_translation"][model_name], 4
            )
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

        pipeline_path = OUTPUT_DIR / f"{input_file.stem}_pipeline_{date_stamp}.json"
        with open(pipeline_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_output, f, ensure_ascii=False, indent=2)

        log.info(f"Pipeline: {pipeline_path}")
        log.info(f"Costo total: ${total_summary['total_cost_usd']['total']:.4f} USD")

    log.info(f"Artículos procesados: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa artículos WikiText con pipeline LLM")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/WikiText/wikitext-103-raw.wiki.valid.raw"),
        help="Archivo de entrada (.raw o .tokens)",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=0,
        help="Offset para IDs de artículos (útil al procesar múltiples archivos)",
    )
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=MAX_RETRIES,
        help=f"Máximo de intentos por artículo (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        default=1,
        help="Empezar desde este número de artículo (default: 1)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Número máximo de artículos a procesar (default: 0 = todos)",
    )
    args = parser.parse_args()

    run(args.input, max_retries=args.retries, start_from=args.start_from, limit=args.limit)
