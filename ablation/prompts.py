"""Evaluator prompt variants for the 2x2 ablation.

Factors:
  G  scoring guide (anchoring): the "ESTRICTO y CRÍTICO" role, the sentence "una
     puntuación perfecta debe ser extremadamente rara" and the six bands of Table 1.
  L  length evidence: the "VERIFICACIÓN DE COMPLETITUD" block of the system prompt and
     the METADATOS section of the user prompt (counts, ratio and warnings < 0.92 / < 0.85).

G1L1 is exactly the production prompt. The other variants are built by removing
blocks from the production text, so any future change in
``agents/agente_wikitext.py`` carries over (the ``assert`` statements fail if a block
no longer exists). Everything else (JSON format, definition of the criteria) is shared.

Usage, to inspect the prompts:
    python prompts.py        # writes prompts/<config>_system.txt and an example user prompt
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
sys.path.insert(0, str(AGENTS))

import agente_wikitext as A  # noqa: E402  (production prompt and functions)

CONFIGS = ["G1L1", "G0L1", "G1L0", "G0L0"]
DESCRIPTION = {
    "G1L1": "producción: guía + evidencia de longitud",
    "G0L1": "sin guía, con evidencia de longitud",
    "G1L0": "con guía, sin evidencia de longitud",
    "G0L0": "sin guía y sin evidencia de longitud",
}

_ROLE_STRICT = "Eres un evaluador ESTRICTO y CRÍTICO experto en calidad de traducción."
_ROLE_NEUTRAL = "Eres un evaluador experto en calidad de traducción."
_RARE = " Sé exigente — una puntuación perfecta (100) debe ser extremadamente rara."
_GUIDE_RE = re.compile(r"GUÍA IMPORTANTE DE PUNTUACIÓN:\n(?:- .*\n)+\n")
_LENGTH_RE = re.compile(r"CRÍTICO — VERIFICACIÓN DE COMPLETITUD:\n(?:- .*\n)+\n")
_METADATA_RE = re.compile(r"=== METADATOS ===\n.*?\n\n(?==== TEXTO ORIGINAL ===)", re.DOTALL)


def _remove_once(pattern: re.Pattern, text: str, what: str) -> str:
    new, n = pattern.subn("", text)
    assert n == 1, f"block '{what}' found {n} times in the production prompt"
    return new


def system_prompt(config: str) -> str:
    assert config in CONFIGS, config
    s = A.EVALUATOR_SYSTEM
    if config[1] == "0":  # no guide
        assert s.count(_ROLE_STRICT) == 1 and s.count(_RARE) == 1
        s = s.replace(_ROLE_STRICT, _ROLE_NEUTRAL).replace(_RARE, "")
        s = _remove_once(_GUIDE_RE, s, "guide")
    if config[3] == "0":  # no length evidence
        s = _remove_once(_LENGTH_RE, s, "completeness")
    return s


def user_prompt(config: str, original: str, translation: str) -> str:
    assert config in CONFIGS, config
    u = A.build_evaluator_prompt(original, translation)
    if config[3] == "0":
        u = _remove_once(_METADATA_RE, u, "metadata")
    return u


def main() -> None:
    out = HERE / "prompts"
    out.mkdir(exist_ok=True)
    sample_orig = "The quick brown fox jumps over the lazy dog. " * 3
    sample_trad = "El rápido zorro marrón salta sobre el perro perezoso. " * 2
    for c in CONFIGS:
        (out / f"{c}_system.txt").write_text(system_prompt(c), encoding="utf-8")
        (out / f"{c}_user_example.txt").write_text(
            user_prompt(c, sample_orig, sample_trad), encoding="utf-8")
        print(f"{c}  ({DESCRIPTION[c]}): system {len(system_prompt(c)):,} chars")
    assert system_prompt("G1L1") == A.EVALUATOR_SYSTEM
    print(f"prompts written to {out.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
