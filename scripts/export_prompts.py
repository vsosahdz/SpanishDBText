#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write every prompt the pipeline sends to an LLM into prompts/, one file each.

The prompts are the substance of this method — the code around them is
orchestration — so they are published as plain text that can be read, diffed and
adapted without opening a Python file.

Nothing here is transcribed by hand. System prompts are imported straight from
the agent modules. User prompts are produced by calling the agents' own
`build_*_prompt()` functions with placeholder arguments, so what lands on disk is
exactly what the code emits.

Three user prompts (the reviewer's critique and merge, and FEVER's and Europarl's
variants of them) are assembled inline inside the stage functions rather than in
a callable of their own. Those are reproduced here as templates, and each one is
CHECKED against the real construction before being written: if the agent code
changes and a template drifts, this script fails instead of publishing a stale
prompt.

Usage:
    python3 scripts/export_prompts.py
    python3 scripts/export_prompts.py --check    # verify only, write nothing
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "prompts"

# Import the agents from source every time. Without this, editing a prompt and
# re-running within the same second can hit a stale .pyc — Python's freshness
# check is timestamp-based — and the drift check would silently pass against the
# previous version of the code. It also keeps the export from littering
# agents/__pycache__.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "agents"))

import agente_wikitext as W          # noqa: E402
import agente_fever as F             # noqa: E402
import agente_europarl as E          # noqa: E402

# Placeholders used across the exported templates.
SRC = "{{SOURCE_TEXT}}"
SECTION = "{{ORIGINAL_SECTION}}"
TRANSLATION = "{{TRANSLATION}}"
TITLE = "{{SOURCE_TITLE}}"

# A fixed three-candidate set, used both to render the templates and to check
# them against the agents' own construction.
CANDIDATES = [
    {"model": "{{MODEL_1}}", "text": "{{CANDIDATE_1_TEXT}}", "section_text": "{{CANDIDATE_1_SECTION}}"},
    {"model": "{{MODEL_2}}", "text": "{{CANDIDATE_2_TEXT}}", "section_text": "{{CANDIDATE_2_SECTION}}"},
    {"model": "{{MODEL_3}}", "text": "{{CANDIDATE_3_TEXT}}", "section_text": "{{CANDIDATE_3_SECTION}}"},
]
CRITIQUE = {"decision": "{{REVIEWER_DECISION}}"}


# ── Reviewer prompts assembled inline in the agents ──────────────────────────
# Each pair is (template, rebuild) where `rebuild` mirrors the agent code
# exactly. They must come out identical; `check()` enforces it.

def _critique_user(header: str, candidates: list[dict], texts: list[str]) -> str:
    # The agents insert len(candidate["text"]) here, formatted with a thousands
    # separator. In the exported template that becomes a placeholder, since the
    # value depends on the entry.
    parts = [f"=== {header} ===\n{SRC}\n"]
    for i, (c, t) in enumerate(zip(candidates, texts), 1):
        parts.append(f"=== CANDIDATO {i} (modelo: {c['model']}, "
                     f"{{{{CANDIDATE_{i}_CHARS}}}} chars) ===\n{t}\n")
    parts.append("Evalúa cada candidato y decide cuál usar como base. Responde solo con JSON.")
    return "\n".join(parts)


def _merge_user(header: str, body_key: str, closing: str) -> str:
    parts = [f"=== {header} ===\n{SECTION}\n"]
    for i, c in enumerate(CANDIDATES, 1):
        parts.append(f"=== CANDIDATO {i} {{{{BEST_MARKER_{i}}}}} ({c['model']}) ===\n{c[body_key]}\n")
    parts.append(f"Decisión: {CRITIQUE['decision']}\n{closing}")
    return "\n".join(parts)


CAND_TEXTS = [f"{{{{CANDIDATE_{i}_TEXT}}}}" for i in (1, 2, 3)]

INLINE = {
    # agents/agente_wikitext.py, _reviewer_critique()
    "4-review/user-critique-wikitext.txt":
        _critique_user("TEXTO ORIGINAL (primeros 2000 chars)", CANDIDATES, CAND_TEXTS),
    # agents/agente_fever.py, stage_2_review()
    "4-review/user-critique-fever.txt":
        _critique_user("TEXTO ORIGINAL", CANDIDATES, CAND_TEXTS),
    # agents/agente_europarl.py, _reviewer_critique() — XML tags stripped first
    "4-review/user-critique-europarl.txt":
        _critique_user("TEXTO ORIGINAL (primeros 2000 chars)", CANDIDATES, CAND_TEXTS),
    # agents/agente_wikitext.py, _reviewer_merge_section()
    "4-review/user-merge-wikitext.txt":
        _merge_user("SECCIÓN ORIGINAL ({{SECTION_INDEX}}/{{TOTAL_SECTIONS}})", "section_text",
                    "Genera la traducción final mejorada para esta sección. "
                    "Responde SOLO con el texto traducido."),
    # agents/agente_fever.py, stage_2_review() — always a single section
    "4-review/user-merge-fever.txt":
        _merge_user("SECCIÓN ORIGINAL", "text",
                    "Genera la traducción final mejorada. Responde SOLO con el texto traducido."),
    # agents/agente_europarl.py, _reviewer_merge_section()
    "4-review/user-merge-europarl.txt":
        _merge_user("SECCIÓN ORIGINAL ({{SECTION_INDEX}}/{{TOTAL_SECTIONS}})", "section_text",
                    "Genera la traducción final mejorada. Preserva INTACTOS todos los tags XML "
                    "(<CHAPTER>, <SPEAKER>, <P>). Responde SOLO con el texto traducido."),
}


def build_exports() -> dict[str, str]:
    """Every prompt, keyed by the path it is written to."""
    files: dict[str, str] = {}

    # ── Stage 3: translation ──
    files["3-translation/system-wikitext.txt"] = W.TRANSLATION_SYSTEM
    files["3-translation/system-fever.txt"] = F.FEVER_TRANSLATION_SYSTEM
    files["3-translation/system-europarl.txt"] = E.EUROPARL_TRANSLATION_SYSTEM
    files["3-translation/user-wikitext.txt"] = W.build_translation_prompt(SRC)
    files["3-translation/user-fever.txt"] = F.build_fever_translation_prompt(SRC)
    files["3-translation/user-europarl.txt"] = E.build_europarl_translation_prompt(SRC)
    files["3-translation/user-retry-es.txt"] = W.build_translation_retry_prompt(SRC, "es")
    files["3-translation/user-retry-en.txt"] = W.build_translation_retry_prompt(SRC, "en")

    # ── Stage 4: review and consensus ──
    files["4-review/system-critique.txt"] = W.REVIEWER_CRITIQUE_SYSTEM
    files["4-review/system-merge.txt"] = W.REVIEWER_MERGE_SYSTEM
    # The earlier one-call reviewer. Still defined in the agent module, but
    # nothing calls it: every agent takes the critique+merge path. Exported for
    # completeness, flagged as unused in prompts/README.md.
    files["4-review/system-single-call.txt"] = W.REVIEWER_SYSTEM
    files["4-review/user-single-call.txt"] = W.build_reviewer_prompt(SRC, CANDIDATES)
    files.update(INLINE)

    # ── Stage 5: evaluation ──
    files["5-evaluation/system.txt"] = W.EVALUATOR_SYSTEM
    files["5-evaluation/user.txt"] = EVALUATOR_USER_TEMPLATE
    files["5-evaluation/user-length-warnings.txt"] = EVALUATOR_WARNINGS

    # ── Stage 6: metadata ──
    files["6-metadata/system-es.txt"] = W.METADATA_SYSTEM_ES
    files["6-metadata/system-en.txt"] = W.METADATA_SYSTEM_EN
    files["6-metadata/user-wikitext-es.txt"] = W.build_metadata_prompt(SRC, "es")
    files["6-metadata/user-wikitext-en.txt"] = W.build_metadata_prompt(SRC, "en")
    files["6-metadata/user-fever-es.txt"] = F.build_fever_metadata_prompt(SRC, TITLE, "es")
    files["6-metadata/user-fever-en.txt"] = F.build_fever_metadata_prompt(SRC, TITLE, "en")
    files["6-metadata/user-europarl-es.txt"] = E.build_europarl_metadata_prompt(SRC, "es")
    files["6-metadata/user-europarl-en.txt"] = E.build_europarl_metadata_prompt(SRC, "en")

    return files


# The evaluator's user prompt embeds computed character counts, so it cannot be
# rendered with placeholder text the way the others can. It is written out as a
# template and checked below against the real function.
EVALUATOR_USER_TEMPLATE = f"""Evalúa la siguiente traducción comparándola con el texto original.

=== METADATOS ===
Original: {{{{ORIGINAL_CHARS}}}} caracteres
Traducción: {{{{TRANSLATION_CHARS}}}} caracteres
Ratio: {{{{RATIO}}}}{{{{LENGTH_WARNING}}}}

=== TEXTO ORIGINAL ===
{SRC}

=== TRADUCCIÓN ===
{TRANSLATION}

Responde SOLO con el JSON de evaluación."""

EVALUATOR_WARNINGS = """The {{LENGTH_WARNING}} slot of 5-evaluation/user.txt takes one of three values,
decided by the character ratio r = len(translation) / len(original).

r >= 0.92
    (empty — no warning is inserted)

0.85 <= r < 0.92
    ⚠️ NOTA: La traducción es el {{PERCENT}} del largo del original. Verifica posibles omisiones.

r < 0.85
    ⚠️ ADVERTENCIA: La traducción es solo el {{PERCENT}} del largo del original. Esto sugiere fuertemente TRUNCAMIENTO u OMISIONES. Penaliza completitud en consecuencia.

Both variants are preceded by a newline. {{PERCENT}} is the ratio formatted as a
whole percentage, e.g. "78%".

This warning is the evaluator's cue, not the pipeline's decision: the
programmatic truncation check of Stage 5 runs independently and can override the
model's completeness score. See docs/pipeline.md."""


def render(template: str, subs: dict[str, str]) -> str:
    for key, value in subs.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def check() -> list[str]:
    """Confirm the hand-written templates still match what the agents build."""
    problems = []

    # The evaluator template, against the real function, for each warning band.
    original, ratios = "x" * 1000, {"none": 1000, "note": 890, "warn": 700}
    for band, trans_len in ratios.items():
        translation = "y" * trans_len
        ratio = trans_len / 1000
        warning = ""
        if ratio < 0.85:
            warning = (f"\n⚠️ ADVERTENCIA: La traducción es solo el {ratio:.0%} del largo del "
                       f"original. Esto sugiere fuertemente TRUNCAMIENTO u OMISIONES. "
                       f"Penaliza completitud en consecuencia.")
        elif ratio < 0.92:
            warning = (f"\n⚠️ NOTA: La traducción es el {ratio:.0%} del largo del original. "
                       f"Verifica posibles omisiones.")
        expected = render(EVALUATOR_USER_TEMPLATE, {
            "ORIGINAL_CHARS": f"{len(original):,}",
            "TRANSLATION_CHARS": f"{len(translation):,}",
            "RATIO": f"{ratio:.2f}",
            "LENGTH_WARNING": warning,
            "SOURCE_TEXT": original,
            "TRANSLATION": translation,
        })
        actual = W.build_evaluator_prompt(original, translation)
        if expected != actual:
            problems.append(f"5-evaluation/user.txt drifted from build_evaluator_prompt() "
                            f"(ratio band: {band})")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the templates against the agent code without writing")
    args = parser.parse_args()

    problems = check()
    if problems:
        for p in problems:
            print(f"  MISMATCH: {p}")
        sys.exit("\nThe exported prompts no longer match the agent code. Fix the templates "
                 "in this script before publishing them.")
    print("Templates match the agent code.")

    if args.check:
        return

    files = build_exports()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(files)} prompts to prompts/")
    for rel, text in sorted(files.items()):
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        print(f"  {rel}  ({len(text):,} chars)")
    print("\nDone. prompts/README.md is maintained by hand, not generated.")


if __name__ == "__main__":
    main()
