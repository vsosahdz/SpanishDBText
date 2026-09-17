#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that the environment is ready before launching a pipeline run.

A production run costs money and takes hours, so it is worth a few seconds to
confirm that the interpreter, the packages, the .env file and every configured
model endpoint are actually in place. Nothing here writes data; the model check
sends one short prompt per configured model, which costs a fraction of a cent.

Usage:
    python3 scripts/verify_setup.py              # local checks only
    python3 scripts/verify_setup.py --models     # also ping every configured model
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = [
    ("httpx", "httpx"),
    ("dotenv", "python-dotenv"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]

OPTIONAL_PACKAGES = [
    ("spacy", "spacy", "syntactic and named-entity metrics"),
    ("sentence_transformers", "sentence-transformers", "cross-lingual semantic metrics"),
    ("torch", "torch", "cross-lingual semantic metrics"),
    ("matplotlib", "matplotlib", "the notebooks"),
    ("jupyterlab", "jupyterlab", "the notebooks"),
]

SPACY_MODELS = ["en_core_web_sm", "es_core_news_sm"]

ENV_KEYS = ["TRANSLATION_MODEL_1", "TRANSLATION_MODEL_2", "TRANSLATION_MODEL_3",
            "REVIEWER_MODEL", "EVALUATOR_MODEL"]

PASS, FAIL, WARN = "  [ ok ]", "  [FAIL]", "  [warn]"
failures = 0


def report(ok: bool, message: str, fatal: bool = True) -> None:
    global failures
    if ok:
        print(f"{PASS} {message}")
    else:
        print(f"{FAIL if fatal else WARN} {message}")
        if fatal:
            failures += 1


def check_python() -> None:
    print("\nInterpreter")
    major, minor = sys.version_info[:2]
    report((major, minor) >= (3, 11),
           f"Python {major}.{minor} (3.11 or newer required)")


def check_packages() -> None:
    print("\nRequired packages")
    for module, package in REQUIRED_PACKAGES:
        try:
            __import__(module)
            report(True, package)
        except ImportError:
            report(False, f"{package} is missing  ->  pip install -r requirements.txt")

    print("\nOptional packages")
    for module, package, purpose in OPTIONAL_PACKAGES:
        try:
            __import__(module)
            report(True, f"{package}  ({purpose})")
        except ImportError:
            report(False, f"{package} is missing, needed for {purpose}", fatal=False)

    try:
        import spacy
    except ImportError:
        return
    for name in SPACY_MODELS:
        try:
            spacy.load(name)
            report(True, f"spaCy model {name}")
        except OSError:
            report(False, f"spaCy model {name} is missing"
                          f"  ->  python3 -m spacy download {name}", fatal=False)


def check_env(required: bool) -> dict:
    """Read .env. It is only needed to CALL models (a pipeline run, or the Stage 4
    ablation); reading the corpus and computing metrics never touch it, so its
    absence is a warning unless --models was asked for."""
    print("\nConfiguration")
    env_path = ROOT / ".env"
    if not env_path.exists():
        report(False, ".env not found. Needed only to run the pipeline or the ablation"
                      "  ->  cp .env.example .env", fatal=required)
        return {}
    report(True, ".env found")

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        report(False, "cannot read .env: python-dotenv is missing")
        return {}

    values = {}
    for key in ENV_KEYS:
        value = os.getenv(key, "")
        report(bool(value), f"{key} = {value or '(not set)'}")
        if value:
            values[key] = value

    providers = {v.split(":", 1)[0] for v in values.values()}
    if "azure" in providers:
        report(bool(os.getenv("AZURE_ENDPOINT")) and bool(os.getenv("AZURE_API_KEY")),
               "AZURE_ENDPOINT and AZURE_API_KEY set (a model uses provider 'azure')")
    if "foundry" in providers:
        report(bool(os.getenv("FOUNDRY_MISTRAL_ENDPOINT")) and bool(os.getenv("FOUNDRY_MISTRAL_KEY")),
               "FOUNDRY_MISTRAL_* set (a model uses provider 'foundry')")
    if "ollama" in providers:
        report(bool(os.getenv("OLLAMA_BASE_URL")),
               f"OLLAMA_BASE_URL = {os.getenv('OLLAMA_BASE_URL', '(not set)')}")
    return values


def check_data() -> None:
    print("\nData")
    corpus = ROOT / "corpus" / "final"
    expanded = list((ROOT / "output").rglob("*.json"))
    archives = list(corpus.glob("*.json.gz")) if corpus.is_dir() else []
    report(bool(archives), f"corpus/final/ holds {len(archives)} compressed corpus files",
           fatal=False)
    report(bool(expanded), f"output/ holds {len(expanded)} expanded JSON files"
                           + ("" if expanded else "  ->  run scripts/unpack_corpus.py --all"),
           fatal=False)

    inputs = [
        ROOT / "data" / "WikiText" / "wikitext-103-raw.wiki.train.selected_1000.raw",
        ROOT / "data" / "FEVER" / "selected" / "fever_selected.jsonl",
        ROOT / "data" / "Europarl" / "selected",
    ]
    present = [p for p in inputs if p.exists()]
    report(len(present) == len(inputs),
           f"{len(present)}/{len(inputs)} pipeline input subsets present in data/"
           + ("" if len(present) == len(inputs) else "  ->  run scripts/unpack_corpus.py --inputs"),
           fatal=False)


def check_models() -> None:
    print("\nModel endpoints (one short call each)")
    sys.path.insert(0, str(ROOT / "agents"))
    try:
        from agente_wikitext import load_pipeline_config, verify_all_models
    except ImportError as exc:
        report(False, f"cannot import the agent module: {exc}")
        return
    try:
        config = load_pipeline_config()
    except Exception as exc:
        report(False, f"cannot build the pipeline configuration: {exc}")
        return
    report(verify_all_models(config), "every configured model answered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true",
                        help="also send one test call to every configured model")
    args = parser.parse_args()

    print("Checking the environment for the bilingual corpus pipeline")
    check_python()
    check_packages()
    values = check_env(required=args.models)
    check_data()
    if args.models:
        if values:
            check_models()
        else:
            report(False, "skipping the model check: no configuration was loaded")

    print()
    if failures:
        print(f"{failures} blocking problem(s) found. See docs/reproduction.md.")
        sys.exit(1)
    print("Environment looks good.")


if __name__ == "__main__":
    main()
