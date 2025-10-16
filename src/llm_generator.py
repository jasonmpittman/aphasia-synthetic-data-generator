#!/usr/bin/env python3
__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
LLM Cat Rescue Synthetic Transcript Generator

This script loads the Cat Rescue prompt pack (config, core-facts lexicon, and per-model prompts),
generates synthetic transcripts one model at a time, computes CIU/core-facts metrics, applies QC,
and saves results to JSONL/CSV.

Backends supported:
- transformers_local: Run Hugging Face `transformers` models locally (Apple Silicon M-series supported).
- openai_compatible: Use any OpenAI-compatible endpoint (vLLM, hosted gateway).
- dummy: Pipeline smoke test with placeholder outputs.

Usage:
    python cat_rescue_runner.py \
        --prompt-pack-dir /path/to/cat_rescue_synthetic_promptpack \
        --output-dir /path/to/outputs \
        --backend transformers_local \
        --prompt-model llama3-8b \
        --hf-model-id meta-llama/Meta-Llama-3-8B-Instruct \
        --samples-per-severity 8

See `python llm_generator.py --help` for all options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

# Optional imports guarded by backend use
try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

# Transformers lazy imports
AutoTokenizer = None
AutoModelForCausalLM = None
pipeline = None
torch = None


@dataclass
class Config:
    """Holds loaded configuration from config.yaml and derived fields."""
    data: Dict

    @property
    def scene_version(self) -> str:
        """Return the scene version string (e.g., 'catrescue_v1')."""
        return self.data.get("scene_version", "catrescue_v1")

    def severity_profile(self, severity: str) -> Dict:
        """Return the dict for a given severity (mild/moderate/severe/very_severe)."""
        return self.data["severity_profiles"][severity]

    def decoding_profile(self, severity: str) -> Dict:
        """Return decoding settings dict for a severity."""
        return self.data["decoding_profiles"][severity]

    @property
    def qc(self) -> Dict:
        """Return QC settings dictionary."""
        return self.data.get("qc", {})


def load_yaml(path: str) -> Dict:
    """Load a YAML file and return as a Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict:
    """Load a JSON file and return as a Python dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Sequence[Dict], path: str) -> None:
    """Save a sequence of dictionaries to a JSONL file."""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def render_user_prompt(template: str, severity: str, config: Config) -> str:
    """Render a user prompt template with severity-specific substitutions."""
    max_len = config.severity_profile(severity)["max_len"]
    return template.replace("{{max_len}}", str(max_len))


def select_templates_by_severity(prompts: Sequence[Dict], severity: str) -> List[Dict]:
    """Filter prompt templates by severity."""
    return [p for p in prompts if p["severity"] == severity]


def ciu_corefacts_score(text: str, core_dict: Dict[str, List[str]]) -> Set[str]:
    """Compute matched core-fact keys by simple substring matching."""
    t = text.lower()
    matched_keys: Set[str] = set()
    for key, variants in core_dict.items():
        for v in variants:
            if v.lower() in t:
                matched_keys.add(key)
                break
    return matched_keys


def compute_metrics(text: str, severity: str, matched_keys: Set[str], config: Config) -> Dict:
    """Compute simple metrics: CIU proxy, TTR, MLU proxy, and error counts."""
    prof = config.severity_profile(severity)
    expected = max(prof["min_core_facts"], 1)
    ciu = len(matched_keys) / expected
    tokens = len(text.strip().split())
    types = len(set(re.findall(r"[A-Za-z]+", text.lower())))
    ttr = types / max(tokens, 1)
    sents = max(len(re.findall(r"[.!?]+", text)), 1)
    mlu = tokens / sents
    errors = {
        "phonemic": len(re.findall(r"\\b\\w{1,2}-\\w+", text)),
        "semantic": 0,
        "neologism": len(re.findall(r"\\b[A-Z]{3,}\\b", text)),
        "perseveration": text.count("[rep]"),
    }
    return {
        "ciu": round(ciu, 3),
        "ttr": round(ttr, 3),
        "mlu": round(mlu, 3),
        "wpm_proxy": tokens,
        "errors": errors,
    }


def passes_qc(text: str, severity: str, matched_keys: Set[str], metrics: Dict, config: Config) -> bool:
    """Decide if the transcript passes QC based on config-controlled rules."""
    prof = config.severity_profile(severity)
    ciu_min, ciu_max = prof["ciu_range"]
    in_band = (metrics["ciu"] >= ciu_min) and (metrics["ciu"] <= ciu_max + 1e-6)
    enough_core = len(matched_keys) >= prof["min_core_facts"]
    bad_phi = bool(re.search(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", text))
    return in_band and (enough_core or severity in ["severe", "very_severe"]) and not bad_phi


def call_openai_compatible(system_prompt: str, user_prompt: str, severity: str, config: Config,
                           base_url: str, api_key: str, model_name: str) -> str:
    """Call an OpenAI-compatible chat endpoint to get a completion."""
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install `openai>=1.0.0`.")
    client = OpenAI(base_url=base_url, api_key=api_key)
    dec = config.decoding_profile(severity)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=dec.get("temperature", 0.7),
        top_p=dec.get("top_p", 0.9),
        presence_penalty=dec.get("presence_penalty", 0.0),
        frequency_penalty=dec.get("frequency_penalty", 0.0),
        max_tokens=config.qc.get("max_tokens", 220),
    )
    return resp.choices[0].message.content.strip()


TRANSFORMERS_INITIALIZED = False
TOKENIZER = None
MODEL = None
PIPE = None


def _lazy_import_transformers() -> None:
    """Import transformers and torch lazily to keep non-transformers runs lightweight."""
    global AutoTokenizer, AutoModelForCausalLM, pipeline, torch
    if AutoTokenizer is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as _pipeline  # type: ignore
        import torch as _torch  # type: ignore
        globals()["AutoTokenizer"] = AutoTokenizer
        globals()["AutoModelForCausalLM"] = AutoModelForCausalLM
        globals()["pipeline"] = _pipeline
        globals()["torch"] = _torch


def init_transformers(hf_model_id: str, hf_token: Optional[str] = None) -> None:
    """Initialize a local transformers pipeline for text-generation/chat models.

    Tries to use Apple Silicon 'mps' if available, otherwise CPU.
    """
    global TRANSFORMERS_INITIALIZED, TOKENIZER, MODEL, PIPE
    _lazy_import_transformers()

    auth_kwargs = {"use_auth_token": hf_token} if hf_token else {}
    TOKENIZER = AutoTokenizer.from_pretrained(hf_model_id, **auth_kwargs)

    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    MODEL = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=getattr(torch, "float16", None) or getattr(torch, "bfloat16", None),
        low_cpu_mem_usage=True,
    )

    PIPE = pipeline(
        "text-generation",
        model=MODEL,
        tokenizer=TOKENIZER,
        device=0 if device == "mps" else -1,
        max_new_tokens=220,
        do_sample=True,
    )
    TRANSFORMERS_INITIALIZED = True


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    """Apply a tokenizer's chat template; fallback to a simple format."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT] "


def call_transformers_local(system_prompt: str, user_prompt: str, severity: str, config: Config) -> str:
    """Generate text using a local transformers pipeline."""
    global TRANSFORMERS_INITIALIZED, TOKENIZER, PIPE
    if not TRANSFORMERS_INITIALIZED or TOKENIZER is None or PIPE is None:
        raise RuntimeError("Transformers pipeline not initialized. Call init_transformers(...) first.")

    dec = config.decoding_profile(severity)
    prompt_text = apply_chat_template(TOKENIZER, system_prompt, user_prompt)

    outputs = PIPE(
        prompt_text,
        temperature=dec.get("temperature", 0.7),
        top_p=dec.get("top_p", 0.9),
        repetition_penalty=1.0 + dec.get("frequency_penalty", 0.0),
        num_return_sequences=1,
        eos_token_id=TOKENIZER.eos_token_id,
    )
    text = outputs[0]["generated_text"]
    if prompt_text in text:
        text = text[len(prompt_text):]
    return text.split("</s>")[0].strip()


def call_dummy(system_prompt: str, user_prompt: str, severity: str, config: Config) -> str:
    """Return a crude, severity-shaped placeholder transcript for pipeline testing."""
    hes = config.data["error_model"]["hesitation_tokens"]
    base_map = {
        "mild": "The girl watches her cat stuck in the tree while her dad is up there after the ladder fell. A dog barks and firefighters run from the truck with a ladder to help.",
        "moderate": "The, um, girl is by a little bike and the, uh, cat is up there in the tree. The dad went up the thing for climbing, ladder, and now it's down. People from the fire truck come with another ladder to help.",
        "severe": "Girl… trike. Cat up tree. Dad up. Ladder down… um… dog bark. Fire men coming with ladder. Get cat, get dad.",
        "very_severe": f"Uh… cat… tree. Dad… up. {hes[0]} … ladder… help.",
    }
    prof = config.severity_profile(severity)
    return " ".join(base_map[severity].split()[: prof["max_len"]])


def generate(
    prompt_pack_dir: str,
    output_dir: str,
    backend: str,
    prompt_model: str,
    samples_per_severity: int,
    severities: Sequence[str],
    hf_model_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Run the generation loop and save JSONL and CSV outputs."""
    config_path = os.path.join(prompt_pack_dir, "config", "config.yaml")
    core_facts_path = os.path.join(prompt_pack_dir, "lexicon", "core_facts.json")
    prompts_path = os.path.join(prompt_pack_dir, "prompts", f"{prompt_model}.jsonl")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not os.path.exists(core_facts_path):
        raise FileNotFoundError(f"Missing core facts: {core_facts_path}")
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    cfg = Config(load_yaml(config_path))
    core = load_json(core_facts_path)
    prompts = load_jsonl(prompts_path)

    if backend == "transformers_local":
        if not hf_model_id:
            raise ValueError("--hf-model-id is required for transformers_local backend")
        init_transformers(hf_model_id=hf_model_id, hf_token=hf_token)
    elif backend == "openai_compatible":
        if not (base_url and api_key and model_name):
            raise ValueError("--base-url, --api-key, and --model-name are required for openai_compatible backend")
    elif backend == "dummy":
        pass
    else:
        raise ValueError(f"Unknown backend: {backend}")

    rows: List[Dict] = []
    run_id = time.strftime("%Y%m%d-%H%M%S")

    for severity in severities:
        subset = select_templates_by_severity(prompts, severity)
        if not subset:
            raise ValueError(f"No templates for severity={severity}")
        need = samples_per_severity
        i = 0
        pbar = tqdm(total=need, desc=f"Generating {severity}")
        while i < need:
            for tpl in subset:
                if i >= need:
                    break
                system = tpl["system"]
                user = render_user_prompt(tpl["user"], severity, cfg)
                prompt_hash = hashlib.sha256((system + user).encode()).hexdigest()[:16]

                if backend == "transformers_local":
                    text = call_transformers_local(system, user, severity, cfg)
                elif backend == "openai_compatible":
                    text = call_openai_compatible(system, user, severity, cfg, base_url, api_key, model_name)  # type: ignore[arg-type]
                else:
                    text = call_dummy(system, user, severity, cfg)

                matched = ciu_corefacts_score(text, core)
                metrics = compute_metrics(text, severity, matched, cfg)
                ok = passes_qc(text, severity, matched, metrics, cfg)

                row = {
                    "id": f"catrescue_{prompt_model}_{severity}_{run_id}_{i:04d}",
                    "severity": severity,
                    "model": prompt_model,
                    "prompt_template_id": tpl["prompt_template_id"],
                    "scene_version": cfg.scene_version,
                    "decoding": cfg.decoding_profile(severity),
                    "targets": {
                        "ciu_min": cfg.severity_profile(severity)["ciu_range"][0],
                        "ciu_max": cfg.severity_profile(severity)["ciu_range"][1],
                        "min_core_facts": cfg.severity_profile(severity)["min_core_facts"],
                    },
                    "transcript": text,
                    "auto_metrics": metrics,
                    "matched_corefacts": sorted(list(matched)),
                    "passes_qc": ok,
                    "prompt_hash": prompt_hash,
                    "seed": None,
                    "run_id": run_id,
                    "version": cfg.data.get("version", 1),
                }
                rows.append(row)
                i += 1
                pbar.update(1)
                if i >= need:
                    break
        pbar.close()

    ensure_dir(output_dir)
    jsonl_path = os.path.join(output_dir, f"catrescue_{prompt_model}_{run_id}.jsonl")
    csv_path = os.path.join(output_dir, f"catrescue_{prompt_model}_{run_id}.csv")

    save_jsonl(rows, jsonl_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return jsonl_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the local runner."""
    p = argparse.ArgumentParser(description="Cat Rescue Synthetic Transcripts — Local Runner")
    p.add_argument("--prompt-pack-dir", required=True, help="Path to prompt pack root (contains config/, lexicon/, prompts/)")
    p.add_argument("--output-dir", required=True, help="Directory to write JSONL/CSV outputs")
    p.add_argument("--backend", choices=["transformers_local", "openai_compatible", "dummy"], default="transformers_local", help="Backend to use")
    p.add_argument("--prompt-model", choices=["llama3.1-8b", "mistral", "deepseek"], default="llama3.1-8b", help="Which prompts file to use")

    p.add_argument("--samples-per-severity", type=int, default=8, help="Number of samples per severity")
    p.add_argument("--severities", type=str, default="mild,moderate,severe,very_severe", help="Comma-separated severities to generate")

    p.add_argument("--base-url", type=str, help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1)")
    p.add_argument("--api-key", type=str, help="OpenAI-compatible API key/token")
    p.add_argument("--model-name", type=str, help="OpenAI-compatible model name")

    p.add_argument("--hf-model-id", type=str, help="HF model id (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
    p.add_argument("--hf-token", type=str, default=None, help="HF token for gated models (optional)")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for local generation."""
    args = parse_args(argv)
    severities = [s.strip() for s in args.severities.split(",") if s.strip()]

    jsonl_path, csv_path = generate(
        prompt_pack_dir=args.prompt_pack_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        prompt_model=args.prompt_model,
        samples_per_severity=args.samples_per_severity,
        severities=severities,
        hf_model_id=args.hf_model_id,
        hf_token=args.hf_token,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    print("Wrote:", jsonl_path)
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
