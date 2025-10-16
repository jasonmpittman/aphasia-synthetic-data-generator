# Cat Rescue Synthetic Prompt Pack

This pack contains severity-conditioned prompt templates and configuration to generate synthetic *Cat Rescue* picture description transcripts across Mild/Moderate/Severe/Very Severe aphasia.

## Contents
- `config/config.yaml` – severity profiles, decoding settings, error-model knobs, QC rules
- `lexicon/core_facts.json` – synonym lists for CIU/core-fact matching
- `prompts/llama3-8b.jsonl`, `prompts/mistral.jsonl`, `prompts/deepseek.jsonl` – per-model chat prompts (4 templates × 4 severities)

## Usage (one model at a time)
1. Pick a model (e.g., Llama 3 8B). Load the corresponding JSONL.
2. For each entry:
   - Render the `system` message.
   - Render the `user` message with `{{max_len}}` replaced from `config.yaml -> severity_profiles[severity].max_len`.
   - Set decoding params from `decoding_profiles[severity]`.
   - Generate 25 samples per template to reach 100 per severity (see `dataset_plan`).

## Suggested JSONL output schema per sample
```
{
  "id": "catrescue_llama3_mild_000123",
  "severity": "mild",
  "model": "llama3-8b",
  "prompt_template_id": "mild_1",
  "scene_version": "catrescue_v1",
  "decoding": { "temperature":0.5, "top_p":0.9, "seed":42 },
  "targets": { "ciu_min":0.60, "ciu_max":0.75, "min_core_facts":8 },
  "transcript": "...",
  "auto_metrics": {
    "ciu": 0.66, "ttr": 0.45, "mlu": 6.2, "wpm_proxy": 68,
    "errors": {"phonemic":2,"semantic":1,"neologism":0,"perseveration":1}
  },
  "passes_qc": true,
  "prompt_hash": "<sha256(system+user)>",
  "seed": 42,
  "run_id": "2025-10-10",
  "version": "1.0.0"
}
```

## CIU/core-fact scoring (sketch)
- Match transcripts against `lexicon/core_facts.json` using case-insensitive substring or lemmatized search.
- CIU% = (# matched core facts) / (severity-specific expected count). Use target bands from `config.yaml`.
- Reject/regenerate if outside CIU band or below `min_core_facts` or if output includes PHI, medical advice, or meta-text.

## License
CC BY 4.0
