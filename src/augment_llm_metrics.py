
import re
import json
import math
import pandas as pd
from typing import List, Dict, Optional
import argparse

FILLERS = {
    "um","uh","er","ah","eh","hmm","mm","umm","uhh",
    "like","youknow","you-know","sorta","kinda"
}

def default_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    tokens = re.findall(r"[A-Za-z']+", text)
    return [t for t in tokens if t]

def default_sentence_split(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = re.split(r"[\.!\?\n]+", text)
    return [s.strip() for s in sents if s.strip()]

def default_is_ciu_token(token: str) -> bool:
    t = token.lower()
    if t in FILLERS: return False
    if len(t) <= 1: return False
    return t.isalpha()

def compute_metrics_for_text(text: str,
                             tokenize=default_tokenize,
                             sentence_split=default_sentence_split,
                             is_ciu_token=default_is_ciu_token) -> Dict[str, float]:
    tokens = tokenize(text)
    total_word_count = len(tokens)
    sents = sentence_split(text)
    avg_word_count = (sum(len(tokenize(s)) for s in sents) / max(1, len(sents))) if sents else float(total_word_count)
    num_cius = sum(1 for tok in tokens if is_ciu_token(tok))
    percent_cius = (num_cius / total_word_count * 100.0) if total_word_count > 0 else 0.0
    return {
        "total_word_count": total_word_count,
        "avg_word_count": round(avg_word_count, 2),
        "num_CIUs": num_cius,
        "percent_CIUs": round(percent_cius, 2),
    }

def parse_auto_metrics_field(val) -> dict:
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or not val.strip():
        return {}
    s = val.strip()
    try:
        s_json = s.replace("'", '"')
        return json.loads(s_json)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return {}

POSSIBLE_TEXT_COLS = [
    "transcript", "story", "story_text", "output_text", "generated_text", "response", "text"
]

def locate_text_column(df: pd.DataFrame) -> Optional[str]:
    for c in POSSIBLE_TEXT_COLS:
        if c in df.columns:
            return c
    string_cols = [c for c in df.columns if df[c].dtype == object]
    if not string_cols:
        return None
    lengths = {c: df[c].dropna().astype(str).str.len().mean() for c in string_cols}
    return max(lengths, key=lengths.get) if lengths else None

def enrich_csv(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)
    if "auto_metrics" in df.columns:
        parsed = df["auto_metrics"].apply(parse_auto_metrics_field)
        am = pd.json_normalize(parsed)
        am.columns = [f"auto_metrics.{c}" for c in am.columns]
        df = pd.concat([df, am], axis=1)

    text_col = locate_text_column(df)
    if text_col is None:
        raise ValueError("Could not locate a transcript text column.")

    met = df[text_col].apply(compute_metrics_for_text).apply(pd.Series)
    out = pd.concat([df, met], axis=1)
    out.to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Augment LLM outputs with word/CIU frequency metrics.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (LLM outputs)")
    ap.add_argument("--out", dest="outp", required=True, help="Output CSV (enriched)")
    args = ap.parse_args()
    enrich_csv(args.inp, args.outp)

if __name__ == "__main__":
    main()
