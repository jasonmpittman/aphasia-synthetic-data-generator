__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import random
import json
import re
from pathlib import Path

def tokenize(sentence: str) -> list[str]:
    """
    Tokenize a sentence into word-like tokens and punctuation.

    Tokens include:
      - Words, including simple contractions and hyphenated forms (e.g., "don't", "fire-fighter")
      - Ellipses ("...")
      - Standalone punctuation

    This tokenizer is used to derive total word counts and CIU counts while
    preserving hyphen and apostrophe segments as single tokens.

    Args:
        sentence: Input sentence or multi-sentence string to tokenize.

    Returns:
        A list of tokens as strings, preserving order.
    """
    return re.findall(r"\w+['-]?\w*|\.\.\.|[^\w\s]", sentence)


def compute_ciu_counts(tokens: list[str]) -> tuple[int, int, float]:
    """
    Compute CIU-related counts from a sequence of tokens per the CIU rules.

    The rules implemented here:
      - Total words: tokens matching [A-Za-z'-]+ (punctuation excluded).
      - CIUs exclude the word "and" and predefined filler words (e.g., "um", "uh", "so", etc.).
      - Percentage CIUs = (num_CIUs / total_word_count) * 100, rounded to 1 decimal.
      - If total_word_count is 0, percent_CIUs is 0.0.

    Note: Higher-level CIU criteria (intelligible, accurate, relevant, informative)
    are approximated via lexical exclusion (fillers, “and”) in this synthetic generator.

    Args:
        tokens: List of tokens returned by `tokenize`.

    Returns:
        A tuple of:
            total_word_count (int): Number of word tokens.
            num_CIUs (int): Number of tokens counted as CIUs under the simplified rules.
            percent_CIUs (float): CIU percentage in [0.0, 100.0].
    """
    FILLER_WORDS = {"and", "um", "uh", "so", "then", "uh-huh", "um-hum", "nope", "yup", "ah", "oh"}
    words = [t for t in tokens if re.match(r"[A-Za-z'-]+", t)]
    total_word_count = len(words)
    ciu_tokens = [w for w in words if w.lower() not in FILLER_WORDS and w.lower() != "and"]
    num_CIUs = len(ciu_tokens)
    percent_CIUs = round((num_CIUs / total_word_count * 100) if total_word_count else 0, 1)
    return total_word_count, num_CIUs, percent_CIUs


def apply_augmentations(sentence: str, severity: str) -> str:
    """
    Apply aphasia-like augmentations to a base sentence according to severity.

    Augmentations include:
      - Word dropping (models agrammatism/telegraphic speech)
      - Paraphasias (semantic substitutions for a limited set of nouns)
      - Filler insertions (disfluencies such as “um”, “uh”, etc.)

    Severity controls the probability of each augmentation:
      - "Mild", "Moderate", "Severe", "Very Severe"

    Content words essential to the picture (e.g., cat, tree, girl, father, ladder,
    fire, department) are protected from dropping to preserve core scene content.

    Args:
        sentence: The base sentence to augment.
        severity: One of {"Mild", "Moderate", "Severe", "Very Severe"}.

    Returns:
        A possibly modified sentence string reflecting the specified severity profile.
    """
    PROBABILITIES = {
        "Mild": {"drop": 0.1, "filler": 0.1, "para": 0.05},
        "Moderate": {"drop": 0.2, "filler": 0.2, "para": 0.1},
        "Severe": {"drop": 0.3, "filler": 0.3, "para": 0.2},
        "Very Severe": {"drop": 0.4, "filler": 0.4, "para": 0.3},
    }
    FILLER_WORDS = {"and", "um", "uh", "so", "then", "uh-huh", "um-hum", "nope", "yup", "ah", "oh"}
    PARAPHS = {"cat": ["dog", "stool"], "girl": ["boy"], "father": ["mother"]}

    probs = PROBABILITIES[severity]
    protected = {"cat", "tree", "girl", "father", "ladder", "fire", "department"}

    words = sentence.split()
    new_words: list[str] = []
    for w in words:
        # Randomly drop non-protected words
        if random.random() < probs["drop"] and w.lower() not in protected:
            continue
        # Paraphasia substitution
        if w.lower() in PARAPHS and random.random() < probs["para"]:
            w = random.choice(PARAPHS[w.lower()])
        new_words.append(w)
        # Disfluency/filler insertion
        if random.random() < probs["filler"]:
            new_words.append(random.choice(list(FILLER_WORDS)))
    return " ".join(new_words)


def generate_dataset(output_dir: str) -> None:
    """
    Generate a synthetic CIU dataset for the cat-rescue picture description task.

    The generator:
      - Produces 10,000 examples evenly split across severities (Mild/Moderate/Severe/Very Severe).
      - Uses sentence-level templates of the cat-rescue scenario.
      - Applies severity-specific augmentations (dropping, paraphasias, fillers).
      - Computes CIU metrics using simplified lexical rules.
      - Writes three JSONL files: train.jsonl, validation.jsonl, and test.jsonl.

    Splits (80/10/10 per severity; total = 8,000/1,000/1,000):
      - train:      2,000 examples per severity
      - validation:   250 examples per severity
      - test:         250 examples per severity

    Each JSONL line has:
      {
        "transcript": str,
        "severity": str,
        "total_word_count": int,
        "avg_word_count": float,
        "num_CIUs": int,
        "percent_CIUs": float,
        "split": "train" | "validation" | "test"
      }

    Args:
        output_dir: Path to the directory where split files will be written.

    Returns:
        None. Files are written to disk:
          - <output_dir>/train.jsonl
          - <output_dir>/validation.jsonl
          - <output_dir>/test.jsonl

    Raises:
        OSError: If the output directory cannot be created or files cannot be written.
        ValueError: If an unknown severity label is encountered (should not happen with defaults).
    """
    NUM_EXAMPLES_PER_SEVERITY = 2500
    SEVERITIES = ["Mild", "Moderate", "Severe", "Very Severe"]
    SPLIT_COUNTS = {"train": 2000, "validation": 250, "test": 250}
    BASE_SENTENCES = [
        "The cat is stuck up the tree",
        "The little girl called her father to use a ladder",
        "But the ladder fell",
        "They called the fire department",
        "The fire department came to rescue the cat",
    ]

    random.seed(42)
    dataset: list[dict] = []

    for severity in SEVERITIES:
        if severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {severity}")
        for i in range(NUM_EXAMPLES_PER_SEVERITY):
            sentences = [apply_augmentations(s, severity) for s in BASE_SENTENCES]
            transcript = ". ".join(sentences) + "."
            tokens = tokenize(transcript)
            total_wc, num_ciu, pct_ciu = compute_ciu_counts(tokens)
            avg_wc = round(total_wc / len(sentences), 2)

            if i < SPLIT_COUNTS["train"]:
                split = "train"
            elif i < SPLIT_COUNTS["train"] + SPLIT_COUNTS["validation"]:
                split = "validation"
            else:
                split = "test"

            dataset.append({
                "transcript": transcript,
                "severity": severity,
                "total_word_count": total_wc,
                "avg_word_count": avg_wc,
                "num_CIUs": num_ciu,
                "percent_CIUs": pct_ciu,
                "split": split,
            })

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "validation": [], "test": []}
    for entry in dataset:
        splits[entry["split"]].append(entry)

    for split_name, entries in splits.items():
        with open(out_path / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_dataset('data/procedural/')