"""Using chatbot_arena_conversations to create a train/val/test dataset for LLM fingerprinting.

Steps:
  1. Load lmsys/chatbot_arena_conversations (33k battles with each = 1 prompt answered by 2 models).
  2. Split every battle into two rows (model_a, prompt, answer_a) and (model_b, prompt, answer_b).
  3. Keep English, non-flagged, first-turn answers from models in config.FAMILY_MAP.
  4. Tag each prompt with a task domain (coding / math / writing / general) for RQ3.
  5. Balance families by downsampling, then make a train/val/test split grouped by prompt.
"""
import argparse
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

CODING_RE = re.compile(
    r"```|\bdef \w+\(|\b(python|java|javascript|typescript|c\+\+|c#|sql|html|css|code|coding|function|"
    r"program|script|algorithm|bug|compile|regex|bash|json|react|rust|golang|numpy|pandas|api)\b"
)
MATH_RE = re.compile(
    r"\b(calculate|solve|equation|math|integral|derivative|probability|algebra|percent|percentage|"
    r"prime|factorial|arithmetic|geometry)\b|\d+\s*[\+\-\*/\^x]\s*\d+"
)
WRITING_RE = re.compile(
    r"\b(write|story|poem|essay|email|letter|song|lyrics|haiku|rewrite|paraphrase|limerick|"
    r"joke|blog|article|speech|tweet|slogan|rap)\b"
)


def tag_domain(prompt: str) -> str:
    """Keyword rules, checked in priority order (a 'write a python function' prompt counts as coding)."""
    p = prompt.lower()
    if CODING_RE.search(p):
        return "coding"
    if MATH_RE.search(p):
        return "math"
    if WRITING_RE.search(p):
        return "writing"
    return "general"


def load_raw(local_path=None) -> pd.DataFrame:
    if local_path:
        return pd.read_parquet(local_path) if local_path.endswith(".parquet") else pd.read_json(local_path, lines=True)
    from datasets import load_dataset
    return load_dataset("lmsys/chatbot_arena_conversations", split="train").to_pandas()


def is_flagged(mod) -> bool:
    try:
        return bool(mod["flagged"])
    except (TypeError, KeyError, IndexError):
        return False


def battles_to_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in raw.itertuples(index=False):
        if getattr(r, "language", config.LANGUAGE) != config.LANGUAGE:
            continue
        if is_flagged(getattr(r, "openai_moderation", None)):
            continue
        for side in ("a", "b"):
            model = getattr(r, f"model_{side}")
            conv = list(getattr(r, f"conversation_{side}"))
            if model not in config.FAMILY_MAP or len(conv) < 2:
                continue
            if conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
                continue
            rows.append({
                "question_id": r.question_id,
                "tstamp": getattr(r, "tstamp", np.nan),
                "llm_name": model,
                "family": config.FAMILY_MAP[model],
                "llm_input": str(conv[0]["content"]).strip(),
                "llm_output": str(conv[1]["content"]).strip(),
            })
    return pd.DataFrame(rows)


def balance(df: pd.DataFrame, rng) -> pd.DataFrame:
    n = df["family"].value_counts().min()
    if config.MAX_PER_CLASS:
        n = min(n, config.MAX_PER_CLASS)
    parts = [g.sample(n=n, random_state=rng.integers(1e9)) for _, g in df.groupby("family")]
    return pd.concat(parts).sample(frac=1, random_state=config.SEED).reset_index(drop=True)


def grouped_split(df: pd.DataFrame) -> pd.DataFrame:
    """Same prompt text -> same group -> same split. Prevents prompt leakage between train and test."""
    groups = df["llm_input"].str.lower().str.strip().map(lambda s: hashlib.md5(s.encode()).hexdigest())
    df = df.copy()
    df["split"] = "train"
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_FRAC, random_state=config.SEED)
    trval_idx, test_idx = next(gss.split(df, groups=groups))
    df.loc[df.index[test_idx], "split"] = "test"
    trval = df.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=config.VAL_FRAC / (1 - config.TEST_FRAC), random_state=config.SEED)
    _, val_idx = next(gss2.split(trval, groups=groups.iloc[trval_idx]))
    df.loc[trval.index[val_idx], "split"] = "val"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="local .parquet or .jsonl copy of the Arena dataset")
    args = ap.parse_args()
    rng = np.random.default_rng(config.SEED)

    raw = load_raw(args.local)
    print(f"raw battles: {len(raw)}")
    df = battles_to_rows(raw)
    df = df[df["family"].isin(config.FAMILIES)]
    df = df[df["llm_output"].str.len() >= config.MIN_OUTPUT_CHARS]
    df = df.drop_duplicates(subset=["llm_name", "llm_input", "llm_output"])
    print("rows per family before balancing:\n", df["family"].value_counts().to_string())

    df = balance(df, rng)
    df["domain"] = df["llm_input"].map(tag_domain)
    df = grouped_split(df)

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.PROCESSED_CSV, index=False)
    print(f"\nsaved {len(df)} rows -> {config.PROCESSED_CSV}")
    print(pd.crosstab(df["family"], df["split"]).to_string())
    print(pd.crosstab(df["family"], df["domain"]).to_string())
    print("\nmodels kept:\n", df["llm_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
