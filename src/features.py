import re

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")

FEATURE_NAMES = [
    "n_chars", "n_words", "n_lines", "avg_word_len", "avg_sent_len", "type_token_ratio",
    "bullet_lines", "numbered_lines", "header_lines", "bold_marks", "code_blocks",
    "exclamations", "questions", "colons", "digits_frac", "upper_frac",
    "starts_with_sure", "as_an_ai", "apology", "first_person_i",
]


def text_features(t: str) -> dict:
    words = WORD_RE.findall(t)
    lw = [w.lower() for w in words]
    lines = t.split("\n")
    n_sent = max(len(SENT_RE.findall(t)), 1)
    low = t.lower()
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "n_lines": len(lines),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        "avg_sent_len": len(words) / n_sent,
        # TTR on the first 100 words so it is not just a proxy for length
        "type_token_ratio": len(set(lw[:100])) / max(len(lw[:100]), 1),
        "bullet_lines": sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines),
        "numbered_lines": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "header_lines": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "bold_marks": t.count("**") // 2,
        "code_blocks": t.count("```") // 2,
        "exclamations": t.count("!"),
        "questions": t.count("?"),
        "colons": t.count(":"),
        "digits_frac": sum(c.isdigit() for c in t) / max(len(t), 1),
        "upper_frac": sum(c.isupper() for c in t) / max(len(t), 1),
        "starts_with_sure": int(bool(re.match(r"\s*(sure|certainly|of course|absolutely)\b", low))),
        "as_an_ai": int("as an ai" in low or "language model" in low),
        "apology": int(bool(re.search(r"\b(sorry|apologi[sz]e)\b", low))),
        "first_person_i": sum(w == "i" for w in lw) / max(len(lw), 1),
    }


def feature_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([text_features(str(t)) for t in texts])[FEATURE_NAMES]


# ablations each removes one kind of signal from the output text

def strip_formatting(t: str) -> str:
    """Remove structure: markdown symbols, list markers, line breaks to one flat paragraph."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"^\s*(#{1,6}|[-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"[*_`#>|]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def truncate_words(t: str, n=50) -> str:
    """Remove length signal: every response is cut to its first n words."""
    return " ".join(str(t).split()[:n])


def lexical_only(t: str) -> str:
    """Remove casing + punctuation + structure and keep only the bag of lowercase words."""
    return " ".join(WORD_RE.findall(str(t).lower()))


def mask_openers(t: str, n=10) -> str:
    """Remove the first n words. """
    return " ".join(str(t).split()[n:])


ABLATIONS = {
    "strip_formatting": strip_formatting,
    "first_50_words": truncate_words,
    "lowercase_words_only": lexical_only,
    "drop_first_10_words": mask_openers,
}
