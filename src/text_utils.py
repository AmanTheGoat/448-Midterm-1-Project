"""Tokenizer, vocabulary and PyTorch Dataset.

Neural nets need numbers, so each text becomes a list of token IDs.
Case, punctuation and newlines are kept as tokens on purpose: formatting
(bullets, line breaks, bold markers) is part of an LLM's style.
"""
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

import config

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return [NL if t == "\n" else t for t in TOKEN_RE.findall(str(text))]


def build_tokens(df, field: str) -> list[list[str]]:
    """field is 'input', 'output' or 'input+output'."""
    max_len = config.MAX_LEN[field]
    if field == "input":
        return [tokenize(t)[:max_len] for t in df["llm_input"]]
    if field == "output":
        return [tokenize(t)[:max_len] for t in df["llm_output"]]
    in_len = config.MAX_LEN["input"]
    seqs = []
    for i, o in zip(df["llm_input"], df["llm_output"]):
        s = tokenize(i)[:in_len] + [SEP] + tokenize(o)
        seqs.append(s[:max_len])
    return seqs


class Vocab:
    def __init__(self, token_lists, max_size=config.VOCAB_SIZE, min_freq=config.MIN_FREQ):
        counts = Counter(t for toks in token_lists for t in toks)
        self.itos = [PAD, UNK, SEP] + [t for t, c in counts.most_common(max_size) if c >= min_freq and t != SEP]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in toks] or [unk]


class TextDataset(Dataset):
    def __init__(self, token_lists, labels, vocab):
        self.ids = [vocab.encode(t) for t in token_lists]
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    """Pad a batch to its longest sequence; also return true lengths (the LSTM needs them)."""
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids])
    width = max(int(lengths.max()), max(config.CNN_KERNELS))  # CNN needs at least kernel-size tokens
    padded = torch.zeros(len(ids), width, dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, : len(x)] = torch.tensor(x)
    return padded, lengths, torch.tensor(labels)
