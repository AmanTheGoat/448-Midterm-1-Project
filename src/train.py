"""Shared training / evaluation loop used by every RQ, so all experiments are trained the same way.

run_experiment():
  - builds the vocabulary from the TRAIN split only
  - trains with Adam + cross-entropy
  - keeps the epoch with the best validation macro-F1
  - reports accuracy, macro-F1, per-class F1 and the confusion matrix on the test split
"""
import copy
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import config
from src.models import MODELS
from src.text_utils import TextDataset, Vocab, build_tokens, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_CSV, keep_default_na=False)
    return df


def splits(df):
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def _predict(model, loader):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x.to(DEVICE), lengths)
            preds.extend(logits.argmax(1).cpu().tolist())
            gold.extend(y.tolist())
    return np.array(gold), np.array(preds)


def run_experiment(train_df, val_df, test_df, field, model_name, tag, labels=config.FAMILIES, verbose=True):
    set_seed()
    lab2id = {l: i for i, l in enumerate(labels)}
    tr_tok, va_tok, te_tok = (build_tokens(d, field) for d in (train_df, val_df, test_df))
    vocab = Vocab(tr_tok)
    mk = lambda tok, d, shuf: DataLoader(
        TextDataset(tok, d["family"].map(lab2id), vocab),
        batch_size=config.BATCH_SIZE, shuffle=shuf, collate_fn=collate,
    )
    tr_dl, va_dl, te_dl = mk(tr_tok, train_df, True), mk(va_tok, val_df, False), mk(te_tok, test_df, False)

    model = MODELS[model_name](len(vocab), len(labels)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, bad, history = -1, None, 0, []
    t0 = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total = 0.0
        for x, lengths, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        g, p = _predict(model, va_dl)
        val_f1 = f1_score(g, p, average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(tr_dl.dataset),
                        "val_acc": accuracy_score(g, p), "val_f1": val_f1})
        if verbose:
            print(f"  [{tag}] epoch {epoch:2d} loss {history[-1]['train_loss']:.4f} "
                  f"val_acc {history[-1]['val_acc']:.4f} val_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break

    model.load_state_dict(best_state)
    g, p = _predict(model, te_dl)
    res = {
        "tag": tag, "field": field, "model": model_name,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "vocab_size": len(vocab), "epochs_run": len(history), "best_val_f1": best_f1,
        "train_seconds": round(time.time() - t0, 1),
        "test_acc": accuracy_score(g, p),
        "test_macro_f1": f1_score(g, p, average="macro"),
        "per_class_f1": dict(zip(labels, f1_score(g, p, average=None, labels=range(len(labels))).tolist())),
        "confusion": confusion_matrix(g, p, labels=range(len(labels))).tolist(),
        "labels": labels, "history": history,
    }
    if verbose:
        print(f"  [{tag}] TEST acc {res['test_acc']:.4f} macro-F1 {res['test_macro_f1']:.4f}")
        print(classification_report(g, p, labels=range(len(labels)), target_names=labels, digits=3, zero_division=0))
    return res


def save_json(obj, name):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / name, "w") as f:
        json.dump(obj, f, indent=2)


def plot_confusion(res, fname):
    cm = np.array(res["confusion"], dtype=float)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = res["labels"]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{res['model']} ({res['field']})\nacc={res['test_acc']:.3f}, F1={res['test_macro_f1']:.3f}", fontsize=10)
    fig.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / fname, dpi=150)
    plt.close(fig)


def plot_history(results, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        ax[0].plot(ep, [h["train_loss"] for h in r["history"]], marker="o", label=r["model"])
        ax[1].plot(ep, [h["val_f1"] for h in r["history"]], marker="o", label=r["model"])
    ax[0].set_title("Train loss"); ax[1].set_title("Validation macro-F1")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / fname, dpi=150)
    plt.close(fig)


def save_md(text, name):
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / name).write_text(text)
    print(text)
