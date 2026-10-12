"""RQ3: Do LLM fingerprints generalize across task domains?

For each held-out domain D (writing, coding):
  cross-domain : train on all OTHER domains, test on domain D
  in-domain    : train on a same-size random sample that INCLUDES domain D, test on the same domain-D test rows
The gap (in-domain minus cross-domain) shows how much the fingerprint is task-specific.
Domains come from the keyword tagger in src/prepare_data.py. Output text only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.train import load_data, run_experiment, save_json, save_md, splits

HELD_OUT = ["writing", "coding"]


def main():
    df = load_data()
    tr, va, te = splits(df)
    print("domain counts (all splits):\n", pd.crosstab(df["domain"], df["family"]).to_string())
    results = []
    for d in HELD_OUT:
        test_d = te[te.domain == d]
        tr_x, va_x = tr[tr.domain != d], va[va.domain != d]
        tr_in = tr.sample(n=len(tr_x), random_state=config.SEED)    # same size, includes domain d
        va_in = va.sample(n=len(va_x), random_state=config.SEED)
        for m in ("CNN", "LSTM"):
            cross = run_experiment(tr_x, va_x, test_d, "output", m, f"rq3-{m}-cross-{d}")
            indom = run_experiment(tr_in, va_in, test_d, "output", m, f"rq3-{m}-in-{d}")
            for setting, r in (("cross-domain", cross), ("in-domain", indom)):
                r.update(held_out=d, setting=setting)
                results.append(r)
    save_json(results, "rq3.json")

    fams = config.FAMILIES
    md = "| Test domain | Model | Setting | Train rows | Test rows | Acc | Macro-F1 | " + \
         " | ".join(f"F1 {f}" for f in fams) + " |\n|---|---|---|---|---|---|---|" + "---|" * len(fams) + "\n"
    for r in results:
        md += f"| {r['held_out']} | {r['model']} | {r['setting']} | {r['n_train']} | {r['n_test']} | " \
              f"{r['test_acc']:.3f} | {r['test_macro_f1']:.3f} | " + \
              " | ".join(f"{r['per_class_f1'][f]:.3f}" for f in fams) + " |\n"
    save_md(md, "rq3_table.md")

    labels = [f"{d}\n{m}" for d in HELD_OUT for m in ("CNN", "LSTM")]
    get = lambda s: [next(r["test_macro_f1"] for r in results if r["held_out"] == d and r["model"] == m and r["setting"] == s)
                     for d in HELD_OUT for m in ("CNN", "LSTM")]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - 0.2, get("in-domain"), 0.4, label="in-domain (domain seen in training)")
    ax.bar(x + 0.2, get("cross-domain"), 0.4, label="cross-domain (domain unseen)")
    ax.axhline(1 / len(fams), ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(x, labels); ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.set_title("RQ3: generalization to an unseen task domain")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "rq3_bars.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
