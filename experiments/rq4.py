"""RQ4: What characteristics distinguish the LLM families?

Four analyses:
  1. Style feature averages per family (length, diversity, formatting, phrases)  -> what differs
  2. Random forest on ONLY those features + permutation importance              -> which features matter
  3. Top TF-IDF n-grams per family                                              -> which words/symbols
  4. Ablations: retrain CNN/LSTM after removing a signal from the text          -> how much each signal is needed
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

import config
from src.baselines import tfidf_lr, top_ngrams
from src.features import ABLATIONS, FEATURE_NAMES, feature_frame
from src.train import load_data, run_experiment, save_json, save_md, splits

BOX_FEATURES = ["n_words", "n_lines", "type_token_ratio", "bullet_lines", "bold_marks", "first_person_i"]


def feature_stats(df):
    feats = feature_frame(df["llm_output"])
    feats["family"] = df["family"].values
    means = feats.groupby("family")[FEATURE_NAMES].mean().loc[config.FAMILIES].T
    means.round(3).to_csv(config.RESULTS_DIR / "rq4_feature_means.csv")
    md = "| Feature | " + " | ".join(config.FAMILIES) + " |\n|---|" + "---|" * len(config.FAMILIES) + "\n"
    for f, row in means.iterrows():
        md += f"| {f} | " + " | ".join(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}" for v in row) + " |\n"
    save_md(md, "rq4_feature_means.md")

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, f in zip(axes.ravel(), BOX_FEATURES):
        data = [feats.loc[feats.family == fam, f] for fam in config.FAMILIES]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(config.FAMILIES) + 1), config.FAMILIES)
        ax.set_title(f, fontsize=10)
    fig.suptitle("RQ4: style features by family (outliers hidden)")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "rq4_features_box.png", dpi=150); plt.close(fig)


def feature_classifier(tr, te):
    Xtr, Xte = feature_frame(tr["llm_output"]), feature_frame(te["llm_output"])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=config.SEED, n_jobs=-1)
    rf.fit(Xtr, tr["family"])
    pred = rf.predict(Xte)
    imp = permutation_importance(rf, Xte, te["family"], scoring="f1_macro", n_repeats=5,
                                 random_state=config.SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1]
    importance = {FEATURE_NAMES[i]: float(imp.importances_mean[i]) for i in order}

    top = order[:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([FEATURE_NAMES[i] for i in top], imp.importances_mean[top], xerr=imp.importances_std[top])
    ax.set_xlabel("Drop in test macro-F1 when feature is shuffled")
    ax.set_title("RQ4: permutation importance (random forest on style features)")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "rq4_importance.png", dpi=150); plt.close(fig)
    return {"test_acc": accuracy_score(te["family"], pred),
            "test_macro_f1": f1_score(te["family"], pred, average="macro"),
            "importance": importance}


def ablations(tr, va, te):
    out = []
    variants = {"none (full text)": None, **ABLATIONS}
    for name, fn in variants.items():
        t = [d if fn is None else d.assign(llm_output=d["llm_output"].map(fn)) for d in (tr, va, te)]
        for m in ("CNN", "LSTM"):
            r = run_experiment(*t, "output", m, f"rq4-{m}-{name}", verbose=False)
            print(f"  ablation {name:22s} {m:4s} acc {r['test_acc']:.3f} F1 {r['test_macro_f1']:.3f}")
            out.append({"ablation": name, "model": m, "test_acc": r["test_acc"],
                        "test_macro_f1": r["test_macro_f1"], "per_class_f1": r["per_class_f1"]})
    return out


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    tr, va, te = splits(df)

    print("== 1. feature statistics")
    feature_stats(df)

    print("== 2. feature-only classifier")
    fc = feature_classifier(tr, te)
    md = f"Random forest on {len(FEATURE_NAMES)} style features only: acc {fc['test_acc']:.3f}, macro-F1 {fc['test_macro_f1']:.3f}\n\n"
    md += "| Feature | Importance (F1 drop) |\n|---|---|\n" + \
          "".join(f"| {k} | {v:.4f} |\n" for k, v in list(fc["importance"].items())[:12])
    save_md(md, "rq4_feature_classifier.md")

    print("== 3. top n-grams")
    _, vec, clf = tfidf_lr(tr, te, "output", return_model=True)
    grams = top_ngrams(vec, clf)
    md = "| Family | Most indicative tokens / bigrams |\n|---|---|\n" + \
         "".join(f"| {c} | " + ", ".join(f"`{g.replace('|', '/')}`" for g in v) + " |\n" for c, v in grams.items())
    save_md(md, "rq4_top_ngrams.md")

    print("== 4. ablations")
    ab = ablations(tr, va, te)
    full = {a["model"]: a["test_macro_f1"] for a in ab if a["ablation"] == "none (full text)"}
    md = "| Text given to model | CNN F1 | CNN change | LSTM F1 | LSTM change |\n|---|---|---|---|---|\n"
    for name in dict.fromkeys(a["ablation"] for a in ab):
        c = next(a for a in ab if a["ablation"] == name and a["model"] == "CNN")
        l = next(a for a in ab if a["ablation"] == name and a["model"] == "LSTM")
        md += f"| {name} | {c['test_macro_f1']:.3f} | {c['test_macro_f1'] - full['CNN']:+.3f} | " \
              f"{l['test_macro_f1']:.3f} | {l['test_macro_f1'] - full['LSTM']:+.3f} |\n"
    save_md(md, "rq4_ablations.md")

    names = list(dict.fromkeys(a["ablation"] for a in ab))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, m in enumerate(("CNN", "LSTM")):
        ax.bar(x + (k - 0.5) * 0.4, [next(a["test_macro_f1"] for a in ab if a["ablation"] == n and a["model"] == m) for n in names], 0.4, label=m)
    ax.axhline(1 / len(config.FAMILIES), ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("RQ4: performance after removing a signal")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "rq4_ablations.png", dpi=150); plt.close(fig)

    save_json({"feature_classifier": fc, "top_ngrams": grams, "ablations": ab}, "rq4.json")


if __name__ == "__main__":
    main()
