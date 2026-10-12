"""RQ2: Does the user's prompt help identify the LLM?

Compares three inputs to the same models: input only, output only, input + output.
Also saves evidence for *why* input-only might work:
  - family share over time (Arena added/retired models at different dates, and prompt
    topics drift over time, so the prompt can leak "when" and therefore "which model")
  - how many prompts were answered by more than one family (same prompt -> no signal)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.baselines import tfidf_lr
from src.train import load_data, plot_confusion, run_experiment, save_json, save_md, splits

FIELDS = ["input", "output", "input+output"]


def time_drift(df):
    if "tstamp" not in df or pd.to_numeric(df["tstamp"], errors="coerce").isna().all():
        return None
    d = df.assign(week=pd.to_datetime(pd.to_numeric(df["tstamp"], errors="coerce"), unit="s").dt.to_period("W").dt.start_time)
    share = pd.crosstab(d["week"], d["family"], normalize="index")[config.FAMILIES]
    ax = share.plot.area(figsize=(7, 3.2), alpha=0.85)
    ax.set_ylabel("Share of rows"); ax.set_xlabel("Week"); ax.set_title("Family share over time (Arena timestamps)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "rq2_time.png", dpi=150); plt.close()
    share.index = share.index.astype(str)
    return share.round(3).to_dict(orient="index")


def main():
    df = load_data()
    tr, va, te = splits(df)
    results, base = [], []
    for field in FIELDS:
        for m in ("CNN", "LSTM"):
            r = run_experiment(tr, va, te, field, m, f"rq2-{m}-{field}")
            plot_confusion(r, f"rq2_cm_{m}_{field.replace('+', '_')}.png")
            results.append(r)
        base.append(tfidf_lr(tr, te, field))

    fam_per_prompt = df.groupby(df["llm_input"].str.lower().str.strip())["family"].nunique()
    stats = {
        "prompts_total": int(len(fam_per_prompt)),
        "prompts_with_2plus_families": int((fam_per_prompt > 1).sum()),
        "avg_prompt_words_by_family": df.assign(w=df["llm_input"].str.split().str.len())
                                        .groupby("family")["w"].mean().round(1).to_dict(),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stats["family_share_by_week"] = time_drift(df)
    save_json({"neural": results, "baseline": base, "stats": stats}, "rq2.json")

    md = "| Input to classifier | CNN acc | CNN F1 | LSTM acc | LSTM F1 | TF-IDF acc | TF-IDF F1 |\n|---|---|---|---|---|---|---|\n"
    for f, b in zip(FIELDS, base):
        c, l = [r for r in results if r["field"] == f]
        md += f"| {f} | {c['test_acc']:.3f} | {c['test_macro_f1']:.3f} | {l['test_acc']:.3f} | {l['test_macro_f1']:.3f} | {b['test_acc']:.3f} | {b['test_macro_f1']:.3f} |\n"
    md += (f"\nChance: {1/len(config.FAMILIES):.3f}. Prompts answered by 2+ families: "
           f"{stats['prompts_with_2plus_families']} of {stats['prompts_total']}.\n"
           f"Avg prompt length (words) by family: {stats['avg_prompt_words_by_family']}\n")
    save_md(md, "rq2_table.md")


if __name__ == "__main__":
    main()
