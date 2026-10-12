"""RQ1: Can we identify which LLM family generated a response, using LLM_output only?

Trains CNN and LSTM on the output text, plus a TF-IDF baseline for reference.
"""
import config
from src.baselines import tfidf_lr
from src.train import load_data, plot_confusion, plot_history, run_experiment, save_json, save_md, splits


def main():
    tr, va, te = splits(load_data())
    print(f"RQ1  train={len(tr)} val={len(va)} test={len(te)}  chance={1/len(config.FAMILIES):.3f}")
    results = []
    for m in ("CNN", "LSTM"):
        r = run_experiment(tr, va, te, "output", m, f"rq1-{m}")
        plot_confusion(r, f"rq1_cm_{m}.png")
        results.append(r)
    plot_history(results, "rq1_curves.png", "RQ1 training (output only)")
    base = tfidf_lr(tr, te, "output")
    save_json({"neural": results, "baseline": base}, "rq1.json")

    fams = config.FAMILIES
    md = "| Model | Accuracy | Macro-F1 | " + " | ".join(f"F1 {f}" for f in fams) + " | Epochs |\n"
    md += "|---|---|---|" + "---|" * len(fams) + "---|\n"
    for r in results:
        md += f"| {r['model']} | {r['test_acc']:.3f} | {r['test_macro_f1']:.3f} | " + \
              " | ".join(f"{r['per_class_f1'][f]:.3f}" for f in fams) + f" | {r['epochs_run']} |\n"
    md += f"| TF-IDF + LR (extra) | {base['test_acc']:.3f} | {base['test_macro_f1']:.3f} |" + " - |" * len(fams) + " - |\n"
    md += f"\nChance level: {1/len(fams):.3f}\n"
    save_md(md, "rq1_table.md")


if __name__ == "__main__":
    main()
