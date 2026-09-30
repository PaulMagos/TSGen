"""Curated LaTeX tables + significance tests for the thesis, from results/.

    .venv/bin/python scripts/thesis_tables.py --out ../M_Sc__Thesis_Abstract_Paul_M__Magos/Chapters/Experiments/Results/generated

Cells are mean ± std over seeds. In each column the best mean is bold; a dagger marks
entries not significantly different from the best (Welch t-test, p ≥ 0.05, n = seeds).
`comparisons.md` lists the research-question tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tsgen.tracking import flatten_metrics  # noqa: E402

DATASETS = ("Synthetic", "Exchange", "AirQuality")
LOWER_IS_BETTER = True
PRED = {"mae": "MAE", "mse": "MSE", "mase": "MASE", "crps": "CRPS"}
GEN = {"wasserstein": "W$_1$", "mmd_rbf": "MMD$^2$", "acf_distance": "ACF", "cross_corr_distance": "XCorr",
       "discriminative_score": "Disc.", "tstr_ratio": "TSTR", "mem_ratio": "Mem.", "vg_divergence": "VG-div"}
MAIN = ("persistence", "linear", "var", "lstm", "rnn", "mdn", "gtm", "sgtm", "asgtm")
GENERATORS = ("var", "mdn", "gtm", "sgtm", "asgtm")
ABLATIONS = ("gtm", "gtm-chain", "gtm-complete", "gtm-hvg", "gtm-simw", "sgtm", "sgtm-randgraph",
             "asgtm", "asgtm-relative", "asgtm-absolute")
ABLATION_METRICS = {"prediction/mae": "Pred. MAE",
                    **{f"generation/vs_train/{k}": GEN[k] for k in ("wasserstein", "mmd_rbf", "acf_distance",
                                                                    "cross_corr_distance", "vg_divergence")}}
CAPACITY = ("mdn", "mdn-h128", "mdn-h256", "gtm", "gtm-h128", "gtm-h256", "asgtm", "asgtm-h128", "asgtm-h256")
NAMES = {"persistence": "Persistence", "linear": "Linear extrap.", "var": "VAR", "lstm": "LSTM", "rnn": "RNN",
         "mdn": "MDN", "gtm": "GTM", "sgtm": "SGTM", "asgtm": "ASGTM", "locf": "LOCF", "interp": "Interp.$^*$"}


def load(root: Path) -> dict:
    """dataset → config → list of flat metric dicts (one per seed)."""
    table: dict = {}
    for f in sorted(root.glob("*/*/seed*.json")):
        ds, cfg = f.parts[-3], f.parts[-2]
        if ds.startswith("_"):
            continue
        table.setdefault(ds, {}).setdefault(cfg, []).append(flatten_metrics(json.loads(f.read_text())))
    return table


def values(table, ds, cfg, key) -> np.ndarray:
    return np.array([r[key] for r in table.get(ds, {}).get(cfg, []) if key in r])


def label(cfg: str) -> str:
    base, _, rest = cfg.partition("-")
    name = NAMES.get(base, base.upper())
    return f"{name} ({rest})" if rest else name


def fmt(v: np.ndarray) -> str:
    m, s = v.mean(), v.std()
    if abs(m) < 1e-3 and m != 0:
        return f"{m:.1e}" + (f" $\\pm$ {s:.0e}" if s > 0 else "")
    return f"{m:.4f}" + (f" $\\pm$ {s:.4f}" if s > 0 else "")


def column_marks(cols: list[np.ndarray]) -> list[str]:
    """'best', 'tie' (not significantly different from best) or ''."""
    means = [c.mean() if len(c) else np.inf for c in cols]
    best = int(np.argmin(means))
    marks = []
    for i, c in enumerate(cols):
        if not len(c):
            marks.append("")
        elif i == best:
            marks.append("best")
        elif len(c) > 1 and len(cols[best]) > 1 and c.std() + cols[best].std() > 0:
            p = stats.ttest_ind(c, cols[best], equal_var=False).pvalue
            marks.append("tie" if p >= 0.05 else "")
        else:
            marks.append("")
    return marks


def latex_table(table, ds, cfgs, metrics: dict, prefix: str, caption: str, lab: str) -> str:
    cfgs = [c for c in cfgs if c in table.get(ds, {}) and len(values(table, ds, c, prefix + next(iter(metrics))))]
    cols = {k: [values(table, ds, c, prefix + k) for c in cfgs] for k in metrics}
    marks = {k: column_marks(cols[k]) for k in metrics}
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\caption{{{caption}}}", rf"\label{{{lab}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
             "Model & " + " & ".join(metrics.values()) + r" \\", r"\midrule"]
    for i, c in enumerate(cfgs):
        cells = []
        for k in metrics:
            v = cols[k][i]
            if not len(v):
                cells.append("--")
                continue
            s = fmt(v)
            if marks[k][i] == "best":
                s = rf"\textbf{{{s}}}"
            elif marks[k][i] == "tie":
                s += r"$^\dagger$"
            cells.append(s)
        lines.append(label(c) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def imputation_table(table) -> str:
    rows, heads = [], []
    problems = [("Synthetic", "point"), ("Synthetic", "block"), ("Exchange", "point"), ("Exchange", "block"),
                ("AirQuality", "eval_mask")]
    cfgs = ("locf", "interp", "lstm", "rnn", "mdn", "gtm", "sgtm", "asgtm")
    cols = [[values(table, ds, c, f"imputation/{p}/mae") for c in cfgs] for ds, p in problems]
    marks = [column_marks(col) for col in cols]
    for ds, p in problems:
        heads.append(f"{ds} ({p.replace('_', ' ')})")
    for i, c in enumerate(cfgs):
        cells = []
        for j in range(len(problems)):
            v = cols[j][i]
            s = fmt(v) if len(v) else "--"
            if len(v) and marks[j][i] == "best":
                s = rf"\textbf{{{s}}}"
            elif len(v) and marks[j][i] == "tie":
                s += r"$^\dagger$"
            cells.append(s)
        rows.append(label(c) + " & " + " & ".join(cells) + r" \\")
    return "\n".join([r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                      r"\caption{Imputation MAE on held-out entries of the test period (mean $\pm$ std over 5 seeds). "
                      r"$^*$Linear interpolation uses future observations and is a non-causal reference.}",
                      r"\label{tab:new_imputation}", r"\resizebox{\textwidth}{!}{%",
                      r"\begin{tabular}{l" + "c" * len(problems) + "}", r"\toprule",
                      "Model & " + " & ".join(heads) + r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


COMPARISONS = [  # (question, dataset, a, b, metric)
    *[("RQ1 VG vs none", ds, "gtm", "mdn", m) for ds in DATASETS for m in ("wasserstein", "mmd_rbf", "acf_distance", "discriminative_score")],
    *[("RQ1 VG vs chain", ds, "gtm", "gtm-chain", m) for ds in DATASETS for m in ("wasserstein", "acf_distance")],
    *[("RQ1 VG vs complete", ds, "gtm", "gtm-complete", m) for ds in DATASETS for m in ("wasserstein", "acf_distance")],
    *[("RQ2 static graph vs none", ds, "sgtm", "gtm", "cross_corr_distance") for ds in DATASETS],
    *[("RQ2 static vs random graph", ds, "sgtm", "sgtm-randgraph", "cross_corr_distance") for ds in DATASETS],
    *[("RQ2 learned vs static graph", ds, "asgtm", "sgtm", "cross_corr_distance") for ds in DATASETS],
    *[("best neural vs VAR", ds, a, "var", m) for ds, a in (("Synthetic", "gtm"), ("Exchange", "asgtm"), ("AirQuality", "gtm-h128"))
      for m in ("wasserstein", "mmd_rbf", "acf_distance", "cross_corr_distance")],
]


def comparisons(table) -> str:
    out = ["| question | dataset | A | B | metric | mean A | mean B | Welch p | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for q, ds, a, b, m in COMPARISONS:
        va, vb = values(table, ds, a, "generation/vs_train/" + m), values(table, ds, b, "generation/vs_train/" + m)
        if len(va) < 2 or len(vb) < 2:
            continue
        p = stats.ttest_ind(va, vb, equal_var=False).pvalue
        verdict = ("A better" if va.mean() < vb.mean() else "B better") if p < 0.05 else "no significant difference"
        out.append(f"| {q} | {ds} | {a} | {b} | {m} | {va.mean():.4f} | {vb.mean():.4f} | {p:.3g} | {verdict} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = load(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = {
        "prediction.tex": [latex_table(table, ds, MAIN, PRED, "prediction/",
                                       f"{ds}: one-step-ahead prediction on the test period (mean $\\pm$ std over 5 seeds).",
                                       f"tab:new_pred_{ds.lower()}") for ds in DATASETS],
        "imputation.tex": [imputation_table(table)],
        "generation.tex": [latex_table(table, ds, GENERATORS, GEN, "generation/vs_train/",
                                       f"{ds}: generation fidelity against training windows, 500 samples (mean $\\pm$ std over 5 seeds; lower is better for every column).",
                                       f"tab:new_gen_{ds.lower()}") for ds in DATASETS],
        "generation_test.tex": [latex_table(table, ds, GENERATORS, {k: v for k, v in GEN.items() if k not in ("tstr_ratio", "mem_ratio")},
                                            "generation/vs_test/",
                                            f"{ds}: generation fidelity against test-period windows (mean $\\pm$ std over 5 seeds).",
                                            f"tab:new_gentest_{ds.lower()}") for ds in DATASETS],
        "ablations.tex": [latex_table(table, ds, ABLATIONS, ABLATION_METRICS, "",
                                      f"{ds}: ablations (prediction MAE and generation fidelity against training windows).",
                                      f"tab:new_abl_{ds.lower()}") for ds in DATASETS],
        "capacity.tex": [latex_table(table, ds, CAPACITY, {k: GEN[k] for k in ("wasserstein", "mmd_rbf", "acf_distance", "cross_corr_distance")},
                                     "generation/vs_train/", f"{ds}: capacity sweep, parameter budget set by the reference width 64 / 128 / 256.",
                                     f"tab:new_cap_{ds.lower()}") for ds in ("Exchange", "AirQuality")],
    }
    for name, parts in blocks.items():
        (out / name).write_text("% generated by TSGen/scripts/thesis_tables.py — do not edit by hand\n" + "\n".join(parts))
    (out / "comparisons.md").write_text(comparisons(table) + "\n")
    print(f"wrote {', '.join(blocks)} and comparisons.md to {out}")


if __name__ == "__main__":
    main()
