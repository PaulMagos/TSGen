"""Aggregate results/<dataset>/<model>/seed*.json into mean ± std tables.

    python aggregate.py                      # markdown to stdout
    python aggregate.py --latex tables.tex   # also write LaTeX tables
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SECTIONS = ("prediction", "imputation", "generation")
HEADLINE = {  # columns shown unless --all
    "prediction": ("mae", "mse", "mase", "crps"),
    "imputation": ("mae", "mse"),
    "generation": ("wasserstein", "ks", "jsd", "mmd_rbf", "acf_distance", "cross_corr_distance",
                   "discriminative_score", "vg_divergence", "tstr_ratio", "mem_ratio"),
}


def flatten(result: dict) -> dict[str, float]:
    """{'prediction': {'mae': x}} → {'prediction/mae': x}; nested imputation problems included."""
    flat: dict[str, float] = {}

    def walk(prefix: str, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}" if prefix else k, v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix] = float(node)

    for section in SECTIONS:
        if section in result:
            walk(section, result[section])
    return flat


def collect(root: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """dataset → model → metric → values over seeds."""
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(root.glob("*/*/seed*.json")):
        dataset, model = path.parts[-3], path.parts[-2]
        for metric, value in flatten(json.loads(path.read_text())).items():
            table[dataset][model][metric].append(value)
    return table


def fmt(values: list[float]) -> str:
    if not values:
        return "–"
    m = float(np.mean(values))
    digits = ".4f" if abs(m) >= 1e-3 or m == 0 else ".2e"
    return f"{m:{digits}} ± {np.std(values):{digits}} (n={len(values)})" if len(values) > 1 else f"{m:{digits}}"


def _shown(metric: str, section: str, show_all: bool) -> bool:
    return show_all or metric.rsplit("/", 1)[-1] in HEADLINE[section]


def section_tables(table, section: str, show_all: bool = False):
    for dataset, by_model in sorted(table.items()):
        metrics = sorted({m for rows in by_model.values() for m in rows
                          if m.startswith(section + "/") and _shown(m, section, show_all)})
        models = [m for m in sorted(by_model) if any(k.startswith(section + "/") for k in by_model[m])]
        if metrics and models:
            yield dataset, models, metrics


def markdown(table, show_all: bool = False) -> str:
    lines = []
    for section in SECTIONS:
        for dataset, models, metrics in section_tables(table, section, show_all):
            short = [m.split("/", 1)[1] for m in metrics]
            lines += [f"\n### {dataset} — {section}\n", "| model | " + " | ".join(short) + " |",
                      "|---|" + "---|" * len(short)]
            for model in models:
                lines.append(f"| {model} | " + " | ".join(fmt(table[dataset][model].get(m, [])) for m in metrics) + " |")
    return "\n".join(lines)


def latex(table, show_all: bool = False) -> str:
    out = []
    for section in SECTIONS:
        for dataset, models, metrics in section_tables(table, section, show_all):
            short = [m.split("/", 1)[1].replace("_", r"\_") for m in metrics]
            out += [r"\begin{table}[ht]", r"\centering\small",
                    rf"\caption{{{dataset}: {section} (mean $\pm$ std over seeds).}}",
                    r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
                    "Model & " + " & ".join(short) + r" \\", r"\midrule"]
            for model in models:
                cells = [fmt(table[dataset][model].get(m, [])).split(" (n=")[0].replace("±", r"$\pm$")
                         for m in metrics]
                out.append(model.upper() + " & " + " & ".join(cells) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--latex", default=None)
    ap.add_argument("--all", action="store_true", help="every metric, not only the headline columns")
    args = ap.parse_args(argv)
    table = collect(Path(args.results))
    print(markdown(table, args.all))
    if args.latex:
        Path(args.latex).write_text(latex(table, args.all))


if __name__ == "__main__":
    main()
