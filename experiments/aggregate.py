"""Comparison plots from the MSE CSVs, saved as PNGs (ported from 'Aggregation.ipynb' cells 5-6)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from loka.report import ExperimentReport, config_from_globals

# ---- CONFIG: input / output locations ----
ROOT        = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
RESULTS_DIR = ROOT / "data/ordered-data/results"   # holds fixed/ and mse_all/
PLOTS_DIR   = ROOT / "data/plots"

# ---- CONFIG: plot settings ----
DPI = 150

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def read_csv_or_warn(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: file not found: {path}")
        return None


def plot_positions_only():
    panels = [
        ("fixed",   r"$MSE_S$", "Evaluated on subset $S$"),
        ("mse_all", r"$MSE_E$", "Evaluated on all data $E$"),
    ]
    datasets = [
        ("k-Center Greedy (no control points)", "k_center_no_S0.csv"),
        ("k-Center Greedy (random control points)", "k_center_rand_S0_25p.csv"),
        ("Herding", "herding.csv"),
        ("Random", "random.csv"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    for ax, (subdir, ylabel, subtitle) in zip(axes, panels):
        for (label, f), color in zip(datasets, colors):  # fixed color per dataset across panels
            df = read_csv_or_warn(RESULTS_DIR / subdir / f)
            if df is None:
                continue
            ax.plot(df["ratio"], df["mse"], "o-", linewidth=1.8, markersize=5,
                    markerfacecolor="white", markeredgewidth=1.5,
                    color=color, label=label)

        ax.set_yscale("log")
        ax.set_xlabel(r"$\frac{|S|}{|E|}$", fontsize=20)
        ax.set_xlim(0.005, 0.11)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_ylim(10**(-1.5), 10**0.5)
        ax.set_title(subtitle, fontsize=18)
        ax.grid(True, which="both", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].tick_params(labelleft=True)

    # one legend entry per label, in dataset order
    by_label = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(l, h)
    order = [label for label, _ in datasets if label in by_label]

    fig.legend([by_label[l] for l in order], order, loc="lower center",
               ncol=max(len(order), 1), fontsize=14, frameon=False)
    fig.suptitle("Model Performance vs Training Subset Size\n"
                 "(Input: white and black positions only, 50 epochs, 1.8M samples)", fontsize=22)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_embeddings():
    results_dir = RESULTS_DIR / "mse_all"

    # (label, positions-only file, key used in the embedding file names)
    methods = [
        ("k-Center Greedy (no control points)",     "k_center_no_S0.csv",       "k_center_no_S0"),
        ("k-Center Greedy (random control points)", "k_center_rand_S0_25p.csv", "k_center_rand_S0"),
        ("Herding",                                 "herding.csv",              "herding"),
    ]
    base_marker = "o"  # positions only
    embeddings = [
        ("Counts",             "with_counts",               "s"),
        ("Near goal",          "with_near_goal",            "^"),
        ("Counts + near goal", "with_counts_and_near_goal", "D"),
    ]

    # (file, color, marker, line style): positions only gets a line, embeddings are markers only
    datasets = []
    for (label, base_file, key), color in zip(methods, colors):
        datasets.append((base_file, color, base_marker, "-"))
        for _, emb_key, marker in embeddings:
            datasets.append((f"{key}_{emb_key}_10p.csv", color, marker, "none"))

    fig, ax = plt.subplots(figsize=(14, 8))

    for f, color, marker, ls in datasets:
        df = read_csv_or_warn(results_dir / f)
        if df is None:
            continue
        ax.plot(df["ratio"], df["mse"], marker=marker, linestyle=ls, linewidth=1.8,
                markersize=6, markerfacecolor="white", markeredgewidth=1.3, color=color)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\frac{|S|}{|E|}$", fontsize=20)
    ax.set_xlim(0.005, 0.11)
    ax.set_ylabel(r"$MSE_E$", fontsize=20)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # two legends: one explains the colors, one the markers
    method_handles = [Patch(color=color, label=label)
                      for (label, _, _), color in zip(methods, colors)]
    emb_handles = [Line2D([], [], color="gray", marker=base_marker, linestyle="-", linewidth=1.8,
                          markersize=8, markerfacecolor="white", markeredgewidth=1.3,
                          label="Positions only")]
    emb_handles += [Line2D([], [], color="gray", marker=marker, linestyle="none",
                           markersize=8, markerfacecolor="white", markeredgewidth=1.3, label=label)
                    for label, _, marker in embeddings]

    fig.legend(handles=method_handles, title="Subset method", loc="lower center",
               bbox_to_anchor=(0.3, 0), ncol=1, fontsize=13, title_fontsize=14, frameon=False)
    fig.legend(handles=emb_handles, title="Features", loc="lower center",
               bbox_to_anchor=(0.75, 0), ncol=2, fontsize=13, title_fontsize=14, frameon=False)

    fig.suptitle("Model Performance vs Training Subset Size\n"
                 "(Evaluated on all data $E$, 50 epochs, 1.8M samples)", fontsize=22)

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    return fig


PLOTS = [
    ("positions_only_mse_S_vs_E.png", plot_positions_only),
    ("embeddings_mse_E.png",          plot_embeddings),
]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    with ExperimentReport('aggregate', config=config_from_globals(globals())) as rep:
        for csv in sorted(RESULTS_DIR.glob("*/*.csv")):  # the inputs of the plots: record which results went in
            rep.add_data(f'{csv.parent.name}/{csv.stem}', csv)
        for fname, plot_fn in PLOTS:
            fig = plot_fn()
            path = PLOTS_DIR / fname
            fig.savefig(path, dpi=DPI)
            plt.close(fig)
            print(f'wrote {path}')
            rep.add_output(path)


if __name__ == '__main__':
    main()
