#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure generator for the N20 true 3D task battery.

This script recreates the figures using the model names used in the paper:

  Autonomy-layer–LMM
  Core-only
  LMM-only
  Rule-based

Input directory must contain:
  episode_summary.csv

Optional but used when available:
  consultation_events.csv
  decision_override_events.csv
  robustness_sensitivity/task_delta_robustness_summary.csv
  robustness_sensitivity/weight_perturbation_task_summary.csv
  robustness_sensitivity/rank_stability_summary.csv

Outputs:
  fig_1_task_agent_autonomy_proper.png
  fig_2_task_agent_system_sovereignty.png
  fig_3_task_agent_heteronomy.png
  fig_4_autonomy_layer_lmm_advantage.png
  fig_5_selective_lmm_use.png
  fig_6_sensitivity_baseline_equal_weight.png        if robustness CSV exists
  fig_7_coefficient_perturbation_robustness.png      if robustness CSV exists
  publication_figure_values.xlsx-like CSV files are not created; all tabular values are CSV.
  figure_generation_report.txt

Example:
cd ~/Desktop

python3 -u recreate_n20_publication_figures_v2.py \
  --indir ~/Desktop/DARCA_TRUE_3D_TASK_BATTERY_V1_N20 \
  --outdir ~/Desktop/DARCA_TRUE_3D_TASK_BATTERY_V1_N20/publication_figures \
  --bootstrap 10000 \
  2>&1 | tee ~/Desktop/recreate_n20_publication_figures_v2.log
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Fixed labels used in the manuscript
# =============================================================================

TASK_ORDER = [
    "viability",
    "semantic_cue",
    "delayed_memory",
    "adversarial_cue",
    "exploration_recovery",
]

TASK_LABELS = {
    "viability": "Viability",
    "semantic_cue": "Semantic cue",
    "delayed_memory": "Delayed memory",
    "adversarial_cue": "Adversarial cue",
    "exploration_recovery": "Exploration–recovery",
}

ARM_ALIASES = {
    "DARCA_AUTONOMY_LAYER": "AUTONOMY_LAYER_LMM",
    "DARCA_ONLY": "CORE_ONLY",
    "LMM_ONLY_AGENT": "LMM_ONLY",
    "RULE_BASED_AGENT": "RULE_BASED",
    "AUTONOMY_LAYER_LMM": "AUTONOMY_LAYER_LMM",
    "CORE_ONLY": "CORE_ONLY",
    "LMM_ONLY": "LMM_ONLY",
    "RULE_BASED": "RULE_BASED",
    "Autonomy-layer–LMM": "AUTONOMY_LAYER_LMM",
    "Core-only": "CORE_ONLY",
    "LMM-only": "LMM_ONLY",
    "Rule-based": "RULE_BASED",
}

ARM_ORDER = [
    "AUTONOMY_LAYER_LMM",
    "CORE_ONLY",
    "LMM_ONLY",
    "RULE_BASED",
]

ARM_LABELS = {
    "AUTONOMY_LAYER_LMM": "Autonomy-layer–LMM",
    "CORE_ONLY": "Core-only",
    "LMM_ONLY": "LMM-only",
    "RULE_BASED": "Rule-based",
}


# =============================================================================
# Utilities
# =============================================================================

class Logger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.log_path = outdir / "figure_generation.log"
        self.log_path.write_text("N20 publication figure generation log\n" + "=" * 80 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.2f}s] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_csv(path: Path, logger: Logger, required: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        logger.log(f"Optional file not found: {path}")
        return None
    logger.log(f"Reading {path}")
    df = pd.read_csv(path)
    logger.log(f"  rows={len(df)}, cols={len(df.columns)}")
    return df


def require_columns(df: pd.DataFrame, required: List[str], filename: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}\nExisting columns: {list(df.columns)}")


def normalize_arms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["arm_original"] = out["arm"].astype(str)
    out["arm"] = out["arm_original"].map(lambda x: ARM_ALIASES.get(x, x))
    return out


def ordered_existing(values: List[str], order: List[str]) -> List[str]:
    values = list(dict.fromkeys(values))
    s = set(values)
    out = [x for x in order if x in s]
    out.extend(sorted(s - set(out)))
    return out


def task_labels(tasks: List[str]) -> List[str]:
    return [TASK_LABELS.get(t, t) for t in tasks]


def arm_labels(arms: List[str]) -> List[str]:
    return [ARM_LABELS.get(a, a) for a in arms]


def safe_float(x, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean_matrix(df: pd.DataFrame, field: str, tasks: List[str], arms: List[str]) -> np.ndarray:
    M = np.full((len(tasks), len(arms)), np.nan, dtype=float)
    for i, task in enumerate(tasks):
        for j, arm in enumerate(arms):
            vals = df.loc[(df["task"] == task) & (df["arm"] == arm), field].dropna().astype(float)
            if len(vals):
                M[i, j] = float(vals.mean())
    return M


def annotate_heatmap(ax, M: np.ndarray, digits: int = 2) -> None:
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = "NA" if not np.isfinite(M[i, j]) else f"{M[i, j]:.{digits}f}"
            ax.text(j, i, txt, ha="center", va="center")


def paired_delta(df: pd.DataFrame, field: str, task: str,
                 arm_a: str = "AUTONOMY_LAYER_LMM",
                 arm_b: str = "CORE_ONLY") -> np.ndarray:
    sub = df.loc[df["task"] == task, ["episode", "arm", field]].copy()
    wide = sub.pivot_table(index="episode", columns="arm", values=field, aggfunc="mean")
    if arm_a not in wide.columns or arm_b not in wide.columns:
        return np.array([], dtype=float)
    return (wide[arm_a] - wide[arm_b]).dropna().astype(float).to_numpy()


def bootstrap_ci(vals: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if len(vals) == 1 or n_boot <= 0:
        return mean, sd, mean, mean, float(mean > 0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (
        mean,
        sd,
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
        float(np.mean(boots > 0)),
    )


def save_fig(fig, outpath: Path) -> None:
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure functions
# =============================================================================

def plot_heatmap(df: pd.DataFrame, field: str, tasks: List[str], arms: List[str],
                 title: str, cbar_label: str, outpath: Path,
                 vmin: float = 0.0, vmax: float = 1.0) -> pd.DataFrame:
    M = mean_matrix(df, field, tasks, arms)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arm_labels(arms), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(task_labels(tasks))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    annotate_heatmap(ax, M, digits=2)
    fig.tight_layout()
    save_fig(fig, outpath)

    rows = []
    for i, task in enumerate(tasks):
        for j, arm in enumerate(arms):
            rows.append({
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "arm": arm,
                "arm_label": ARM_LABELS.get(arm, arm),
                field: M[i, j],
            })
    return pd.DataFrame(rows)


def plot_advantage(df: pd.DataFrame, tasks: List[str], outpath: Path,
                   n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx, task in enumerate(tasks):
        vals = paired_delta(df, "autonomy_proper_index", task)
        mean, sd, lo, hi, p_pos = bootstrap_ci(vals, n_boot=n_boot, seed=seed + idx * 17)
        rows.append({
            "task": task,
            "task_label": TASK_LABELS.get(task, task),
            "n_paired_episodes": int(len(vals)),
            "delta_mean": mean,
            "delta_sd": sd,
            "ci_low": lo,
            "ci_high": hi,
            "bootstrap_probability_positive": p_pos,
            "paired_deltas": ";".join(f"{x:.10g}" for x in vals),
        })
    d = pd.DataFrame(rows)

    x = np.arange(len(d))
    y = d["delta_mean"].to_numpy(dtype=float)
    err_low = y - d["ci_low"].to_numpy(dtype=float)
    err_high = d["ci_high"].to_numpy(dtype=float) - y
    yerr = np.vstack([err_low, err_high])

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["task_label"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Δ autonomy proper vs Core-only")
    ax.set_title("Autonomy-layer–LMM advantage over Core-only by task")
    fig.tight_layout()
    save_fig(fig, outpath)

    return d


def plot_selective_lmm_use(df: pd.DataFrame, tasks: List[str], outpath: Path) -> pd.DataFrame:
    rows = []
    for task in tasks:
        sub = df.loc[(df["task"] == task) & (df["arm"] == "AUTONOMY_LAYER_LMM")].copy()
        rows.append({
            "task": task,
            "task_label": TASK_LABELS.get(task, task),
            "lmm_calls_mean": float(sub["lmm_calls"].mean()) if "lmm_calls" in sub.columns and len(sub) else np.nan,
            "lmm_decision_fraction_mean": float(sub["lmm_external_decision_fraction"].mean()) if "lmm_external_decision_fraction" in sub.columns and len(sub) else np.nan,
            "core_decision_fraction_mean": float(sub["darca_internal_decision_fraction"].mean()) if "darca_internal_decision_fraction" in sub.columns and len(sub) else np.nan,
        })
    d = pd.DataFrame(rows)

    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.bar(x, d["lmm_decision_fraction_mean"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(d["task_label"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Final action fraction from LMM semantic hints")
    ax.set_title("Selective LMM use under autonomy-layer control")
    fig.tight_layout()
    save_fig(fig, outpath)

    return d


def plot_sensitivity_baseline_equal(robust_df: pd.DataFrame, tasks: List[str], outpath: Path) -> Optional[pd.DataFrame]:
    required = {"task", "index", "delta_mean", "ci_low", "ci_high"}
    if robust_df is None or not required.issubset(set(robust_df.columns)):
        return None

    b = robust_df.loc[robust_df["index"] == "baseline"].set_index("task").reindex(tasks)
    e = robust_df.loc[robust_df["index"] == "equal_weight"].set_index("task").reindex(tasks)
    if b.empty or e.empty:
        return None

    x = np.arange(len(tasks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.2, 6.0))

    b_y = b["delta_mean"].to_numpy(dtype=float)
    e_y = e["delta_mean"].to_numpy(dtype=float)
    b_err = np.vstack([b_y - b["ci_low"].to_numpy(dtype=float),
                       b["ci_high"].to_numpy(dtype=float) - b_y])
    e_err = np.vstack([e_y - e["ci_low"].to_numpy(dtype=float),
                       e["ci_high"].to_numpy(dtype=float) - e_y])

    ax.bar(x - width/2, b_y, width, yerr=b_err, capsize=4, label="Baseline")
    ax.bar(x + width/2, e_y, width, yerr=e_err, capsize=4, label="Equal weight")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels(tasks), rotation=25, ha="right")
    ax.set_ylabel("Δ autonomy proper vs Core-only")
    ax.set_title("Sensitivity to autonomy-index weighting")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outpath)

    out = []
    for task in tasks:
        for label, dd in [("baseline", b), ("equal_weight", e)]:
            if task in dd.index:
                r = dd.loc[task]
                out.append({
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "index": label,
                    "delta_mean": r["delta_mean"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                })
    return pd.DataFrame(out)


def plot_coefficient_perturbation(perturb_df: pd.DataFrame, tasks: List[str], outpath: Path) -> Optional[pd.DataFrame]:
    required = {"task", "probability_delta_positive_across_weights"}
    if perturb_df is None or not required.issubset(set(perturb_df.columns)):
        return None

    d = perturb_df.set_index("task").reindex(tasks).reset_index()
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.bar(x, d["probability_delta_positive_across_weights"].to_numpy(dtype=float))
    ax.axhline(0.95, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels(tasks), rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("P(Δ autonomy > 0) across coefficient samples")
    ax.set_title("Coefficient-perturbation robustness")
    fig.tight_layout()
    save_fig(fig, outpath)

    d["task_label"] = d["task"].map(lambda t: TASK_LABELS.get(t, t))
    return d


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path,
                 episode: pd.DataFrame,
                 tasks: List[str],
                 arms: List[str],
                 generated: List[str],
                 fig_tables: Dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("Publication figure generation report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Model labels used in figures")
    lines.append("----------------------------")
    for arm in arms:
        lines.append(f"{arm}: {ARM_LABELS.get(arm, arm)}")
    lines.append("")
    lines.append("Task labels used in figures")
    lines.append("---------------------------")
    for task in tasks:
        lines.append(f"{task}: {TASK_LABELS.get(task, task)}")
    lines.append("")
    lines.append("Input summary")
    lines.append("-------------")
    lines.append(f"episode_summary rows: {len(episode)}")
    lines.append(f"episodes detected per task-arm condition: see figure_values_episode_counts.csv")
    lines.append("")
    lines.append("Generated figures")
    lines.append("-----------------")
    for f in generated:
        lines.append(f)
    lines.append("")
    lines.append("Generated value tables")
    lines.append("----------------------")
    for name in fig_tables:
        lines.append(f"{name}.csv")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("The labels in all figures match the manuscript terminology: Autonomy-layer–LMM, Core-only, LMM-only, and Rule-based.")
    lines.append("Both PNG and PDF versions were saved for each figure.")
    lines.append("Figure 4 uses paired episode-level differences between Autonomy-layer–LMM and Core-only with bootstrap confidence intervals.")
    (outdir / "figure_generation_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript-consistent N20 publication figures")
    parser.add_argument("--indir", required=True, help="Input directory containing episode_summary.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for figures")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations for Figure 4")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    indir = Path(args.indir).expanduser()
    outdir = Path(args.outdir).expanduser()
    logger = Logger(outdir)

    logger.log(f"Input directory: {indir}")
    logger.log(f"Output directory: {outdir}")

    episode = read_csv(indir / "episode_summary.csv", logger, required=True)
    assert episode is not None

    require_columns(
        episode,
        [
            "task", "arm", "episode",
            "autonomy_proper_index",
            "system_sovereignty",
            "heteronomy_index",
        ],
        "episode_summary.csv",
    )

    episode = normalize_arms(episode)
    tasks = ordered_existing(episode["task"].astype(str).unique().tolist(), TASK_ORDER)
    arms = ordered_existing(episode["arm"].astype(str).unique().tolist(), ARM_ORDER)

    logger.log(f"Detected tasks: {tasks}")
    logger.log(f"Detected arms: {arms}")
    logger.log("Figure labels will use manuscript names:")
    for arm in arms:
        logger.log(f"  {arm} -> {ARM_LABELS.get(arm, arm)}")

    counts = (
        episode.groupby(["task", "arm"], as_index=False)["episode"]
        .nunique()
        .rename(columns={"episode": "n_episodes"})
    )
    counts["task_label"] = counts["task"].map(lambda t: TASK_LABELS.get(t, t))
    counts["arm_label"] = counts["arm"].map(lambda a: ARM_LABELS.get(a, a))
    counts.to_csv(outdir / "figure_values_episode_counts.csv", index=False)

    fig_tables: Dict[str, pd.DataFrame] = {}
    generated: List[str] = []

    logger.log("Creating Fig. 1: autonomy proper matrix")
    fig_tables["fig_1_values_task_agent_autonomy_proper"] = plot_heatmap(
        episode,
        "autonomy_proper_index",
        tasks,
        arms,
        "Task × agent autonomy proper",
        "Autonomy proper index",
        outdir / "fig_1_task_agent_autonomy_proper.png",
        vmin=0.0,
        vmax=1.0,
    )
    generated.append("fig_1_task_agent_autonomy_proper.png/pdf")

    logger.log("Creating Fig. 2: system sovereignty matrix")
    fig_tables["fig_2_values_task_agent_system_sovereignty"] = plot_heatmap(
        episode,
        "system_sovereignty",
        tasks,
        arms,
        "Task × agent system sovereignty",
        "System sovereignty",
        outdir / "fig_2_task_agent_system_sovereignty.png",
        vmin=0.0,
        vmax=1.0,
    )
    generated.append("fig_2_task_agent_system_sovereignty.png/pdf")

    logger.log("Creating Fig. 3: heteronomy matrix")
    fig_tables["fig_3_values_task_agent_heteronomy"] = plot_heatmap(
        episode,
        "heteronomy_index",
        tasks,
        arms,
        "Task × agent heteronomy",
        "Heteronomy index",
        outdir / "fig_3_task_agent_heteronomy.png",
        vmin=0.0,
        vmax=1.0,
    )
    generated.append("fig_3_task_agent_heteronomy.png/pdf")

    logger.log("Creating Fig. 4: Autonomy-layer–LMM advantage over Core-only")
    fig_tables["fig_4_values_autonomy_layer_lmm_advantage"] = plot_advantage(
        episode,
        tasks,
        outdir / "fig_4_autonomy_layer_lmm_advantage.png",
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    generated.append("fig_4_autonomy_layer_lmm_advantage.png/pdf")

    if {"lmm_calls", "lmm_external_decision_fraction", "darca_internal_decision_fraction"}.issubset(set(episode.columns)):
        logger.log("Creating Fig. 5: selective LMM use")
        fig_tables["fig_5_values_selective_lmm_use"] = plot_selective_lmm_use(
            episode,
            tasks,
            outdir / "fig_5_selective_lmm_use.png",
        )
        generated.append("fig_5_selective_lmm_use.png/pdf")
    else:
        logger.log("Skipping Fig. 5: LMM-use columns are missing.")

    robust_path = indir / "robustness_sensitivity" / "task_delta_robustness_summary.csv"
    perturb_path = indir / "robustness_sensitivity" / "weight_perturbation_task_summary.csv"

    robust_df = read_csv(robust_path, logger, required=False)
    if robust_df is not None:
        logger.log("Creating Fig. 6: baseline vs equal-weight sensitivity")
        t = plot_sensitivity_baseline_equal(
            robust_df,
            tasks,
            outdir / "fig_6_sensitivity_baseline_equal_weight.png",
        )
        if t is not None:
            fig_tables["fig_6_values_sensitivity_baseline_equal_weight"] = t
            generated.append("fig_6_sensitivity_baseline_equal_weight.png/pdf")

    perturb_df = read_csv(perturb_path, logger, required=False)
    if perturb_df is not None:
        logger.log("Creating Fig. 7: coefficient-perturbation robustness")
        t = plot_coefficient_perturbation(
            perturb_df,
            tasks,
            outdir / "fig_7_coefficient_perturbation_robustness.png",
        )
        if t is not None:
            fig_tables["fig_7_values_coefficient_perturbation_robustness"] = t
            generated.append("fig_7_coefficient_perturbation_robustness.png/pdf")

    logger.log("Writing figure value CSVs")
    for name, table in fig_tables.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    logger.log("Writing consolidated report")
    write_report(outdir, episode, tasks, arms, generated, fig_tables)

    logger.log("DONE")
    logger.log(f"Report: {outdir / 'figure_generation_report.txt'}")


if __name__ == "__main__":
    main()
