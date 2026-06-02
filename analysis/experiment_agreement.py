"""Agreement + per-class PR comparisons across experiments.

Reads saved run artifacts under `results/` and writes plots/summaries under `analysis/plots/agreement/`.

Outputs:
- Cohen's Kappa heatmap across runs (agreement between model predictions)
- Fleiss' Kappa single score across a group of runs (global agreement)
- McNemar exact test for selected model pairs (correctness disagreement vs gold)
- Per-class precision/recall distribution across runs

Terminology:
- Use **Cohen's Kappa agreement** (pairwise, discrete labels), not correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = ANALYSIS_DIR / "plots" / "agreement"


@dataclass(frozen=True)
class RunRef:
    run_dir: Path  # absolute path to run folder containing metrics.json
    topic: str
    name: str  # display name (experiment/variation)
    run_timestamp: str


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _parse_timestamp(metrics: dict[str, Any], run_dir: Path) -> str:
    raw = metrics.get("run_timestamp") or run_dir.name
    if not isinstance(raw, str):
        return run_dir.name
    try:
        dt = datetime.strptime(raw, "%Y%m%d_%H%M%S")
        return dt.isoformat(sep=" ")
    except ValueError:
        return raw


def _run_name(metrics: dict[str, Any], run_dir: Path) -> str:
    exp = metrics.get("experiment") or metrics.get("variation")
    if isinstance(exp, str) and exp:
        return exp
    return run_dir.parent.name


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:120]


def discover_runs(results_root: Path, *, topic: str | None, max_runs: int | None) -> list[RunRef]:
    run_dirs = sorted({p.parent for p in results_root.rglob("metrics.json")})
    runs: list[RunRef] = []
    for rd in run_dirs:
        metrics = _load_json(rd / "metrics.json")
        t = metrics.get("topic")
        if topic is not None and t != topic:
            continue
        runs.append(
            RunRef(
                run_dir=rd,
                topic=str(t or ""),
                name=_run_name(metrics, rd),
                run_timestamp=_parse_timestamp(metrics, rd),
            )
        )

    runs.sort(key=lambda r: r.run_timestamp, reverse=True)
    if max_runs is not None:
        runs = runs[:max_runs]
    logger.info("Discovered {} runs (topic={})", len(runs), topic or "*")
    return runs


def _load_predictions(run_dir: Path) -> pd.DataFrame | None:
    correct_p = run_dir / "predictions_correct.json"
    incorrect_p = run_dir / "predictions_incorrect.json"
    if not correct_p.exists() or not incorrect_p.exists():
        return None

    def load_list(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        return [x for x in payload if isinstance(x, dict)]

    rows = load_list(correct_p) + load_list(incorrect_p)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if not {"id", "gold", "pred"}.issubset(set(df.columns)):
        return None
    df = df[["id", "gold", "pred"]].copy()
    df["id"] = df["id"].astype(str)
    df["gold"] = df["gold"].astype(str)
    df["pred"] = df["pred"].astype(str)
    return df


def load_prediction_table(
    runs: list[RunRef],
    *,
    gold_filter: str | None,
) -> tuple[pd.DataFrame, pd.Series, list[RunRef]]:
    """Return:
    - preds_wide: index=id, columns=run.name, values=pred label
    - gold: index=id, values=gold label (from first run)
    - kept_runs: runs with predictions
    """
    series_by_run: dict[str, pd.Series] = {}
    gold_by_id: pd.Series | None = None
    kept: list[RunRef] = []

    for run in runs:
        pred_df = _load_predictions(run.run_dir)
        if pred_df is None:
            logger.warning("No predictions found for {} ({})", run.name, run.run_dir)
            continue
        if gold_by_id is None:
            gold_by_id = pred_df.set_index("id")["gold"]
        kept.append(run)
        series_by_run[run.name] = pred_df.set_index("id")["pred"]

    if not series_by_run or gold_by_id is None:
        return pd.DataFrame(), pd.Series(dtype=str), []

    preds_wide = pd.DataFrame(series_by_run).dropna(axis=0, how="any")
    gold = gold_by_id.loc[preds_wide.index].astype(str)

    if gold_filter:
        rx = re.compile(gold_filter)
        mask = gold.apply(lambda s: bool(rx.search(s)))
        preds_wide = preds_wide.loc[mask]
        gold = gold.loc[preds_wide.index]

    logger.info("Prediction intersection: {} items x {} runs", preds_wide.shape[0], preds_wide.shape[1])
    return preds_wide, gold, kept


def cohen_kappa_matrix(preds_wide: pd.DataFrame) -> pd.DataFrame:
    names = list(preds_wide.columns)
    n = len(names)
    mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                a = preds_wide.iloc[:, i].tolist()
                b = preds_wide.iloc[:, j].tolist()
                mat[i, j] = float(cohen_kappa_score(a, b))
    return pd.DataFrame(mat, index=names, columns=names)


def fleiss_kappa(preds_wide: pd.DataFrame) -> float:
    """Fleiss' Kappa for nominal multiclass ratings."""
    if preds_wide.empty:
        return float("nan")
    labels = sorted({str(x) for x in preds_wide.to_numpy().ravel().tolist()})
    if not labels:
        return float("nan")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_items = preds_wide.shape[0]
    n_raters = preds_wide.shape[1]
    n_cats = len(labels)
    if n_raters < 2:
        return float("nan")

    counts = np.zeros((n_items, n_cats), dtype=float)
    for row_i, row in enumerate(preds_wide.itertuples(index=False, name=None)):
        for lab in row:
            counts[row_i, label_to_idx[str(lab)]] += 1.0

    p_i = (np.sum(counts * (counts - 1.0), axis=1)) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_i))
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j**2))
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def _heatmap(
    mat: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    vmin: float = -0.2,
    vmax: float = 1.0,
) -> None:
    if mat.empty:
        return
    n = mat.shape[0]
    fig_w = max(10.0, 0.55 * n + 4.0)
    fig_h = max(8.0, 0.55 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's Kappa", rotation=90)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title(title)

    data = mat.to_numpy()
    mid = (vmin + vmax) / 2.0
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.2f}"
            color = "white" if (not np.isnan(v) and v < mid) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote {}", out_path)


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * prob))


def mcnemar_table(gold: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> tuple[dict[str, int], float]:
    a_ok = pred_a.astype(str) == gold.astype(str)
    b_ok = pred_b.astype(str) == gold.astype(str)
    both_ok = int((a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    both_wrong = int((~a_ok & ~b_ok).sum())
    p = _mcnemar_exact_p(a_only, b_only)
    return {"both_correct": both_ok, "a_only": a_only, "b_only": b_only, "both_wrong": both_wrong}, p


def per_class_pr_table(runs: list[RunRef]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics_path = run.run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        prec = metrics.get("per_class_precision") or {}
        rec = metrics.get("per_class_recall") or {}
        if not isinstance(prec, dict) or not isinstance(rec, dict):
            continue
        for label, p in prec.items():
            rows.append({"run": run.name, "label": str(label), "metric": "precision", "value": float(p)})
        for label, r in rec.items():
            rows.append({"run": run.name, "label": str(label), "metric": "recall", "value": float(r)})
    return pd.DataFrame(rows)


def _boxplot_per_class(pr_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if pr_df.empty:
        return
    labels = sorted(pr_df["label"].unique().tolist())
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(12, 0.8 * len(labels) + 6), 10),
        sharex=True,
    )
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ["precision", "recall"], strict=True):
        sub = pr_df[pr_df["metric"] == metric]
        data: list[list[float]] = []
        for lab in labels:
            vals = sub[sub["label"] == lab]["value"].astype(float).tolist()
            data.append(vals)
        ax.boxplot(data, showfliers=False)
        for i, vals in enumerate(data, start=1):
            if not vals:
                continue
            xs = rng.normal(loc=i, scale=0.06, size=len(vals))
            ax.scatter(xs, vals, s=14, alpha=0.55, color="#1f77b4")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{metric} by class (across runs)")

    axes[-1].set_xticks(range(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote {}", out_path)


def write_agreement_bundle(
    *,
    results_root: Path,
    out_dir: Path,
    topic: str,
    max_runs: int,
    gold_filter: str | None,
    mcnemar_top_k: int,
) -> None:
    runs = discover_runs(results_root, topic=topic, max_runs=max_runs)
    preds_wide, gold, kept_runs = load_prediction_table(runs, gold_filter=gold_filter)
    if preds_wide.empty or len(kept_runs) < 2:
        logger.warning("Not enough runs with predictions to compute agreement (topic={}).", topic)
        return

    suffix = _safe_filename(topic + (f"__gold={gold_filter}" if gold_filter else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cohen's Kappa heatmap
    kappa_mat = cohen_kappa_matrix(preds_wide)
    _heatmap(
        kappa_mat,
        out_path=out_dir / f"cohen_kappa_heatmap__{suffix}.png",
        title=f"Cohen's Kappa agreement heatmap (topic={topic})"
        + (f" | gold~/{gold_filter}/" if gold_filter else ""),
    )

    # Fleiss Kappa (global)
    fk = fleiss_kappa(preds_wide)

    # Per-class PR distributions
    pr_df = per_class_pr_table(kept_runs)
    _boxplot_per_class(
        pr_df,
        out_path=out_dir / f"per_class_pr__{suffix}.png",
        title=f"Per-class precision/recall across runs (topic={topic})",
    )

    # McNemar (exact) on correctness-vs-gold for model pairs.
    names = list(preds_wide.columns)
    pair_rows: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = preds_wide[names[i]]
            b = preds_wide[names[j]]
            tbl, p = mcnemar_table(gold, a, b)
            pair_rows.append(
                {
                    "model_a": names[i],
                    "model_b": names[j],
                    "a_only": tbl["a_only"],
                    "b_only": tbl["b_only"],
                    "both_correct": tbl["both_correct"],
                    "both_wrong": tbl["both_wrong"],
                    "disagreements": tbl["a_only"] + tbl["b_only"],
                    "p_value_exact": p,
                }
            )
    pairs_df = pd.DataFrame(pair_rows).sort_values(by="disagreements", ascending=False)

    summary_path = out_dir / f"summary__{suffix}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"topic={topic}\n")
        f.write(f"runs_with_predictions={len(kept_runs)}\n")
        f.write(f"items_intersection={preds_wide.shape[0]}\n")
        if gold_filter:
            f.write(f"gold_filter=/{gold_filter}/\n")
        f.write(f"fleiss_kappa={fk:.6f}\n\n")
        f.write("Top McNemar pairs by disagreements (exact two-sided p)\n")
        f.write(pairs_df.head(mcnemar_top_k).to_string(index=False))
        f.write("\n")
    logger.info("Wrote {}", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement analysis across experiment runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--topic", type=str, required=True, help="Group by results/<topic>/...")
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument(
        "--gold-filter",
        type=str,
        default=None,
        help="Regex applied to gold labels (e.g. 'Finance' or '^Finance$')",
    )
    parser.add_argument("--mcnemar-top-k", type=int, default=10)
    args = parser.parse_args()

    write_agreement_bundle(
        results_root=args.results_root,
        out_dir=args.out_dir / _safe_filename(args.topic),
        topic=args.topic,
        max_runs=args.max_runs,
        gold_filter=args.gold_filter,
        mcnemar_top_k=args.mcnemar_top_k,
    )


if __name__ == "__main__":
    main()

