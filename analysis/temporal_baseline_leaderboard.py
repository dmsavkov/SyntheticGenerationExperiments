"""Build a CSV leaderboard and summary plots from results/**/metrics.json runs.

Supports both result layouts:
- Legacy: results/synthetic/<variation_slug>/<timestamp>/metrics.json (one eval per folder)
- Session: results/baseline/<experiment>/<timestamp>/metrics.json (primary eval + holdout_metrics)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
LEADERBOARD_CSV = ANALYSIS_DIR / "leaderboard.csv"
PLOTS_DIR = ANALYSIS_DIR / "plots"

# Keep the CSV focused on the fields that matter for comparing runs.
LEADERBOARD_COLS = [
    "run_timestamp",
    "results_format",
    "topic",
    "experiment",
    "phase",
    "eval_split",
    "train_split",
    "n",
    "accuracy",
    "macro_f1",
    "total_seconds",
    "n_synthetic",
    "classifier",
    "run_dir",
]

# Top-level metrics keys that are not eval scores (session envelope).
_SESSION_ENVELOPE_KEYS = frozenset(
    {
        "holdout_metrics",
        "generation_stats",
        "results_path",
        "run_timestamp",
    }
)


def _parse_run_timestamp(run_dir: Path, metrics: dict[str, Any]) -> datetime | None:
    raw = metrics.get("run_timestamp") or run_dir.name
    if not isinstance(raw, str):
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _infer_path_fields(run_dir: Path, results_root: Path) -> dict[str, str]:
    rel = run_dir.relative_to(results_root)
    parts = rel.parts
    topic = parts[0] if parts else ""
    slug = parts[1] if len(parts) >= 2 else run_dir.parent.name
    return {
        "run_dir": str(rel).replace("\\", "/"),
        "topic": topic,
        "experiment": slug,
        "variation": slug,
    }


def _is_session_format(run_dir: Path, metrics: dict[str, Any], results_root: Path) -> bool:
    if "holdout_metrics" in metrics:
        return True
    try:
        return run_dir.relative_to(results_root).parts[0] == "baseline"
    except ValueError:
        return False


def _infer_legacy_phase(variation: str, eval_split: str | None) -> str:
    if variation and "__" in variation:
        return variation.rsplit("__", 1)[-1]
    return eval_split or "unknown"


def _format_timestamp(ts: datetime | None, fallback: str) -> str:
    return ts.isoformat(sep=" ") if ts else fallback


def _session_extras(metrics: dict[str, Any]) -> dict[str, Any]:
    gen = metrics.get("generation_stats") or {}
    extras: dict[str, Any] = {
        "n_synthetic": metrics.get("n_synthetic", gen.get("n_synthetic")),
        "generation_mode": metrics.get("generation_mode"),
        "n_iterations": metrics.get("n_iterations"),
    }
    return {k: v for k, v in extras.items() if v is not None}


def _eval_row(
    eval_metrics: dict[str, Any],
    *,
    shared: dict[str, Any],
    phase: str,
    is_holdout: bool = False,
) -> dict[str, Any]:
    eval_split = eval_metrics.get("eval_split")
    if is_holdout:
        eval_split = "holdout_500"

    row = {
        **shared,
        "phase": phase,
        "eval_split": eval_split,
        "train_split": eval_metrics.get("train_split", shared.get("train_split")),
        "n": eval_metrics.get("n"),
        "accuracy": eval_metrics.get("accuracy"),
        "macro_f1": eval_metrics.get("macro_f1"),
        "total_seconds": eval_metrics.get("total_seconds"),
        "classifier": eval_metrics.get("classifier", shared.get("classifier")),
    }
    return {col: row.get(col) for col in LEADERBOARD_COLS}


def _shared_context(
    run_dir: Path,
    results_root: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    path_fields = _infer_path_fields(run_dir, results_root)
    ts = _parse_run_timestamp(run_dir, metrics)
    experiment = str(metrics.get("experiment") or path_fields["experiment"])
    return {
        "run_timestamp": _format_timestamp(ts, run_dir.name),
        "results_format": "session" if _is_session_format(run_dir, metrics, results_root) else "legacy",
        "topic": str(metrics.get("topic") or path_fields["topic"]),
        "experiment": experiment,
        "train_split": metrics.get("train_split") or config.get("train_split"),
        "classifier": metrics.get("classifier") or config.get("classifier"),
        "run_dir": path_fields["run_dir"],
        **_session_extras(metrics),
    }


def _rows_from_session_run(
    run_dir: Path,
    results_root: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    shared = _shared_context(run_dir, results_root, metrics, config)
    rows: list[dict[str, Any]] = []

    primary = {k: v for k, v in metrics.items() if k not in _SESSION_ENVELOPE_KEYS and not isinstance(v, dict)}
    rows.append(_eval_row(primary, shared=shared, phase="final"))

    holdout = metrics.get("holdout_metrics") or {}
    if isinstance(holdout, dict):
        for phase_name, phase_metrics in holdout.items():
            if isinstance(phase_metrics, dict):
                rows.append(_eval_row(phase_metrics, shared=shared, phase=str(phase_name), is_holdout=True))

    return rows


def _rows_from_legacy_run(
    run_dir: Path,
    results_root: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    shared = _shared_context(run_dir, results_root, metrics, config)
    path_fields = _infer_path_fields(run_dir, results_root)
    phase = _infer_legacy_phase(path_fields["variation"], metrics.get("eval_split"))
    return [_eval_row(metrics, shared=shared, phase=str(phase))]


def discover_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        logger.warning("Results root does not exist: {}", results_root)
        return []
    run_dirs = sorted({p.parent for p in results_root.rglob("metrics.json")})
    logger.info("Found {} run directories under {}", len(run_dirs), results_root)
    return run_dirs


def load_run_rows(run_dir: Path, results_root: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return []

    metrics = _load_json(metrics_path)
    config: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = _load_json(config_path)

    if _is_session_format(run_dir, metrics, results_root):
        return _rows_from_session_run(run_dir, results_root, metrics, config)
    return _rows_from_legacy_run(run_dir, results_root, metrics, config)


def build_leaderboard(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(results_root):
        rows.extend(load_run_rows(run_dir, results_root))

    if not rows:
        logger.warning("No runs found; writing empty leaderboard")
        return pd.DataFrame(columns=LEADERBOARD_COLS)

    df = pd.DataFrame(rows)
    present = [c for c in LEADERBOARD_COLS if c in df.columns]
    df = df[present]
    return df.sort_values(by=["macro_f1", "run_timestamp"], ascending=[False, False], na_position="last")


def _label_for_plot(row: pd.Series) -> str:
    experiment = str(row.get("experiment", ""))
    phase = str(row.get("phase", ""))
    ts = str(row.get("run_timestamp", ""))
    short_ts = ts.replace("T", " ").split(".")[0][-8:] if ts else ""
    label = f"{experiment} / {phase}" if phase else experiment
    if short_ts:
        return f"{label}\n{short_ts}"
    return label


def _plot_top10(df: pd.DataFrame, *, title: str, out_path: Path, ascending: bool) -> None:
    if df.empty:
        logger.warning("Skipping plot {} — no runs", out_path.name)
        return

    metric_col = "macro_f1"
    if metric_col not in df.columns:
        logger.warning("Skipping plot {} — missing {}", out_path.name, metric_col)
        return

    plot_df = df.sort_values(by=metric_col, ascending=ascending, na_position="last").head(10)
    if plot_df.empty:
        return

    labels = [_label_for_plot(row) for _, row in plot_df.iterrows()]
    values = plot_df[metric_col].astype(float).tolist()

    fig_h = max(4.0, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    color = "#2ca02c" if not ascending else "#d62728"
    ax.barh(range(len(plot_df)), values, color=color, alpha=0.85)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("macro_f1")
    ax.set_title(title)
    ax.set_xlim(0, min(1.0, max(values) + 0.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote {}", out_path)


def write_plots(df: pd.DataFrame, plots_dir: Path = PLOTS_DIR) -> None:
    if df.empty:
        logger.warning("Skipping plots — leaderboard is empty")
        return

    # Compare final eval runs only so holdout checkpoints do not dominate rankings.
    final_df = df[df["phase"] == "final"] if "phase" in df.columns else df
    if final_df.empty:
        final_df = df

    recent_df = final_df.sort_values(by="run_timestamp", ascending=False, na_position="last")
    _plot_top10(final_df, title="Top 10 best final evals (macro_f1)", out_path=plots_dir / "top10_best.png", ascending=False)
    _plot_top10(final_df, title="Top 10 worst final evals (macro_f1)", out_path=plots_dir / "top10_worst.png", ascending=True)
    _plot_top10(
        recent_df,
        title="10 most recent final evals (macro_f1)",
        out_path=plots_dir / "top10_recent.png",
        ascending=False,
    )


def update_leaderboard(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    csv_path: Path = LEADERBOARD_CSV,
    plots_dir: Path = PLOTS_DIR,
    write_csv: bool = True,
    write_plot_files: bool = True,
) -> pd.DataFrame:
    df = build_leaderboard(results_root)
    if write_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Updated {} ({} rows from {} run dirs)", csv_path, len(df), df["run_dir"].nunique() if not df.empty else 0)
    if write_plot_files:
        write_plots(df, plots_dir)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the experiment results leaderboard.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory to scan (default: project results/)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=LEADERBOARD_CSV,
        help="Output CSV path (overwritten each run)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for summary PNG plots",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    df = update_leaderboard(
        results_root=args.results_root,
        csv_path=args.csv,
        plots_dir=args.plots_dir,
        write_plot_files=not args.no_plots,
    )
    if not df.empty and "macro_f1" in df.columns:
        final_df = df[df["phase"] == "final"] if "phase" in df.columns else df
        best = final_df.sort_values(by="macro_f1", ascending=False).iloc[0]
        logger.info(
            "Best final eval: {} / {} macro_f1={} ({})",
            best.get("experiment"),
            best.get("phase"),
            best.get("macro_f1"),
            best.get("run_dir"),
        )


if __name__ == "__main__":
    main()
