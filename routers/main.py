"""CLI for synthetic Domain experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

EXPERIMENTS = {
    "exp01": "routers.experiments.exp01_baseline",
    "exp02": "routers.experiments.exp02_failure_synth",
    "exp03": "routers.experiments.exp03_multi_iter",
    "exp04": "routers.experiments.exp04_rare_class",
    "exp05": "routers.experiments.exp05_donor_ablation",
    "exp06": "routers.experiments.exp06_proportional",
    "exp07": "routers.experiments.exp07_pair_boundary",
    "exp08": "routers.experiments.exp08_validated",
    "exp09": "routers.experiments.exp09_baseline_500",
    "exp10": "routers.experiments.exp10_synth_ctx_in_question",
    "exp11": "routers.experiments.exp11_synth_no_context",
    "setfit01": "routers.experiments.setfit.setfit01_baseline",
    "setfit02": "routers.experiments.setfit.setfit02_uncertainty_synth",
    "setfit03": "routers.experiments.setfit.setfit03_high_conf",
    "setfit04": "routers.experiments.setfit.setfit04_low_conf",
    "setfit05": "routers.experiments.setfit.setfit05_verification_v2",
    "setfit06": "routers.experiments.setfit.setfit06_diversity_mutation",
    "setfit07": "routers.experiments.setfit.setfit07_variation_rewrite",
    "setfit08": "routers.experiments.setfit.setfit08_gemma_synth",
    "setfit09": "routers.experiments.setfit.setfit09_opentdb_low_conf",
    "setfit10": "routers.experiments.setfit.setfit10_acceptance_band",
}


def _run(module_path: str, preliminary: bool, rebuild_splits: bool) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    mod.run(save=not preliminary, rebuild_splits=rebuild_splits)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Synthetic HardNLP Domain experiments")
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()) + ["all"], help="Experiment id")
    parser.add_argument("--preliminary", action="store_true", help="Smoke run (smaller save)")
    parser.add_argument("--rebuild-splits", action="store_true", help="Regenerate split JSON files")
    args = parser.parse_args()

    if args.experiment == "all":
        for key in EXPERIMENTS:
            _run(EXPERIMENTS[key], args.preliminary, args.rebuild_splits)
    else:
        _run(EXPERIMENTS[args.experiment], args.preliminary, args.rebuild_splits)


if __name__ == "__main__":
    main()
