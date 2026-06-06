"""Build seed42.json and domain_synth_seed42.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from routers.core.data import load_arena
from routers.core.splits import (
    get_domain_synth_splits,
    get_opentdb_combination_binary_splits,
    get_opentdb_ensemble_splits,
    get_opentdb_setfit_splits,
    get_splits,
)


def main() -> None:
    df = load_arena()
    bundle = get_splits(df, rebuild=True)
    get_domain_synth_splits(df, bundle, rebuild=True)
    get_opentdb_setfit_splits(df, bundle, rebuild=True)
    get_opentdb_ensemble_splits(df, bundle, rebuild=True)
    get_opentdb_combination_binary_splits(df, bundle, rebuild=True)
    print("Splits written under data/splits/")


if __name__ == "__main__":
    main()
