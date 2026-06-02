"""Build playground_arc_med_mmlu_seed42.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from routers.core.data import load_arena
from routers.core.splits import get_splits
from routers.playground.splits import get_playground_splits


def main() -> None:
    df = load_arena()
    bundle = get_splits(df)
    get_playground_splits(df, bundle, preset="arc_med_mmlu", rebuild=True)
    print("Playground splits written under data/splits/")


if __name__ == "__main__":
    main()
