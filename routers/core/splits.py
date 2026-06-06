"""Frozen train/eval ID management + domain synthesis splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from routers.core.data import SHUFFLE_SEED, load_arena
from routers.core.constants import (
    COMBINATION_BINARY_HOLDOUT_N,
    COMBINATION_BINARY_LABELS,
    COMBINATION_BINARY_TRAIN_PER_CLASS,
)

SEED = SHUFFLE_SEED


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def splits_path() -> Path:
    return project_root() / "data" / "splits" / "seed42.json"


def domain_synth_splits_path() -> Path:
    return project_root() / "data" / "splits" / "domain_synth_seed42.json"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class SplitBundle:
    eval_50_ids: list[Any]
    eval_1000_ids: list[Any]
    train_pool_ids: list[Any]
    train_20_ids: list[Any]
    train_500_ids: list[Any]
    train_remaining_ids: list[Any]
    id_to_idx: dict[Any, int]

    def eval_ids(self, name: str, preliminary: bool = False) -> list[Any]:
        if name == "eval_50":
            base = self.eval_50_ids
        elif name == "eval_1000":
            base = self.eval_1000_ids
        else:
            raise ValueError(name)
        if not preliminary:
            return list(base)
        return _mini_stratified(base, n=min(5, len(base)), seed=SEED + 99)

    def train_ids(self, name: str, preliminary: bool = False) -> list[Any]:
        mapping = {
            "train_20": self.train_20_ids,
            "train_500": self.train_500_ids,
            "train_remaining": self.train_remaining_ids,
        }
        if name not in mapping:
            raise ValueError(name)
        base = mapping[name]
        if not preliminary:
            return list(base)
        return list(base[: min(10, len(base))])


@dataclass
class DomainSynthSplits:
    train_2k_ids: list[Any]
    holdout_500_ids: list[Any]
    train_500_ids: list[Any]
    train_100_ids: list[Any]
    donor_pool_ids: list[Any]


@dataclass
class OpenTdbSetfitSplits:
    """OpenTDB-filtered SetFit splits (train_100, holdout_500, eval)."""

    train_100_ids: list[Any]
    holdout_500_ids: list[Any]
    eval_ids: list[Any]
    donor_pool_ids: list[Any]


@dataclass
class OpenTdbEnsembleSplits:
    """OpenTDB ensemble/combination: train_100, holdout_100, test = remainder."""

    train_100_ids: list[Any]
    holdout_100_ids: list[Any]
    test_ids: list[Any]
    label_universe: list[str]


@dataclass
class OpenTdbCombinationBinarySplits:
    """OpenTDB CS vs Technology: 20 train (10/class), 80 holdout, rest test."""

    train_20_ids: list[Any]
    holdout_80_ids: list[Any]
    test_ids: list[Any]
    label_universe: list[str]


def opentdb_setfit_splits_path() -> Path:
    return project_root() / "data" / "splits" / "opentdb_setfit_seed42.json"


def opentdb_ensemble_splits_path() -> Path:
    return project_root() / "data" / "splits" / "opentdb_ensemble_seed42.json"


def opentdb_combination_binary_splits_path() -> Path:
    return project_root() / "data" / "splits" / "opentdb_combination_binary_seed42.json"


def _is_opentdb_id(rid: Any) -> bool:
    return str(rid).startswith("OpenTDB")


def _stratified_ids(
    df: pd.DataFrame,
    pool_ids: list[Any],
    n: int,
    stratify_col: str,
    seed: int,
    id_to_idx: dict[Any, int],
) -> list[Any]:
    if n <= 0:
        return []
    by_label: dict[str, list[Any]] = {}
    for rid in pool_ids:
        row = df.iloc[id_to_idx[rid]]
        label = str(row[stratify_col])
        by_label.setdefault(label, []).append(rid)

    n_classes = len(by_label)
    per_class = max(1, n // n_classes)
    chosen: list[Any] = []
    rng = random.Random(seed)
    for _label, ids in sorted(by_label.items()):
        take = min(per_class, len(ids))
        chosen.extend(rng.sample(ids, take))

    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        rest = [i for i in pool_ids if i not in set(chosen)]
        need = n - len(chosen)
        if rest:
            chosen.extend(rng.sample(rest, min(need, len(rest))))

    return chosen[:n]


def _mini_stratified(ids: list[Any], n: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    if len(ids) <= n:
        return list(ids)
    return rng.sample(ids, n)


def _build_split_file(df: pd.DataFrame) -> dict[str, Any]:
    id_col = "Global Index"
    if id_col not in df.columns:
        all_ids = list(range(len(df)))
    else:
        all_ids = df[id_col].tolist()

    id_to_idx = {all_ids[i]: i for i in range(len(all_ids))}

    eval_50 = _stratified_ids(df, all_ids, 50, "Domain", SEED, id_to_idx)
    used = set(eval_50)
    pool_after_50 = [i for i in all_ids if i not in used]

    eval_1000 = _stratified_ids(df, pool_after_50, 1000, "Domain", SEED + 1, id_to_idx)
    used |= set(eval_1000)
    train_pool = [i for i in all_ids if i not in used]

    train_20 = _stratified_ids(df, train_pool, 20, "Domain", SEED + 2, id_to_idx)
    train_500 = _stratified_ids(df, train_pool, 500, "Domain", SEED + 3, id_to_idx)

    return {
        "seed": SEED,
        "eval_50_ids": eval_50,
        "eval_1000_ids": eval_1000,
        "train_pool_ids": train_pool,
        "train_20_ids": train_20,
        "train_500_ids": train_500,
        "train_remaining_ids": list(train_pool),
    }


def build_domain_synth_splits(df: pd.DataFrame, train_pool_ids: list[Any], id_to_idx: dict[Any, int]) -> dict[str, Any]:
    pool = list(train_pool_ids)
    train_2k = _stratified_ids(df, pool, 2000, "Domain", SEED + 46, id_to_idx)
    used = set(train_2k)
    pool2 = [i for i in pool if i not in used]
    holdout_500 = _stratified_ids(df, pool2, 500, "Domain", SEED + 47, id_to_idx)
    used |= set(holdout_500)
    pool3 = [i for i in pool if i not in used]
    train_500 = _stratified_ids(df, pool3, 500, "Domain", SEED + 48, id_to_idx)
    used |= set(train_500)
    pool4 = [i for i in pool if i not in used]
    train_100 = _stratified_ids(df, pool4, 100, "Domain", SEED + 49, id_to_idx)
    used |= set(train_100)
    donor_pool = [i for i in pool if i not in used]
    return {
        "seed": SEED,
        "train_2k_ids": train_2k,
        "holdout_500_ids": holdout_500,
        "train_500_ids": train_500,
        "train_100_ids": train_100,
        "donor_pool_ids": donor_pool,
    }


def get_splits(df: pd.DataFrame | None = None, *, rebuild: bool = False) -> SplitBundle:
    path = splits_path()
    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        if df is None:
            df = load_arena()
        payload = _build_split_file(df)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote frozen splits to {}", path)

    if df is None:
        df = load_arena()
    id_col = "Global Index"
    if id_col in df.columns:
        id_to_idx = {df.iloc[i][id_col]: i for i in range(len(df))}
    else:
        id_to_idx = {i: i for i in range(len(df))}

    return SplitBundle(
        eval_50_ids=payload["eval_50_ids"],
        eval_1000_ids=payload["eval_1000_ids"],
        train_pool_ids=payload["train_pool_ids"],
        train_20_ids=payload["train_20_ids"],
        train_500_ids=payload["train_500_ids"],
        train_remaining_ids=payload["train_remaining_ids"],
        id_to_idx=id_to_idx,
    )


def get_domain_synth_splits(
    df: pd.DataFrame | None = None,
    bundle: SplitBundle | None = None,
    *,
    rebuild: bool = False,
) -> DomainSynthSplits:
    path = domain_synth_splits_path()
    if bundle is None:
        bundle = get_splits(df, rebuild=rebuild)
    if df is None:
        df = load_arena()

    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = build_domain_synth_splits(df, bundle.train_pool_ids, bundle.id_to_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote domain synth splits to {}", path)

    return DomainSynthSplits(
        train_2k_ids=payload["train_2k_ids"],
        holdout_500_ids=payload["holdout_500_ids"],
        train_500_ids=payload["train_500_ids"],
        train_100_ids=payload.get("train_100_ids", []),
        donor_pool_ids=payload["donor_pool_ids"],
    )


def build_opentdb_setfit_splits(
    df: pd.DataFrame,
    bundle: SplitBundle,
    *,
    eval_target: int = 1000,
) -> dict[str, Any]:
    """Build SetFit splits from OpenTDB rows only."""
    id_to_idx = bundle.id_to_idx
    otd_train_pool = [i for i in bundle.train_pool_ids if _is_opentdb_id(i)]
    otd_eval_pool = [i for i in bundle.eval_1000_ids if _is_opentdb_id(i)]

    train_100 = _stratified_ids(df, otd_train_pool, 100, "Domain", SEED + 60, id_to_idx)
    used = set(train_100)
    pool2 = [i for i in otd_train_pool if i not in used]
    holdout_500 = _stratified_ids(df, pool2, 500, "Domain", SEED + 61, id_to_idx)
    used |= set(holdout_500)
    donor_pool = [i for i in otd_train_pool if i not in used]

    eval_n = min(eval_target, len(otd_eval_pool) + len(donor_pool))
    eval_ids = _stratified_ids(df, otd_eval_pool, eval_n, "Domain", SEED + 62, id_to_idx)
    if len(eval_ids) < eval_n:
        rest = [i for i in donor_pool if i not in set(eval_ids)]
        need = eval_n - len(eval_ids)
        if rest:
            extra = _stratified_ids(df, rest, need, "Domain", SEED + 63, id_to_idx)
            eval_ids = list(dict.fromkeys(eval_ids + extra))[:eval_n]

    logger.info(
        "OpenTDB splits: train_100={}, holdout_500={}, eval={} (pool train={}, eval={})",
        len(train_100),
        len(holdout_500),
        len(eval_ids),
        len(otd_train_pool),
        len(otd_eval_pool),
    )
    return {
        "seed": SEED,
        "train_100_ids": train_100,
        "holdout_500_ids": holdout_500,
        "eval_ids": eval_ids,
        "donor_pool_ids": donor_pool,
    }


def get_opentdb_setfit_splits(
    df: pd.DataFrame | None = None,
    bundle: SplitBundle | None = None,
    *,
    rebuild: bool = False,
) -> OpenTdbSetfitSplits:
    path = opentdb_setfit_splits_path()
    if bundle is None:
        bundle = get_splits(df, rebuild=rebuild)
    if df is None:
        df = load_arena()

    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = build_opentdb_setfit_splits(df, bundle)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote OpenTDB SetFit splits to {}", path)

    return OpenTdbSetfitSplits(
        train_100_ids=payload["train_100_ids"],
        holdout_500_ids=payload["holdout_500_ids"],
        eval_ids=payload["eval_ids"],
        donor_pool_ids=payload.get("donor_pool_ids", []),
    )


def _labels_for_ids(
    df: pd.DataFrame,
    ids: list[Any],
    id_to_idx: dict[Any, int],
    col: str = "Domain",
) -> list[str]:
    labels: set[str] = set()
    for rid in ids:
        labels.add(str(df.iloc[id_to_idx[rid]][col]))
    return sorted(labels)


def build_opentdb_ensemble_splits(
    df: pd.DataFrame,
    bundle: SplitBundle,
) -> dict[str, Any]:
    """100 train + 100 holdout from OTD pool; test = all other OTD ids."""
    id_to_idx = bundle.id_to_idx
    otd_train_pool = [i for i in bundle.train_pool_ids if _is_opentdb_id(i)]
    otd_eval_pool = [i for i in bundle.eval_1000_ids if _is_opentdb_id(i)]
    all_otd = list(dict.fromkeys(otd_train_pool + otd_eval_pool))

    train_100 = _stratified_ids(df, otd_train_pool, 100, "Domain", SEED + 70, id_to_idx)
    used = set(train_100)
    pool2 = [i for i in otd_train_pool if i not in used]
    holdout_100 = _stratified_ids(df, pool2, 100, "Domain", SEED + 71, id_to_idx)
    used |= set(holdout_100)
    test_ids = [i for i in all_otd if i not in used]

    label_universe = _labels_for_ids(
        df, train_100 + holdout_100 + test_ids, id_to_idx
    )
    logger.info(
        "OpenTDB ensemble splits: train={}, holdout={}, test={}, labels={}",
        len(train_100),
        len(holdout_100),
        len(test_ids),
        len(label_universe),
    )
    return {
        "seed": SEED,
        "train_100_ids": train_100,
        "holdout_100_ids": holdout_100,
        "test_ids": test_ids,
        "label_universe": label_universe,
    }


def get_opentdb_ensemble_splits(
    df: pd.DataFrame | None = None,
    bundle: SplitBundle | None = None,
    *,
    rebuild: bool = False,
) -> OpenTdbEnsembleSplits:
    path = opentdb_ensemble_splits_path()
    if bundle is None:
        bundle = get_splits(df, rebuild=rebuild)
    if df is None:
        df = load_arena()

    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = build_opentdb_ensemble_splits(df, bundle)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote OpenTDB ensemble splits to {}", path)

    return OpenTdbEnsembleSplits(
        train_100_ids=payload["train_100_ids"],
        holdout_100_ids=payload["holdout_100_ids"],
        test_ids=payload["test_ids"],
        label_universe=payload.get("label_universe", []),
    )


def _opentdb_ids_for_labels(
    df: pd.DataFrame,
    bundle: SplitBundle,
    labels: tuple[str, ...] | list[str],
) -> list[Any]:
    """OpenTDB rows whose Domain is in labels."""
    id_to_idx = bundle.id_to_idx
    label_set = set(labels)
    pool = [i for i in bundle.train_pool_ids + bundle.eval_1000_ids if _is_opentdb_id(i)]
    return [
        rid
        for rid in pool
        if str(df.iloc[id_to_idx[rid]]["Domain"]) in label_set
    ]


def build_opentdb_combination_binary_splits(
    df: pd.DataFrame,
    bundle: SplitBundle,
) -> dict[str, Any]:
    """20 train (10/class) + 80 holdout from CS/Technology OTD pool; test = remainder."""
    labels = tuple(COMBINATION_BINARY_LABELS)
    id_to_idx = bundle.id_to_idx
    otd_pool = _opentdb_ids_for_labels(df, bundle, labels)

    by_label: dict[str, list[Any]] = {lab: [] for lab in labels}
    for rid in otd_pool:
        lab = str(df.iloc[id_to_idx[rid]]["Domain"])
        by_label[lab].append(rid)

    rng = random.Random(SEED + 80)
    train_20: list[Any] = []
    for lab in labels:
        ids = by_label.get(lab, [])
        take = min(COMBINATION_BINARY_TRAIN_PER_CLASS, len(ids))
        if take:
            train_20.extend(rng.sample(ids, take))

    used = set(train_20)
    pool2 = [i for i in otd_pool if i not in used]
    holdout_80 = _stratified_ids(
        df, pool2, COMBINATION_BINARY_HOLDOUT_N, "Domain", SEED + 81, id_to_idx
    )
    used |= set(holdout_80)
    test_ids = [i for i in otd_pool if i not in used]

    logger.info(
        "OpenTDB binary combination splits: train={}, holdout={}, test={}, labels={}",
        len(train_20),
        len(holdout_80),
        len(test_ids),
        len(labels),
    )
    return {
        "seed": SEED,
        "train_20_ids": train_20,
        "holdout_80_ids": holdout_80,
        "test_ids": test_ids,
        "label_universe": list(labels),
    }


def get_opentdb_combination_binary_splits(
    df: pd.DataFrame | None = None,
    bundle: SplitBundle | None = None,
    *,
    rebuild: bool = False,
) -> OpenTdbCombinationBinarySplits:
    path = opentdb_combination_binary_splits_path()
    if bundle is None:
        bundle = get_splits(df, rebuild=rebuild)
    if df is None:
        df = load_arena()

    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = build_opentdb_combination_binary_splits(df, bundle)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote OpenTDB binary combination splits to {}", path)

    return OpenTdbCombinationBinarySplits(
        train_20_ids=payload["train_20_ids"],
        holdout_80_ids=payload["holdout_80_ids"],
        test_ids=payload["test_ids"],
        label_universe=payload.get("label_universe", list(COMBINATION_BINARY_LABELS)),
    )
