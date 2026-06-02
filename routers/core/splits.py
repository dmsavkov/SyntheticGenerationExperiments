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


def opentdb_setfit_splits_path() -> Path:
    return project_root() / "data" / "splits" / "opentdb_setfit_seed42.json"


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
