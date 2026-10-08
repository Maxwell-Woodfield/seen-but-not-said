"""Phase 6 subsets: probe sets and the Retina confirmation set (notes/diagnosis_plan.md §5, B2).

Drawn from the **train** split and kept disjoint from everything else **by pixel content**, not just by
index: an image whose raw pixels (sha1) also occur in tune, val or test is never eligible. A train image
that appears more than once in train is eligible only once (its first occurrence), so no image can land
in two Phase 6 subsets through a twin.

- `retinamnist_confirm`: 300, class-stratified, seed 0, drawn FIRST. Pooled with val for Retina's
  confirmation (val alone has n = 120). Never used for exploration.
- `retinamnist_probe`: the rest of the Retina pool (<= 480) -- Retina has only 1080 train images.
- `pneumoniamnist_probe`, `dermamnist_probe`: 1000 each, class-stratified, seed 0.

Model-free: this module only reads image arrays and labels. The CLI is scripts/make_diag_subsets.py.
"""
import hashlib
import json

import medmnist
import numpy as np

from effvar.data import SIZE, load_split, stratified_indices, subset_path

SEED = 0
CONFIRM_N = 300
PROBE = {  # flag -> (n_total or None for "the rest of the pool", min_per_class)
    "retinamnist": (None, 0),
    "pneumoniamnist": (1000, 2),
    "dermamnist": (1000, 10),
}
CONFIRM_MIN_PER_CLASS = 2


def pixel_hashes(images) -> list[str]:
    """sha1 of each image's raw pixel bytes -- the same identity as scripts/find_duplicates.py."""
    return [hashlib.sha1(np.ascontiguousarray(images[i]).tobytes()).hexdigest() for i in range(len(images))]


def eligible_pool(train_hashes: list[str], exclude_idx: set[int], forbidden: set[str]) -> tuple[list[int], dict]:
    """Train indices a Phase 6 subset may use, and why the others were dropped.

    An index is dropped if it is in `exclude_idx` (tune), if its pixels occur in `forbidden` (the hashes
    of tune, val and test), or if it repeats an earlier train image's pixels.
    """
    seen: set[str] = set()
    pool, dropped = [], {"excluded_index": 0, "pixels_in_tune_val_test": 0, "within_train_duplicate": 0}
    for i, h in enumerate(train_hashes):
        if h in seen:
            dropped["within_train_duplicate"] += 1
            continue
        seen.add(h)
        if i in exclude_idx:
            dropped["excluded_index"] += 1
        elif h in forbidden:
            dropped["pixels_in_tune_val_test"] += 1
        else:
            pool.append(i)
    return pool, dropped


def draw(pool: list[int], labels: np.ndarray, classes: list[int], n_total: int, min_per_class: int,
         seed: int) -> tuple[list[int], np.ndarray]:
    """Class-stratified sample from `pool` (split indices); returns (sorted split indices, allocation)."""
    pool_arr = np.asarray(pool)
    local, alloc = stratified_indices(labels[pool_arr], classes, n_total, min_per_class, seed)
    return sorted(int(pool_arr[j]) for j in local), alloc


def _record(flag: str, name: str, indices: list[int], labels: np.ndarray, hashes: list[str], classes: list[int],
            min_per_class: int, rule: str, pool_n: int, dropped: dict) -> dict:
    info = medmnist.INFO[flag]
    disjoint = ["tune", "val", "test"] + (["confirm"] if name == "probe" and flag == "retinamnist" else [])
    return {
        "dataset": flag,
        "split": "train",
        "size": SIZE,
        "seed": SEED,
        "n": len(indices),
        "min_per_class": min_per_class,
        "medmnist_version": medmnist.__version__,
        "class_names": {str(c): info["label"][str(c)] for c in classes},
        "train_class_counts": {str(c): int((labels == c).sum()) for c in classes},
        "subset_class_counts": {str(c): int((labels[indices] == c).sum()) for c in classes},
        "purpose": name,
        "rule": rule,
        "pool_n": pool_n,
        "pool_dropped": dropped,
        "disjoint_by_pixels_from": disjoint,
        "indices": indices,
        "labels": [int(labels[i]) for i in indices],
        "pixel_sha1": [hashes[i] for i in indices],
    }


def build_records(flag: str) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """All Phase 6 subset records for one dataset, plus the hash sets they must avoid (for --check)."""
    train = load_split(flag, "train")
    labels = np.asarray(train.labels).reshape(-1).astype(int)
    classes = sorted(int(k) for k in medmnist.INFO[flag]["label"])
    train_h = pixel_hashes(train.imgs)
    others = {split: set(pixel_hashes(load_split(flag, split).imgs)) for split in ("val", "test")}
    tune = json.loads(subset_path(flag, "tune").read_text())
    others["tune"] = {train_h[i] for i in tune["indices"]}

    pool, dropped = eligible_pool(train_h, set(tune["indices"]), others["tune"] | others["val"] | others["test"])
    records = {}
    if flag == "retinamnist":
        confirm, _ = draw(pool, labels, classes, CONFIRM_N, CONFIRM_MIN_PER_CLASS, SEED)
        records["confirm"] = _record(flag, "confirm", confirm, labels, train_h, classes, CONFIRM_MIN_PER_CLASS,
                                     f"stratified {CONFIRM_N} from the pool, drawn first", len(pool), dropped)
        rest = sorted(set(pool) - set(confirm))
        records["probe"] = _record(flag, "probe", rest, labels, train_h, classes, 0,
                                   "the whole pool minus confirm", len(pool), dropped)
    else:
        n_total, min_pc = PROBE[flag]
        probe, _ = draw(pool, labels, classes, n_total, min_pc, SEED)
        records["probe"] = _record(flag, "probe", probe, labels, train_h, classes, min_pc,
                                   f"stratified {n_total} from the pool", len(pool), dropped)
    return records, others


def assert_disjoint(records: dict[str, dict], others: dict[str, set[str]]) -> None:
    """Pairwise pixel-disjointness of every Phase 6 subset with tune/val/test and with each other."""
    sets = {name: set(r["pixel_sha1"]) for name, r in records.items()}
    for name, r in records.items():
        assert len(sets[name]) == r["n"], f"{name}: repeated pixels inside the subset"
        for other, hashes in others.items():
            assert not sets[name] & hashes, f"{name} shares pixels with {other}"
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], f"{a} and {b} share pixels"
