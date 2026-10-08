"""MedMNIST subsets: stratified index selection, subset files, and image loading."""
import json
from dataclasses import dataclass
from pathlib import Path

import medmnist
import numpy as np

from effvar import ROOT

DATA_DIR = ROOT / "data" / "medmnist"
SUBSET_DIR = ROOT / "data" / "subsets"
SIZE = 224


def allocate(counts: np.ndarray, n_total: int, min_per_class: int) -> np.ndarray:
    """Per-class sample counts: a floor of `min_per_class`, the rest proportional (largest remainder)."""
    alloc = np.minimum(counts, min_per_class)
    capacity = counts - alloc
    remaining = min(n_total - alloc.sum(), capacity.sum())
    quota = remaining * capacity / capacity.sum()  # quota_i <= capacity_i, so no class overflows
    extra = np.floor(quota).astype(int)
    leftover = remaining - extra.sum()
    extra[np.argsort(-(quota - extra), kind="stable")[:leftover]] += 1
    return alloc + extra


def stratified_indices(labels: np.ndarray, classes: list[int], n_total: int, min_per_class: int,
                       seed: int) -> tuple[list[int], np.ndarray]:
    """Class-stratified sample without replacement; returns (sorted indices, per-class allocation)."""
    labels = np.asarray(labels).reshape(-1).astype(int)
    counts = np.array([(labels == c).sum() for c in classes])
    alloc = allocate(counts, n_total, min_per_class)
    rng = np.random.default_rng(seed)
    picked = []
    for c, k in zip(classes, alloc):
        picked.extend(rng.choice(np.flatnonzero(labels == c), size=int(k), replace=False).tolist())
    return sorted(int(i) for i in picked), alloc


def load_split(flag: str, split: str, download: bool = False):
    """The medmnist dataset object (224 px, RGB); memory-mapped so large splits are not loaded eagerly."""
    info = medmnist.INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    return cls(split=split, size=SIZE, as_rgb=True, download=download, root=str(DATA_DIR), mmap_mode="r")


def build_subset_record(flag: str, split: str, n_total: int, min_per_class: int, seed: int,
                        download: bool = False) -> dict:
    """The JSON record for a stratified subset of one split (the format of data/subsets/*.json)."""
    info = medmnist.INFO[flag]
    ds = load_split(flag, split, download=download)
    labels = np.asarray(ds.labels).reshape(-1).astype(int)
    classes = sorted(int(k) for k in info["label"])
    indices, alloc = stratified_indices(labels, classes, n_total, min_per_class, seed)
    return {
        "dataset": flag,
        "split": split,
        "size": SIZE,
        "seed": seed,
        "n": len(indices),
        "min_per_class": min_per_class,
        "medmnist_version": medmnist.__version__,
        "class_names": {str(c): info["label"][str(c)] for c in classes},
        f"{split}_class_counts": {str(c): int((labels == c).sum()) for c in classes},
        "subset_class_counts": {str(c): int(k) for c, k in zip(classes, alloc)},
        "indices": indices,
        "labels": [int(labels[i]) for i in indices],
    }


@dataclass(frozen=True)
class Subset:
    flag: str
    name: str
    split: str
    indices: list[int]
    labels: list[int]
    class_names: dict[str, str]


def subset_path(flag: str, name: str) -> Path:
    return SUBSET_DIR / f"{flag}_{name}.json"


def load_subset(flag: str, name: str) -> Subset:
    rec = json.loads(subset_path(flag, name).read_text())
    return Subset(flag=flag, name=name, split=rec["split"], indices=rec["indices"], labels=rec["labels"],
                  class_names=rec["class_names"])


def iter_samples(subset: Subset, limit: int | None = None):
    """Yield (index, label, PIL image) for a subset, in the saved index order."""
    ds = load_split(subset.flag, subset.split)
    pairs = list(zip(subset.indices, subset.labels))[:limit]
    for idx, label in pairs:
        img, target = ds[idx]
        assert int(np.asarray(target).reshape(-1)[0]) == label, (subset.flag, idx)
        yield idx, label, img
