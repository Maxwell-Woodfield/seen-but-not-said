import json

import numpy as np
import pytest

from effvar import FLAGS
from effvar.data import allocate, build_subset_record, stratified_indices, subset_path
from effvar.diag_subsets import assert_disjoint, build_records, draw, eligible_pool


@pytest.mark.parametrize("flag", FLAGS)
def test_smoke_subsets_reproduce_exactly(flag):
    saved = json.loads(subset_path(flag, "smoke").read_text())
    rebuilt = build_subset_record(flag, "val", n_total=30, min_per_class=2, seed=0)
    assert rebuilt == saved


@pytest.mark.parametrize("flag", FLAGS)
def test_tune_subsets_reproduce_exactly(flag):
    """The tune set is frozen too: hyperparameters are chosen on it, so it must never drift."""
    saved = json.loads(subset_path(flag, "tune").read_text())
    rebuilt = build_subset_record(flag, "train", n_total=300,
                                  min_per_class=5 if flag == "dermamnist" else 2, seed=0)
    assert rebuilt == saved
    assert saved["split"] == "train"  # val stays clean for H0, test for the final runs


def test_allocation_floor_and_total():
    counts = np.array([500, 3, 40, 1, 12])
    alloc = allocate(counts, n_total=30, min_per_class=5)
    assert alloc.sum() == 30
    assert all(a >= min(c, 5) for a, c in zip(alloc, counts))
    assert all(a <= c for a, c in zip(alloc, counts))


def test_eligible_pool_drops_excluded_forbidden_and_twins():
    # 0: tune; 1: pixels also in val; 2: kept; 3: twin of 2; 4: twin of 0 (tune pixels); 5: kept
    hashes = ["t", "v", "a", "a", "t", "b"]
    pool, dropped = eligible_pool(hashes, exclude_idx={0}, forbidden={"t", "v"})
    assert pool == [2, 5]
    assert dropped == {"excluded_index": 1, "pixels_in_tune_val_test": 1, "within_train_duplicate": 2}


def test_draw_stays_in_pool_and_matches_allocation():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, size=800)
    pool = sorted(rng.choice(800, size=500, replace=False).tolist())
    idx, alloc = draw(pool, labels, list(range(5)), n_total=200, min_per_class=3, seed=0)
    assert len(idx) == len(set(idx)) == 200 and set(idx) <= set(pool)
    assert (np.bincount(labels[idx], minlength=5) == alloc).all()


@pytest.mark.parametrize("flag", FLAGS)
def test_diag_subsets_reproduce_and_are_disjoint(flag):
    """Phase 6 subsets are frozen: rebuilt from the raw data they must match the committed files exactly,
    and be pixel-disjoint from tune, val, test and each other (effvar/diag_subsets.py)."""
    records, others = build_records(flag)
    assert_disjoint(records, others)
    for name, record in records.items():
        assert json.loads(subset_path(flag, name).read_text()) == record
    if flag == "retinamnist":
        assert records["confirm"]["n"] == 300 and set(records) == {"confirm", "probe"}


def test_stratified_indices_unique_in_range():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 7, size=1000)
    idx, alloc = stratified_indices(labels, list(range(7)), n_total=300, min_per_class=5, seed=0)
    assert len(idx) == len(set(idx)) == 300
    assert min(idx) >= 0 and max(idx) < 1000
    got = np.bincount(labels[idx], minlength=7)
    assert (got == alloc).all() and got.min() >= 5
