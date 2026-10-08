"""Create the fixed MedMNIST subsets. Written once, then never changed.

- `smoke` (CLAUDE.md experiment protocol): 30 images per dataset from **val**, >= 2 per class, seed 0.
- `tune` (validation_plan.md §3): 300 per dataset from **train** -- which zero-shot evaluation never
  touches otherwise -- >= 5 per class for Derma (7 classes) and >= 2 elsewhere, seed 0.
- `val` (validation_plan.md §3, the H0 / `R`-pick set): the **whole** val split, every image, no
  sampling (n = 120 / 524 / 1003). It is a subset file only so that every run goes through one code
  path; `allocate` with n_total = len(split) and min_per_class = 0 returns all indices, so the seed
  does not matter here.
- `test` (added 2026-09-21, for the final runs only): the **whole** test split, as `val` (n = 400 / 624 /
  2005). Its file is written before the test plan is launched and verified with `--check`.

All are class-stratified: a floor per class, the remainder in proportion to class frequency by largest
remainder (effvar.data.allocate). Existing files are never overwritten.

    python scripts/make_subsets.py                    # create whatever is missing
    python scripts/make_subsets.py --check            # recompute and compare with the saved files
    python scripts/make_subsets.py --subset tune      # one kind only
"""
import argparse
import json

import numpy as np

from effvar import FLAGS
from effvar.data import DATA_DIR, SUBSET_DIR, build_subset_record, load_split, subset_path

SEED = 0
SUBSETS = {
    "smoke": {"split": "val", "n": 30, "min_per_class": {}, "default_min": 2},
    "tune": {"split": "train", "n": 300, "min_per_class": {"dermamnist": 5}, "default_min": 2},
    # n = None means "the whole split": every index, no sampling.
    "val": {"split": "val", "n": None, "min_per_class": {}, "default_min": 0},
    "test": {"split": "test", "n": None, "min_per_class": {}, "default_min": 0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=sorted(SUBSETS), nargs="+", default=sorted(SUBSETS))
    parser.add_argument("--check", action="store_true", help="verify saved files instead of writing")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.subset:
        spec = SUBSETS[name]
        for flag in FLAGS:
            min_per_class = spec["min_per_class"].get(flag, spec["default_min"])
            path = subset_path(flag, name)
            n_total = spec["n"]
            if n_total is None:  # whole split
                labels = np.asarray(load_split(flag, spec["split"], download=not args.check).labels)
                n_total = int(labels.reshape(-1).shape[0])
            record = build_subset_record(flag, spec["split"], n_total, min_per_class, SEED,
                                         download=not args.check)
            assert record["n"] == n_total and len(set(record["indices"])) == n_total
            assert min(record["subset_class_counts"].values()) >= min_per_class
            if spec["n"] is None:  # every image, in order
                assert record["indices"] == list(range(n_total))
            if args.check:
                saved = json.loads(path.read_text())
                print(f"{path.name}: {'OK' if saved == record else 'MISMATCH'}")
                continue
            if path.exists():
                print(f"{path.name}: exists, not overwritten")
                continue
            path.write_text(json.dumps(record, indent=2))
            print(f"{path.name}: wrote n={record['n']} {record['subset_class_counts']}")


if __name__ == "__main__":
    main()
