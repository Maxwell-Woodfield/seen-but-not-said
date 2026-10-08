"""Exact-duplicate images within a MedMNIST split, by sha1 of the raw pixel bytes.

    python scripts/find_duplicates.py --split test

Writes `data/subsets/<flag>_<split>_duplicates.json` per dataset: the duplicate groups, their labels,
and the index kept (the first of each group) vs dropped. The test report uses this to report
PneumoniaMNIST on both the raw n and the effective n with duplicates collapsed.

Reads image arrays only; no model, no GPU.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from effvar import FLAGS, ROOT
from effvar.data import load_split


def duplicate_groups(images) -> list[list[int]]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i in range(len(images)):
        by_hash[hashlib.sha1(np.ascontiguousarray(images[i]).tobytes()).hexdigest()].append(i)
    return sorted((v for v in by_hash.values() if len(v) > 1), key=lambda v: v[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    args = ap.parse_args()

    for flag in args.datasets:
        ds = load_split(flag, args.split)
        labels = np.asarray(ds.labels).ravel()
        groups = duplicate_groups(ds.imgs)
        drop = sorted(i for g in groups for i in g[1:])
        out = {
            "dataset": flag,
            "split": args.split,
            "n": int(len(ds.imgs)),
            "n_distinct": int(len(ds.imgs) - len(drop)),
            "groups": [{"indices": [int(i) for i in g], "labels": [int(labels[i]) for i in g],
                        "same_label": len({int(labels[i]) for i in g}) == 1} for g in groups],
            "keep_first_drop": [int(i) for i in drop],
        }
        path = ROOT / "data" / "subsets" / f"{flag}_{args.split}_duplicates.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"{flag}: n = {out['n']}, distinct = {out['n_distinct']}, "
              f"{len(groups)} duplicate group(s) -> {path.name}")


if __name__ == "__main__":
    main()
