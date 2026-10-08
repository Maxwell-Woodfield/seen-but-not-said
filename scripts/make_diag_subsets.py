"""Create the Phase 6 subsets (effvar/diag_subsets.py). Written once, committed before any Phase 6 run.

    python scripts/make_diag_subsets.py            # write whatever is missing (never overwrites)
    python scripts/make_diag_subsets.py --check    # recompute, compare with the saved files, assert disjointness

Writes data/subsets/retinamnist_confirm.json and data/subsets/<flag>_probe.json. Every record carries the
sha1 of each image's pixels, so disjointness from tune/val/test (and confirm vs probe) can be re-verified
from the files alone. Reads image arrays only; no model, no GPU. scripts/make_subsets.py is untouched.
"""
import argparse
import json

from effvar import FLAGS
from effvar.data import SUBSET_DIR, subset_path
from effvar.diag_subsets import assert_disjoint, build_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify saved files instead of writing")
    args = ap.parse_args()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    ok = True
    for flag in FLAGS:
        records, others = build_records(flag)
        assert_disjoint(records, others)
        for name, record in records.items():
            path = subset_path(flag, name)
            if args.check:
                same = path.exists() and json.loads(path.read_text()) == record
                ok &= same
                print(f"{path.name}: {'OK' if same else 'MISMATCH'} (n={record['n']}, pool={record['pool_n']})")
                continue
            if path.exists():
                print(f"{path.name}: exists, not overwritten")
                continue
            path.write_text(json.dumps(record, indent=2))
            print(f"{path.name}: wrote n={record['n']} pool={record['pool_n']} dropped={record['pool_dropped']} "
                  f"{record['subset_class_counts']}")
    if args.check:
        print("disjointness: OK (tune, val, test, and between Phase 6 subsets)")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
