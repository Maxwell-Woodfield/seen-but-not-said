"""Amendment 1 (notes/diagnosis_plan.md §7): pixel overlap between each Phase 3 tune subset and val/test.

    python scripts/check_tune_overlap.py

The tune subsets were drawn by index before pixel hashing existed. This finds every tune image whose raw
pixels (sha1, as scripts/find_duplicates.py) also occur in the val or test split, and writes
`data/subsets/<flag>_tune_overlap.json`. Those tune indices are excluded from every tune-fit -> val-confirm
comparison (A1, E) and from calibration fits. Reads image arrays only; no model, no GPU.
"""
import json
from collections import defaultdict

from effvar import FLAGS
from effvar.data import load_split, subset_path
from effvar.diag_subsets import pixel_hashes


def main() -> None:
    for flag in FLAGS:
        tune = json.loads(subset_path(flag, "tune").read_text())
        train_h = pixel_hashes(load_split(flag, "train").imgs)
        where: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for split in ("val", "test"):
            for i, h in enumerate(pixel_hashes(load_split(flag, split).imgs)):
                where[h].append((split, i))
        hits = [{"tune_index": i, "label": lab, "matches": [{"split": s, "index": j} for s, j in where[train_h[i]]]}
                for i, lab in zip(tune["indices"], tune["labels"]) if train_h[i] in where]
        out = {"dataset": flag, "tune_n": tune["n"], "n_overlap": len(hits),
               "n_overlap_val": sum(any(m["split"] == "val" for m in h["matches"]) for h in hits),
               "n_overlap_test": sum(any(m["split"] == "test" for m in h["matches"]) for h in hits),
               "exclude_tune_indices": [h["tune_index"] for h in hits], "overlaps": hits}
        path = subset_path(flag, "tune_overlap")
        path.write_text(json.dumps(out, indent=2))
        print(f"{flag}: {out['n_overlap']} of {tune['n']} tune images pixel-identical to val/test "
              f"(val {out['n_overlap_val']}, test {out['n_overlap_test']}) -> {path.name}")


if __name__ == "__main__":
    main()
