"""Regenerate the committed per-image summary of a run from its predictions.jsonl.

    python scripts/export_summary.py results/test/base_sdpa results/test/var_tau10 ...

`predictions.jsonl` is gitignored (`results/**/*.jsonl`), so the bulk rows stay local. This writes
`<run>/summary.csv`, one row per image over all datasets of the run, holding everything the test
report needs: the label, the predicted letter, the FULL per-option probability vector (so ranking
AUC is recomputable, not just the target probability), correctness, the first-token diagnostics,
per-image latency, and `logits_sha1`. Every CI, flip count and AUC in the test report is computed
from these committed files.

`p_option` is stored as a JSON array of Python float reprs, so it round-trips exactly.
Post-hoc only: reads saved files, no model, no GPU.
"""
import argparse
import csv
import json
from pathlib import Path

from effvar import FLAGS
from effvar.runio import load_config, load_predictions

COLUMNS = [
    "dataset", "index", "label", "label_letter", "pred", "pred_letter", "correct",
    "match_source", "first_token_in_options", "n_new",
    "target_p", "entropy", "leakage", "p_option", "logits_sha1",
    "vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "e2e_ms",
]


def rows_for(run: Path, flag: str) -> list[dict]:
    path = run / flag
    labels = load_config(path)["labels"]
    out = []
    for r in load_predictions(path):
        out.append({
            "dataset": flag,
            "index": r["index"],
            "label": r["label"],
            "label_letter": labels[r["label"]],
            "pred": r["pred"],
            "pred_letter": labels[r["pred"]],
            "correct": int(r["pred"] == r["label"]),
            "match_source": r["match_source"],
            "first_token_in_options": int(r["first_token_in_options"]),
            "n_new": len(r["new_ids"]),
            "target_p": repr(r["target_p"]),
            "entropy": repr(r["entropy"]),
            "leakage": repr(r["leakage"]),
            "p_option": json.dumps(r["p_option"]),
            "logits_sha1": r.get("logits_sha1", ""),
            **{k: ("" if r[k] is None else repr(r[k])) for k in
               ("vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "e2e_ms")},
        })
    return out


def export(run: Path) -> Path:
    rows = [row for flag in FLAGS if (run / flag).exists() for row in rows_for(run, flag)]
    if not rows:
        raise SystemExit(f"{run}: no dataset directories found")
    out = run / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, e.g. results/test/var_tau10")
    for run in ap.parse_args().runs:
        export(Path(run))


if __name__ == "__main__":
    main()
