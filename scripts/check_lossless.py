"""Lossless gate: a variant's predictions must be bitwise identical to R's. No model, reads results only.

m = 24 (`effvar_rank1_topm`, band 9-32) is bitwise R on all val and tune images (notes/status.md). On
test that claim is checked, not assumed (approved 2026-09-21): on every image the generated ids AND the
first-token logits must match. The logits are compared through `logits_sha1` (sha1 of the ids plus the
fp32 first-token logits, `effvar.generation.Generation.digest`), which run_eval writes since 2026-09-21.
Any divergence is a FINDING to report, never noise -- both runs are greedy, bf16, batch 1, one machine.

For older runs without `logits_sha1` it falls back to ids + `p_option` equality and says so: a weaker
check (option probabilities only, not the full vocabulary).

    python scripts/check_lossless.py results/test/var_tau10 results/test/effvar_topm24
    python scripts/check_lossless.py results/rehearsal/var_tau10 results/rehearsal/effvar_topm24

Exit 0 only if every image of every dataset present in both runs matches, and the image sets agree.
"""
import argparse
import json
from pathlib import Path

from effvar import FLAGS

ROOT = Path(__file__).resolve().parent.parent


def load(run: Path, flag: str) -> dict[int, dict] | None:
    path = run / flag / "predictions.jsonl"
    if not path.exists():
        return None
    return {r["index"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="the reference run, e.g. results/test/var_tau10 (R)")
    ap.add_argument("variant", help="the variant claimed lossless, e.g. results/test/effvar_topm24")
    args = ap.parse_args()
    ref_run, var_run = (ROOT / args.ref), (ROOT / args.variant)

    ok, compared = True, 0
    for flag in FLAGS:
        ref, var = load(ref_run, flag), load(var_run, flag)
        if ref is None or var is None:
            print(f"{flag}: skipped (missing in {'ref' if ref is None else 'variant'})")
            continue
        if set(ref) != set(var):
            print(f"{flag}: FAIL image sets differ ({len(ref)} vs {len(var)}; "
                  f"{len(set(ref) ^ set(var))} not shared)")
            ok = False
            continue
        strong = all("logits_sha1" in r for r in ref.values()) and all("logits_sha1" in r for r in var.values())
        bad = []
        for idx in sorted(ref):
            a, b = ref[idx], var[idx]
            same = a["new_ids"] == b["new_ids"] and (
                a["logits_sha1"] == b["logits_sha1"] if strong else a["p_option"] == b["p_option"])
            if not same:
                bad.append((idx, a["pred"], b["pred"], a["new_ids"] == b["new_ids"]))
        compared += len(ref)
        how = "ids + first-token logits (sha1)" if strong else "ids + p_option ONLY (no logits_sha1: weaker)"
        print(f"{flag}: {len(ref) - len(bad)}/{len(ref)} identical [{how}]")
        for idx, pa, pb, ids_same in bad[:20]:
            print(f"  DIVERGENCE index {idx}: pred {pa} -> {pb}, ids {'same' if ids_same else 'differ'}")
        if len(bad) > 20:
            print(f"  ... {len(bad) - 20} more")
        ok &= not bad

    if compared == 0:
        print("FAIL: no dataset present in both runs")
        return 1
    print(f"\n{'PASS: bitwise identical' if ok else 'FAIL: divergence is a FINDING, report it'} "
          f"({compared} images)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
