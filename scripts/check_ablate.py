"""Bitwise gate for the timing ladder. TIMING ONLY -- no accuracy is evaluated here.

Every arm declares, in `effvar.ablate.arms()`, which arm's output it must reproduce EXACTLY:

- `Ev`, `Hk`, `Dt`, `DtS`, `DtG` all disable the edit, so every one of them must be byte-identical to
  stock eager. This is a complete correctness net over the ablation machinery: if a `stop_after` return
  landed in the wrong place, or a hook changed something it should not, the group splits.
- `R`, `R_dup`, `R_legacy`, `M24` and `M24_legacy` must be byte-identical to `R`. This is the
  load-bearing one. Since 2026-09-20 the lean path IS `R`, and `legacy` restores the first port's host
  work: it changes only HOW the work is done, so if it changed WHAT is computed, R's accuracy results
  (all produced by the first port) would no longer describe the current R. m = 24 is bitwise R already.
- `VA` / `VA_legacy`: the same swap check for `var_attn`, whose prep is not cached (live I_v).
- `S` (SDPA numerics) and `R_det1215` (a different sink set, deliberately) are expected to differ and
  are checked only for not crashing.

    python scripts/check_ablate.py [--limit N] [--datasets retinamnist ...] [--arms R R_legacy ...]

Exit 0 only if every declared bitwise group is exact.
"""
import argparse
import hashlib

import torch

from effvar import FLAGS
from effvar.ablate import arms, method_kwargs
from effvar.data import iter_samples, load_subset
from effvar.generation import run_one
from effvar.methods import make_method
from effvar.model import ATTN_MODES, load_model, load_processor
from effvar.prompts import get_spec
from effvar.var import stats_summary


def digest(gen) -> str:
    """Identity of a generation: the emitted ids plus the exact first-token logits."""
    h = hashlib.sha1()
    h.update(repr(list(gen.new_ids)).encode())
    h.update(gen.first_logits.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6, help="images per dataset")
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS))
    ap.add_argument("--arms", nargs="+", default=None, help="subset of arms (default: all)")
    args = ap.parse_args()

    table = arms()
    if args.arms:
        table = {k: v for k, v in table.items() if k in args.arms}
        missing = {v["bitwise"] for v in table.values() if v["bitwise"]} - set(table)
        if missing:
            raise SystemExit(f"--arms must include the reference arms {sorted(missing)}")
    processor = load_processor()
    model = load_model("var_eager")   # superset; swapped per arm, as scripts/check_gates.py does
    methods = {name: make_method(a["method"], attn=a["attn"], **method_kwargs(a)) for name, a in table.items()}

    ok, checked, mismatches = True, 0, []
    for flag in args.datasets:
        spec = get_spec(flag)
        subset = load_subset(flag, "smoke")
        stats_seen: dict[str, dict] = {}
        for idx, label, image in iter_samples(subset, limit=args.limit):
            seen: dict[str, str] = {}
            for name, arm in table.items():
                model.set_attn_implementation(ATTN_MODES[arm["attn"]])
                gen = run_one(model, processor, methods[name], spec, image, 8)
                state = getattr(methods[name], "last_state", None)
                if state is not None:
                    state.finalise_stats()
                    stats_seen.setdefault(name, stats_summary(state.stats))
                seen[name] = digest(gen)
            for name, arm in table.items():
                ref = arm["bitwise"]
                if ref is None:
                    continue
                checked += 1
                if seen[name] != seen[ref]:
                    ok = False
                    mismatches.append(f"{flag} img {idx}: {name} != {ref}")
        print(f"{flag}: {args.limit} images x {len(table)} arms")
        for name in table:
            s = stats_seen.get(name)
            if s:
                print(f"    {name:<16} edit_events={s['edit_events']:<4} rows_edited={s['rows_edited']:<7}"
                      f" n_vis={s['n_vis']} n_text={s['n_text']}")

    for m in mismatches[:20]:
        print("  MISMATCH:", m)
    print(f"\n{'PASS' if ok else 'FAIL'}: {checked - len(mismatches)}/{checked} bitwise checks exact")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
