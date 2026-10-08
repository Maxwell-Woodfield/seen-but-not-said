"""Do VAR's sink sets survive batching-level numerical noise? READ-ONLY probe; no accuracy evaluated.

Runs each smoke image alone (R's index path) and inside a batch of B distinct images of the same
dataset (the per-row path, which is R per row: tests/test_var.py), then compares, per image:
- which layers' detector sink sets differ between the two,
- the first-token logits (max |d|) and argmax.
`base`'s own batching noise is printed for scale. Found 2026-09-21 by scripts/check_batch.py, whose
R logits moved ~10x more than base's under batching (notes/sink_geometry.md, "Finding").

    python scripts/probe_batch_sinks.py [--limit 8] [--batch 4]
"""
import argparse
from collections import Counter

import torch

from effvar import FLAGS
from effvar.ablate import Ablation
from effvar.data import iter_samples, load_subset
from effvar.generation import run_batch
from effvar.methods import make_method
from effvar.model import ATTN_MODES, load_model, load_processor
from effvar.prompts import get_spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    if args.limit % args.batch:
        raise SystemExit("--limit must be a multiple of --batch")

    processor = load_processor()
    model = load_model("var_eager")
    model.set_attn_implementation(ATTN_MODES["var_eager"])
    base = make_method("base", attn="var_eager")
    r_index = make_method("var", attn="var_eager")
    r_rows = make_method("var", attn="var_eager", ablate=Ablation(per_row=True))

    total_flips, n_img = Counter(), 0
    for flag in FLAGS:
        spec = get_spec(flag)
        ims = [im for _, _, im in iter_samples(load_subset(flag, "smoke"), limit=args.limit)]
        single, ssets, bsingle = [], [], []
        for im in ims:
            g = run_batch(model, processor, r_index, spec, [im])
            single.append(g.first_logits[0])
            ssets.append({ly: set(v.tolist()) for ly, v in r_index.last_state.sinks.items()})
            bsingle.append(run_batch(model, processor, base, spec, [im]).first_logits[0])
        flips, dl, db, amx, hit = Counter(), [], [], 0, 0
        for k in range(0, len(ims), args.batch):
            g = run_batch(model, processor, r_rows, spec, ims[k:k + args.batch])
            gb = run_batch(model, processor, base, spec, ims[k:k + args.batch])
            masks = r_rows.last_state.sinks
            for j in range(args.batch):
                i = k + j
                changed = [ly for ly in ssets[i]
                           if set(torch.nonzero(masks[ly][j]).flatten().tolist()) != ssets[i][ly]]
                flips.update(changed)
                hit += bool(changed)
                dl.append(float((g.first_logits[j] - single[i]).abs().max()))
                db.append(float((gb.first_logits[j] - bsingle[i]).abs().max()))
                amx += int(g.first_logits[j].argmax() != single[i].argmax())
                print(f"  {flag} img {i}: sink sets differ in layers {changed or '-'}; "
                      f"R max|dlogit| {dl[-1]:.3f} (base {db[-1]:.3f})")
        n_img += len(ims)
        total_flips.update(flips)
        print(f"{flag}: {hit}/{len(ims)} images with >= 1 flipped layer; flips per layer {dict(sorted(flips.items()))}; R max|dlogit| max {max(dl):.3f} "
              f"(base {max(db):.3f}); argmax changes {amx}/{len(ims)}")
    print(f"\nALL: {n_img} images at B = {args.batch}; flips per layer {dict(sorted(total_flips.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
