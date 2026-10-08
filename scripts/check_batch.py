"""Gate for measurement 3's batched path. TIMING PREP ONLY -- no accuracy is evaluated here.

The per-row path (`Ablation(per_row=True)`) is R with each image's sink set held as a mask row. It is
not bitwise R: its masked reductions sum in a different order (tests/test_var.py pins the math at
rel. err <= 1e-5), and batched GEMMs differ numerically from single-image ones even for `base`. So the
check is calibrated against `base`'s own batching noise, on the smoke images:

- B = 1: per-row R vs index-path R, first-token argmax on every image (the math is the same).
- each B: first-token argmax disagreements between batched and single-image outputs, per row. PASS
  if R's never exceed base's + 1 at that B (batching must not perturb R more than it perturbs base).
- peak VRAM per B, so the m3 commands know whether B = 8 fits in 12 GB.

Result 2026-09-21: PASS; B = 1 is bitwise (max|dlogit| 0.000), argmax agrees 24/24 at every B, peak VRAM
8.13 / 8.23 / 8.45 / 8.88 GiB at B = 1/2/4/8. But R's logits move by up to ~5 at B > 1 against base's
~0.4. Diagnosed (not a row-mixing bug: 4 copies of one image give identical rows): batching's GEMM noise
flips VAR's HARD thresholds -- the detector's tau in the image-specific layers 12/13/15, and the gate --
and R's own effect is 3.5-5.9 logits, so one flipped sink moves a lot. The work done is of the same
kind, so timing is valid; it also means R's outputs are numerically fragile under batching.

    python scripts/check_batch.py [--limit 8] [--batches 1 2 4 8]
"""
import argparse

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
    ap.add_argument("--limit", type=int, default=8, help="smoke images per dataset (multiple of every B)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()
    if any(args.limit % b for b in args.batches):
        raise SystemExit("--limit must be a multiple of every batch size")

    processor = load_processor()
    model = load_model("var_eager")
    base = make_method("base", attn="sdpa")
    r_index = make_method("var", attn="var_eager")
    r_rows = make_method("var", attn="var_eager", ablate=Ablation(per_row=True))

    def gen(method, spec, imgs):
        model.set_attn_implementation(ATTN_MODES[method.attn])
        return run_batch(model, processor, method, spec, imgs)

    data = {f: (get_spec(f), [im for _, _, im in iter_samples(load_subset(f, "smoke"), limit=args.limit)])
            for f in FLAGS}
    # Single-image references: base, and R on the index path (R's implementation).
    ref = {f: {"base": [gen(base, s, [im]).first_logits[0] for im in ims],
               "R": [gen(r_index, s, [im]).first_logits[0] for im in ims]} for f, (s, ims) in data.items()}

    ok = True
    for B in args.batches:
        stats = {"base": [0, 0.0, 0.0], "R": [0, 0.0, 0.0]}   # argmax disagreements, max |dlogit|, VRAM
        try:
            for f, (spec, ims) in data.items():
                for k in range(0, len(ims), B):
                    for name, method in (("base", base), ("R", r_rows)):
                        g = gen(method, spec, ims[k:k + B])
                        st = stats[name]
                        st[2] = max(st[2], g.peak_vram_gib)
                        for j in range(B):
                            single = ref[f][name][k + j]
                            st[0] += int(g.first_logits[j].argmax() != single.argmax())
                            st[1] = max(st[1], float((g.first_logits[j] - single).abs().max()))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"B = {B}: OUT OF MEMORY -- leave B = {B} out of measurement 3")
            continue
        n = len(FLAGS) * args.limit
        b_ok = stats["R"][0] <= stats["base"][0] + 1 and (B != 1 or stats["R"][0] == 0)
        ok &= b_ok
        print(f"B = {B}: base argmax diff {stats['base'][0]}/{n}, max|dlogit| {stats['base'][1]:.3f}, "
              f"peak {stats['base'][2]:.2f} GiB | R per-row argmax diff {stats['R'][0]}/{n}, "
              f"max|dlogit| {stats['R'][1]:.3f}, peak {stats['R'][2]:.2f} GiB -> {'ok' if b_ok else 'FAIL'}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
