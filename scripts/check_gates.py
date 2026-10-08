"""Model gates for the VAR port (notes/validation_plan.md §1).

    python scripts/check_gates.py [--limit 10]

G0  (HARD): `var` at p = 0 reproduces `base` (text eager, vision SDPA) exactly -- token ids and
            first-token logits bitwise -- on all 3 smoke subsets.
G0b (HARD): 0 < |I_v| < P in layers >= first_layer for most images, and HeadFork keeps some but not
            all heads. `var_literal` is expected to edit nothing; that is its result, not a failure.
"""
import argparse

import numpy as np
import torch

from effvar import FLAGS
from effvar.data import iter_samples, load_subset
from effvar.generation import run_one
from effvar.methods import make_method
from effvar.model import VAR_ATTN_NAME, load_model, load_processor
from effvar.prompts import get_spec

EAGER = {"text_config": "eager", "vision_config": "sdpa"}
VAR_EAGER = {"text_config": VAR_ATTN_NAME, "vision_config": "sdpa"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="images per dataset")
    args = ap.parse_args()

    processor = load_processor()
    model = load_model("var_eager")
    base = make_method("base", attn="text_eager")
    var0 = make_method("var", attn="var_eager", p=0.0)
    var = make_method("var", attn="var_eager")
    literal = make_method("var_literal", attn="var_eager")

    totals = {"ids": 0, "logits": 0, "n": 0}
    for flag in FLAGS:
        spec = get_spec(flag)
        subset = load_subset(flag, "smoke")
        ids = logits = n = literal_edits = changed = 0
        edited, n_vis, n_text = [], [], []
        for _, _, image in iter_samples(subset, limit=args.limit):
            model.set_attn_implementation(EAGER)
            g_base = run_one(model, processor, base, spec, image)
            model.set_attn_implementation(VAR_EAGER)
            g_zero = run_one(model, processor, var0, spec, image)
            g_var = run_one(model, processor, var, spec, image)
            run_one(model, processor, literal, spec, image)

            ids += g_base.new_ids == g_zero.new_ids
            logits += bool(torch.equal(g_base.first_logits, g_zero.first_logits))
            stats = var.last_state.stats
            edited.append(stats["layers_edited"])
            n_vis += stats["n_vis"]
            n_text += stats["n_text"]
            literal_edits += literal.last_state.stats["layers_edited"]
            changed += g_var.new_ids != g_base.new_ids
            n += 1

        totals["ids"] += ids
        totals["logits"] += logits
        totals["n"] += n
        print(f"{flag}: G0 ids {ids}/{n}, first-logits bitwise {logits}/{n} | "
              f"var layer-visits edited median {np.median(edited):.0f}, "
              f"|I_v| median {np.median(n_vis):.0f} max {max(n_vis)}, |I_t| median {np.median(n_text):.0f} | "
              f"answer changed vs base {changed}/{n} | var_literal edits {literal_edits}")

    ok = totals["ids"] == totals["logits"] == totals["n"]
    print(f"\nG0 {'PASS' if ok else 'FAIL'}: ids {totals['ids']}/{totals['n']}, "
          f"logits {totals['logits']}/{totals['n']}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
