"""P0 equivalence gate for the Phase 6 `diag` family (notes/diagnosis_plan.md §2). EQUIVALENCE ONLY.

    python scripts/check_diag.py [--smoke-limit 30] [--arm-limit 2]

Three bitwise checks (sha1 of emitted ids + exact first-token logits, as scripts/check_ablate.py):
1. `R` vs history: `var` (R) on the rehearsal images (first 20 val images per dataset) must reproduce the
   committed `results/rehearsal/var_tau10/summary.csv` logits_sha1. The `new_state` refactor of VarMethod
   must not have moved `R`.
2. `diag` at defaults vs `R` on the smoke subsets: the diag code path IS R's at defaults.
3. Every diag arm at p = 0 vs `base` (text eager): each arm's code runs on the real model and is the
   identity when the budget is zero. (No arm is run at p > 0: that is P2, not an equivalence check.)

Writes results/diag/P0/check_diag.txt. Exit 0 only if every check is exact. G0 is scripts/check_gates.py.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import torch

from effvar import FLAGS, ROOT
from effvar.data import iter_samples, load_subset
from effvar.generation import run_one
from effvar.methods import make_method
from effvar.model import ATTN_MODES, load_model, load_processor
from effvar.prompts import get_spec

ARMS = {  # every non-default diag knob, each alone, at p = 0
    "rows=answer": {"rows": "answer"}, "rows=prompt": {"rows": "prompt"}, "answer_summ=0": {"answer_summ": 0.0},
    "source=visual": {"source": "visual"}, "source=text": {"source": "text"},
    "sinks=random": {"sinks": "random"}, "sinks=const": {"sinks": "const"},
    "target=uniform": {"target": "uniform"}, "target=sharpen": {"target": "sharpen"},
    "target=centre": {"target": "centre", "centre_sigma": 4.0}, "target=text": {"target": "text"},
}
OUT = ROOT / "results" / "diag" / "P0" / "check_diag.txt"


def digest(gen) -> str:
    h = hashlib.sha1()
    h.update(repr(list(gen.new_ids)).encode())
    h.update(gen.first_logits.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-limit", type=int, default=30)
    ap.add_argument("--arm-limit", type=int, default=2)
    args = ap.parse_args()

    processor = load_processor()
    model = load_model("var_eager")
    R = make_method("var", attn="var_eager")
    diag = make_method("diag", attn="var_eager")
    base = make_method("base", attn="text_eager")
    arms = {k: make_method("diag", attn="var_eager", p=0.0, **v) for k, v in ARMS.items()}
    lines, ok = [], True

    def run(method, spec, image, attn="var_eager"):
        model.set_attn_implementation(ATTN_MODES[attn])
        return digest(run_one(model, processor, method, spec, image, 8))

    # 1. R vs the committed rehearsal
    ref = {(r["dataset"], int(r["index"])): r["logits_sha1"]
           for r in csv.DictReader(open(ROOT / "results/rehearsal/var_tau10/summary.csv", encoding="utf-8"))}
    hit = n = 0
    for flag in FLAGS:
        spec = get_spec(flag)
        for idx, _, image in iter_samples(load_subset(flag, "val"), limit=20):
            n += 1
            hit += run(R, spec, image) == ref[(flag, idx)]
    ok &= hit == n
    lines.append(f"1. R vs committed rehearsal (results/rehearsal/var_tau10): {hit}/{n} sha1 exact")

    # 2. diag at defaults vs R, and 3. every arm at p = 0 vs base
    hit = n = 0
    arm_hits = {k: [0, 0] for k in arms}
    for flag in FLAGS:
        spec = get_spec(flag)
        for i, (_, _, image) in enumerate(iter_samples(load_subset(flag, "smoke"), limit=args.smoke_limit)):
            n += 1
            hit += run(diag, spec, image) == run(R, spec, image)
            if i < args.arm_limit:
                b = run(base, spec, image, attn="text_eager")
                for k, m in arms.items():
                    arm_hits[k][0] += run(m, spec, image) == b
                    arm_hits[k][1] += 1
    ok &= hit == n
    lines.append(f"2. diag at defaults vs R (smoke): {hit}/{n} sha1 exact")
    arm_ok = all(h == t for h, t in arm_hits.values())
    ok &= arm_ok
    lines.append(f"3. every diag arm at p = 0 vs base text-eager (smoke, {args.arm_limit}/dataset): "
                 + ", ".join(f"{k} {h}/{t}" for k, (h, t) in arm_hits.items()))
    lines.append("PASS" if ok else "FAIL")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
