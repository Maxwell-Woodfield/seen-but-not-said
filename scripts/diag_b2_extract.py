"""B2 feature extraction (notes/diagnosis_plan.md §3 B2; Amendment 2). GPU, accuracy-only, untimed.

    python scripts/diag_b2_extract.py [--datasets ...] [--limit N]

One instrumented prefill per image through the SAME generate() path as the saved runs (max_new_tokens = 1),
so the first-token logits can be checked against them bit for bit:
- `base` over `<flag>_probe` (uncached vision; nothing is written to the vision cache) and over tune
  (cached vision), with `ObserveMethod(edit=False)`: R's detector runs, nothing is edited;
- `R` over tune with `ObserveMethod(edit=True, record_budget=True)` (= diag at defaults = R bitwise).
Per image: answer-row residual at all 35 hidden states; image-token residuals at layers 2, 5, 8, 9-32
pooled uniformly, by the head-averaged answer-row attention of that layer, and by that attention with R's
visual sinks zeroed (VAR's target distribution); projector output (= hidden_states[0] at the image block)
pooled mean and max; option-token logit lens at every hidden state; per-head answer-row image mass, S_v,
S_t, BOS, entropy; for R the realised decision-row budget. Stop rule: the base tune pass must reproduce
results/tune/base p_option exactly (and the R pass results/tune/var/rho0.5_summ0.1), else exit 1.
Features -> results/tune/diag/B2/<flag>/features_<pass>.pt (gitignored); checks -> results/diag/B2/extract_check.json.
"""
import argparse
import json
import time
from contextlib import nullcontext

import torch

from effvar import FLAGS, ROOT
from effvar.data import iter_samples, load_subset
from effvar.methods.diag import ObserveMethod
from effvar.model import build_inputs, load_model, load_processor
from effvar.prompts import get_spec
from effvar.scoring import first_token_metrics, option_token_sets
from effvar.vision_cache import VisionCache

POOL_LAYERS = (2, 5, 8) + tuple(range(9, 33))
FEAT_DIR = ROOT / "results" / "tune" / "diag" / "B2"
CHECK = ROOT / "results" / "diag" / "B2" / "extract_check.json"
PASSES = (("probe", "probe", False), ("tune", "tune", False), ("tune_R", "tune", True))


class Lens:
    """Option-token logit lens. Which reading of the last hidden state reproduces the model's own logits
    (already normed or not) is decided on the first image and asserted on every image after."""

    def __init__(self, model, sets):
        self.lm = model.model.language_model
        self.head = model.lm_head
        self.cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        self.sets = sets
        self.last_normed = None

    def logits(self, h, norm: bool):
        x = self.lm.norm(h) if norm else h
        z = self.head(x).float()
        if self.cap is not None:
            z = torch.tanh(z / self.cap) * self.cap
        return z

    def __call__(self, hs, model_logits):
        n = len(hs)
        if self.last_normed is None:
            a = (self.logits(hs[-1][0, -1], False) - model_logits).abs().max().item()
            b = (self.logits(hs[-1][0, -1], True) - model_logits).abs().max().item()
            self.last_normed = a <= b
            self.first_err = min(a, b)
        z_last = self.logits(hs[-1][0, -1], not self.last_normed)
        err = (z_last - model_logits).abs().max().item()
        rows = []
        for li in range(n):
            z = z_last if li == n - 1 else self.logits(hs[li][0, -1], True)
            p = torch.softmax(z, -1)
            rows.append(torch.stack([p[s].sum() for s in self.sets]))
        return torch.stack(rows).cpu(), err


def raw_p_option(run_dir, flag) -> dict[int, list[float]]:
    """The saved p_option lists, unnormalised: the stop rule compares these floats exactly."""
    with open(ROOT / run_dir / flag / "predictions.jsonl", encoding="utf-8") as f:
        return {int(r["index"]): r["p_option"] for r in map(json.loads, f)}


def extract(model, processor, method, spec, image, lens):
    inputs = build_inputs(processor, spec.messages(image), model.device)
    with torch.inference_mode(), method.activate(model, inputs):
        out = model.generate(**inputs, do_sample=False, max_new_tokens=1, return_dict_in_generate=True,
                             output_logits=True, output_hidden_states=True)
        st = method.last_state
        hs = out.hidden_states[0]
        assert len(hs) == 35, len(hs)
        logits = out.logits[0][0].float()
        im, P = st.im, st.P
        ans = torch.stack([h[0, -1] for h in hs]).half().cpu()                       # [35, D]
        proj = hs[0][0, im:im + P].float()
        pools = []
        for ly in POOL_LAYERS:
            X = hs[ly][0, im:im + P].float()                                          # input to layer ly
            a = st.answer[ly]["img_attn"].to(X.device)
            w_att = a / a.sum()
            w_sf = a.clone()
            w_sf[st.answer[ly]["vis"].to(X.device)] = 0.0
            w_sf = w_sf / w_sf.sum()
            pools.append(torch.stack([X.mean(0), w_att @ X, w_sf @ X]))
        lens_p, lens_err = lens(hs, logits)
        feats = {"ans": ans, "proj": torch.stack([proj.mean(0), proj.max(0).values]).half().cpu(),
                 "pools": torch.stack(pools).half().cpu(),                               # [27, 3, D]
                 "lens": lens_p, "heads": torch.stack([st.answer[ly]["heads"] for ly in range(34)]),  # [34, H, 5]
                 "logits": logits.cpu(), "lens_err": lens_err}   # CPU, as the harness computes p_option
        if getattr(st.cfg, "record_budget", False):
            feats["budget"] = st.budget_summary()
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test only")
    args = ap.parse_args()
    torch.manual_seed(0)
    processor = load_processor()
    model = load_model("var_eager")
    cache = VisionCache.for_model(model, processor)
    checks = json.loads(CHECK.read_text()) if CHECK.exists() and args.limit is None else {}
    ok = True
    with cache.attached(model):
        for flag in args.datasets:
            spec = get_spec(flag)
            sets = option_token_sets(processor.tokenizer, spec)
            refs = {"tune": raw_p_option("results/tune/base", flag),
                    "tune_R": raw_p_option("results/tune/var/rho0.5_summ0.1", flag)}
            for name, subset_name, edit in PASSES:
                subset = load_subset(flag, subset_name)
                method = ObserveMethod(edit=edit, record_budget=edit)
                lens = Lens(model, [torch.tensor(s, device=model.device) for s in sets])
                rows, t0 = [], time.time()
                exact = n = 0
                max_err = 0.0
                for idx, label, image in iter_samples(subset, limit=args.limit):
                    ctx = cache.sample(flag, subset.split, idx) if subset_name == "tune" else nullcontext()
                    with ctx:
                        f = extract(model, processor, method, spec, image, lens)
                    ft = first_token_metrics(f.pop("logits"), sets, label)   # full-vocab logits are not kept
                    f.update(index=idx, label=label, p_option=ft["p_option"])
                    if name in refs:
                        ref = refs[name][idx]
                        n += 1
                        exact += ft["p_option"] == ref
                        max_err = max(max_err, max(abs(a - b) for a, b in zip(ft["p_option"], ref)))
                    rows.append(f)
                out = FEAT_DIR / flag
                out.mkdir(parents=True, exist_ok=True)
                torch.save(rows, out / f"features_{name}.pt")
                rec = {"n": len(rows), "seconds": round(time.time() - t0, 1), "lens_last_normed": lens.last_normed,
                       "lens_max_err": max(r["lens_err"] for r in rows)}
                if name in refs:
                    rec.update(reproduces_saved=f"{exact}/{n}", max_abs_diff_p=max_err)
                    ok &= exact == n
                checks[f"{flag}/{name}"] = rec
                print(flag, name, rec, flush=True)
    if args.limit is None:
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        CHECK.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("STOP RULE:", "PASS" if ok else "FAIL (instrumented pass does not reproduce the saved outputs)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
