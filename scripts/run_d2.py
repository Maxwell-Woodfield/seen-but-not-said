"""D2 — per-class yes/no prompts, one arm per call (notes/diagnosis_plan.md §5 D2; Amendments 3b-c, 4d-f). Tune only.

    python scripts/run_d2.py --method base --subset tune --cache-vision --exp-name tune/diag/D2/base_yn [--dry-run]
    python scripts/run_d2.py --method var  --subset tune --cache-vision --exp-name tune/diag/D2/R_yn
    (smoke, Amendment 4f: add --limit 10 and use --exp-name tune/diag/D2/smoke/{base_yn,R_yn})

Per image and class c, one prompt "Does this {modality} show {class}? Answer yes or no."
(`effvar.prompt_variants.yesno_specs`) with the frozen chat template and "Answer:" prefill; greedy,
MAX_NEW_TOKENS, the untimed path (no latency is measured). With --cache-vision the image features are read from
the vision cache for each of the K prompts of an image. Scored by effvar/yesno.py. Both arms run attn var_eager,
as the saved letter tune runs (Amendment 4e); `var` = R at its defaults.

Refuses to run unless the tree is clean: no tracked edits, and nothing modified or untracked under effvar/,
scripts/, tests/ (Amendment 4d). config.yaml records HEAD and the code trees (`git rev-parse HEAD:<dir>`) at the
start and at the end of each dataset. Writes results/<exp-name>/<dataset>/{predictions.jsonl (one row per
image, K-vectors in class order), config.yaml, metrics.json}. metrics.json holds code-check quantities only
(compliance, leakage, range of s, seconds per prompt); every metric that reads labels is scripts/diag_d2.py's.

--dry-run loads the processor and the model config only (no weights, no GPU): it renders the K prompts, checks
the yes/no ids, the "Answer:" ending and the image span, counts prompts, checks the output dirs and the git
state, and prints a wall-time estimate. Exit 1 if the real run would refuse.
"""
import argparse
import json
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import transformers
import yaml
from PIL import Image

from effvar import FLAGS, MODEL_ID, ROOT
from effvar.data import iter_samples, load_subset
from effvar.generation import MAX_NEW_TOKENS, SEED, run_untimed
from effvar.model import TEMPLATE_KWARGS, build_inputs, image_span, load_processor
from effvar.prompt_variants import YESNO_INSTRUCTION, YESNO_QUESTION, yesno_specs
from effvar.prompts import ANSWER_PREFIX
from effvar.runio import RESULTS_DIR, prepare_dir, write_metrics, write_predictions
from effvar.yesno import NO_TOKENS, YES_TOKENS, image_row, yesno_first_token, yesno_summary

ATTN = "var_eager"
TUNE_PREFIX = "tune/"
CODE_DIRS = ("effvar", "scripts", "tests")
SMOKE_EXP = "tune/diag/D2/smoke"
# Wall-time fallback when no tune smoke has run: C4's cached rate (P2, 2026-09-28: 1,800 {base, R} letter
# generations in ~0.21 GPU-h incl. model loads and analysis, i.e. ~0.42 s per generation).
SEC_PER_PROMPT_FALLBACK = 0.42
LOAD_S = 60.0


def git_code_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
            "code_changes": run("status", "--porcelain", "--untracked-files=all", "--", *CODE_DIRS).splitlines(),
            "code_trees": {d: run("rev-parse", f"HEAD:{d}") for d in CODE_DIRS}}


def is_clean(git: dict) -> bool:
    return not git["dirty"] and not git["code_changes"]


def check_token_ids(tokenizer) -> None:
    for tokens in (YES_TOKENS, NO_TOKENS):
        for tid, form in tokens.items():
            enc = tokenizer(form, add_special_tokens=False)["input_ids"]
            if tokenizer.decode([tid]) != form or enc != [tid]:
                raise SystemExit(f"yes/no token check failed: id {tid} decodes to {tokenizer.decode([tid])!r}, "
                                 f"{form!r} encodes to {enc}")


def check_prompts(processor, config, flag) -> list[str]:
    """Render every class prompt with a gray image: prefilled turn, one contiguous image span."""
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    texts = []
    for spec in yesno_specs(flag):
        kwargs = {**TEMPLATE_KWARGS, "tokenize": False, "return_dict": False, "return_tensors": None}
        rendered = processor.apply_chat_template(spec.messages(image), **kwargs)
        assert rendered.endswith(ANSWER_PREFIX) and spec.text() in rendered, (flag, spec.cls)
        _, n_img = image_span(build_inputs(processor, spec.messages(image), "cpu")["input_ids"], config)
        assert n_img == config.mm_tokens_per_image, (flag, spec.cls, n_img)
        texts.append(spec.text())
    return texts


def sec_per_prompt(method_name: str) -> tuple[float, str]:
    arm = "base_yn" if method_name == "base" else "R_yn"
    vals = []
    for flag in FLAGS:
        path = RESULTS_DIR / SMOKE_EXP / arm / flag / "metrics.json"
        if path.exists():
            vals.append(json.loads(path.read_text(encoding="utf-8"))["sec_per_prompt"])
    if vals:
        return float(np.mean(vals)), f"measured in {SMOKE_EXP}/{arm} (mean over {len(vals)} datasets)"
    return SEC_PER_PROMPT_FALLBACK, "fallback: C4's cached rate"


def dry_run(args, git, processor) -> int:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    check_token_ids(processor.tokenizer)
    ok, total = True, 0
    print(f"git HEAD {git['commit']}  tracked edits: {'YES' if git['dirty'] else 'none'}  "
          f"code changes/untracked under {'/'.join(CODE_DIRS)}: {git['code_changes'] or 'none'}")
    if not is_clean(git):
        print("WOULD REFUSE: the tree is not clean (Amendment 4d)")
        ok = False
    for flag in args.datasets:
        subset = load_subset(flag, args.subset)
        n = len(subset.indices[:args.limit])
        texts = check_prompts(processor, config, flag)
        total += n * len(texts)
        out = RESULTS_DIR / args.exp_name / flag
        exists = out.exists()
        print(f"\n{flag}: subset {subset.name}/{subset.split}, {n} images x K = {len(texts)} -> {n * len(texts)} "
              f"prompts; output {out.relative_to(ROOT).as_posix()} {'EXISTS' if exists else 'free'}")
        for c, t in enumerate(texts):
            print(f"  class {c}: {t}")
        if exists and not args.overwrite:
            print("WOULD REFUSE: output exists (pass --overwrite to replace it)")
            ok = False
    rate, src = sec_per_prompt(args.method)
    print(f"\nyes/no ids: 12/12 decode and encode as listed. image span: {config.mm_tokens_per_image} tokens.")
    print(f"total {total} prompts; vision cache {'ON' if args.cache_vision else 'off'}; method {args.method}, "
          f"attn {ATTN}")
    print(f"estimated wall time: {total} x {rate:.3f} s ({src}) + ~{LOAD_S:.0f} s load = "
          f"{(total * rate + LOAD_S) / 60:.1f} min")
    print("DRY RUN OK" if ok else "DRY RUN: the real run would refuse (see above)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=("base", "var"), help="var = R at its defaults")
    ap.add_argument("--subset", required=True, choices=("tune",))
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--limit", type=int, default=None, help="first N images per dataset (the Amendment 4f smoke)")
    ap.add_argument("--cache-vision", action="store_true", help="reuse cached image features; results/tune/ only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cache_vision and not args.exp_name.startswith(TUNE_PREFIX):
        ap.error(f"--cache-vision may only write under {TUNE_PREFIX!r}; got --exp-name {args.exp_name!r}")

    git = git_code_state()
    processor = load_processor()
    if args.dry_run:
        return dry_run(args, git, processor)
    if not is_clean(git):
        raise SystemExit(f"refusing to run: the tree is not clean (Amendment 4d). tracked edits: {git['dirty']}; "
                         f"code changes: {git['code_changes']}")
    check_token_ids(processor.tokenizer)

    from effvar.methods import make_method
    from effvar.model import load_model
    from effvar.vision_cache import VisionCache

    torch.manual_seed(SEED)
    method = make_method(args.method, attn=ATTN)
    model = load_model(ATTN)
    cache = VisionCache.for_model(model, processor) if args.cache_vision else None
    with cache.attached(model) if cache else nullcontext():
        for flag in args.datasets:
            out_dir = prepare_dir(args.exp_name, flag, args.overwrite)
            subset = load_subset(flag, args.subset)
            specs = yesno_specs(flag)
            rows, span, vram = [], None, 0.0
            t0 = time.perf_counter()
            for i, (idx, label, image) in enumerate(iter_samples(subset, limit=args.limit)):
                if i == 0:
                    span = image_span(build_inputs(processor, specs[0].messages(image), model.device)["input_ids"],
                                      model.config)
                per_class = []
                with cache.sample(flag, subset.split, idx) if cache else nullcontext():
                    for spec in specs:
                        gen = run_untimed(model, processor, method, spec, image)
                        per_class.append({**yesno_first_token(gen.first_logits), "text": gen.text,
                                          "new_ids": gen.new_ids, "logits_sha1": gen.digest()})
                        vram = max(vram, gen.peak_vram_gib)
                rows.append(image_row(idx, label, per_class))
            wall = time.perf_counter() - t0
            summary = yesno_summary(rows)
            metrics = {**summary, "wall_s": wall, "sec_per_prompt": wall / summary["n_prompts"],
                       "peak_vram_gib_max": vram, "vision_cache": cache is not None}
            if cache is not None:
                metrics["vision_cache_stats"] = cache.stats()
            cfg = {"model": MODEL_ID, "dtype": "bfloat16", "method": method.name, "method_params": method.params(),
                   "attn": ATTN, "dataset": flag,
                   "subset": {"name": subset.name, "split": subset.split, "n": len(subset.indices)},
                   "limit": args.limit, "prompt_variant": "yesno",
                   "prompt_template": f"{YESNO_QUESTION} {YESNO_INSTRUCTION}",
                   "prompts": [s.text() for s in specs], "answer_prefix": ANSWER_PREFIX,
                   "option_texts": list(specs[0].option_texts),
                   "yes_tokens": YES_TOKENS, "no_tokens": NO_TOKENS,
                   "score": "s_c = p(yes)/(p(yes)+p(no)), first generated token, float32 softmax over the vocabulary",
                   "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS}, "seed": SEED,
                   "torch": str(torch.__version__), "transformers": str(transformers.__version__),
                   "gpu": torch.cuda.get_device_name(0), "image_span": {"start": span[0], "P": span[1]},
                   "vision_cache": cache is not None, "git": git, "git_end": git_code_state(),
                   "argv": sys.argv[1:]}
            write_predictions(out_dir, rows)
            write_metrics(out_dir, metrics)
            (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            print(f"{flag}: n={summary['n']} prompts={summary['n_prompts']} compliance={summary['compliance_rate']:.4f}"
                  f"{' BROKEN' if summary['broken'] else ''} leakage mean={summary['leakage_mean']:.4f} "
                  f"max={summary['leakage_max']:.4f} s in [{summary['s_min']:.3g}, {summary['s_max']:.6g}] "
                  f"finite={summary['s_finite']} sec/prompt={metrics['sec_per_prompt']:.3f} vram={vram:.2f}")
            for r in rows[:3]:
                print(f"  index {r['index']} label {r['label']}: s = [" + ", ".join(f"{v:.3g}" for v in r["s"]) + "]")
    if cache:
        print("vision cache:", cache.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
