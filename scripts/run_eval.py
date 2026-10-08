"""Evaluate one method on MedMNIST subsets with MedGemma-1.5-4b-it.

    python scripts/run_eval.py --method base --attn sdpa --subset smoke --exp-name smoke/base_sdpa
        [--datasets retinamnist pneumoniamnist dermamnist] [--limit N] [--overwrite] [--prompt reversed]

Writes results/<exp-name>/<dataset>/{predictions.jsonl,config.yaml,metrics.json}.

`--cache-vision` reuses projected image features across runs (effvar/vision_cache.py). It switches to
the untimed path and may only write under results/tune/, because cached latencies are fiction.
"""
import argparse

import torch

from effvar import FLAGS
from effvar.diag import parse_overrides
from effvar.evaluate import evaluate_dataset
from effvar.generation import MAX_NEW_TOKENS, SEED, WARMUP_RUNS
from effvar.methods import REGISTRY, make_method
from effvar.model import ATTN_MODES, load_model, load_processor
from effvar.prompt_variants import VARIANTS, variant_spec
from effvar.runio import prepare_dir, write_config, write_metrics, write_predictions
from effvar.vision_cache import VisionCache

TUNE_PREFIX = "tune/"


def parse_layers(spec: str) -> tuple[int, ...]:
    """'9-32' or '15,29' or a mix -> a sorted tuple of layer indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        elif part:
            out.add(int(part))
    if not out:
        raise ValueError(f"no layers parsed from {spec!r}")
    return tuple(sorted(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--attn", required=True, choices=sorted(ATTN_MODES))
    ap.add_argument("--subset", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--limit", type=int, default=None, help="harness checks only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cache-vision", action="store_true",
                    help="tuning only: reuse cached image features, no timings, results/tune/ only")
    ap.add_argument("--tau", type=float, default=None,
                    help="var/var_attn/var_literal only: override the sink-detector threshold")
    ap.add_argument("--layers", default=None,
                    help="effvar_rank1_topm: restrict the mechanism to these layers, e.g. '9-32' or "
                         "'15,29'. Layers outside the set skip the detector and the gate, not just the edit.")
    ap.add_argument("--profile", action="store_true",
                    help="P16: per-region CUDA-event timing of the VAR path. Inflates absolute latency, "
                         "so its numbers are shares only and never a latency result.")
    ap.add_argument("--log-sinks", action="store_true",
                    help="log the per-layer sink SETS (prefill only). Untimed path only, like --log-delta.")
    ap.add_argument("--diag", nargs="+", default=None, metavar="KEY=VALUE",
                    help="--method diag only: Phase 6 knobs (effvar/diag.py), e.g. target=uniform p=0.6")
    ap.add_argument("--prompt", default="frozen", choices=sorted(VARIANTS),
                    help="Phase 6 prompt variant (effvar/prompt_variants.py); 'frozen' = effvar/prompts.py")
    ap.add_argument("--log-delta", action="store_true",
                    help="H2 instrumentation: per-layer ||delta_o||. Costs a matmul per edited layer, so "
                         "it requires --cache-vision (the untimed path) and never touches a timed run.")
    args = ap.parse_args()

    if args.profile and args.cache_vision:
        ap.error("--profile needs the real attention path; drop --cache-vision")
    if args.log_delta and not args.cache_vision:
        ap.error("--log-delta changes the attention path's cost; use it with --cache-vision (untimed)")
    if args.log_sinks and not args.cache_vision:
        ap.error("--log-sinks syncs on every layer; use it with --cache-vision (untimed)")

    if args.cache_vision and not args.exp_name.startswith(TUNE_PREFIX):
        ap.error(f"--cache-vision may only write under {TUNE_PREFIX!r}; got --exp-name {args.exp_name!r}")

    torch.manual_seed(SEED)
    overrides = {} if args.tau is None else {"tau": args.tau}
    if args.log_delta:
        overrides["record_delta"] = True
    if args.log_sinks:
        overrides["record_sinks"] = True
    if args.profile:
        overrides["profile"] = True
    if args.layers is not None:
        overrides["layers"] = parse_layers(args.layers)
    if args.diag is not None:
        if args.method != "diag":
            ap.error("--diag applies to --method diag only")
        try:
            overrides.update(parse_overrides(args.diag))
        except ValueError as err:
            ap.error(str(err))
    if overrides and args.method == "base":
        ap.error("--tau / --layers / --log-delta / --diag do not apply to --method base")
    method = make_method(args.method, attn=args.attn, **overrides)
    processor = load_processor()
    model = load_model(method.attn)
    cache = VisionCache.for_model(model, processor) if args.cache_vision else None
    warmed = cache is not None  # cached runs measure nothing, so they skip the warm-up

    with cache.attached(model) if cache else _null():
        for flag in args.datasets:
            out_dir = prepare_dir(args.exp_name, flag, args.overwrite)
            rows, metrics, extra, spec, subset, token_sets = evaluate_dataset(
                model, processor, method, flag, args.subset, limit=args.limit, cache=cache, warmed=warmed,
                spec=variant_spec(args.prompt, flag))
            warmed = True
            if args.prompt != "frozen":
                extra["prompt_variant"] = args.prompt
            write_predictions(out_dir, rows)
            write_metrics(out_dir, metrics)
            write_config(out_dir, method=method, spec=spec, subset=subset, token_sets=token_sets,
                         max_new_tokens=MAX_NEW_TOKENS, seed=SEED, warmup_runs=WARMUP_RUNS,
                         limit=args.limit, extra=extra)
            timing = "" if cache else (
                f"vision_ms={metrics['vision_ms_median']:.0f} prefill_ms={metrics['prefill_ms_median']:.0f} "
                f"lm_prefill_ms={metrics['lm_prefill_ms_median']:.0f} "
                f"decode_ms/tok={metrics['decode_ms_per_token_median']:.1f} "
                f"lm_ms={metrics['lm_ms_median']:.0f} e2e_ms={metrics['e2e_ms_median']:.0f} ")
            print(f"{flag}: n={metrics['n']} acc={metrics['accuracy']:.3f} "
                  f"macro_f1={metrics['macro_f1']:.3f} compliance={metrics['compliance_rate']:.2f} "
                  f"leakage={metrics['leakage_mean']:.4f} {timing}"
                  f"vram={metrics['peak_vram_gib_max']:.2f} vision_cache={metrics['vision_cache']}")
    if cache:
        print("vision cache:", cache.stats())


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
