"""HeadFork gate sweep (rho x summ) on the TRAIN tune subsets, before the H0 gate.

Why: p and rho are the paper's LLaVA values. On MedGemma 1-2 image tokens hold 52-62% of all image
attention, so rho = 0.5 (keep iff sink share <= 1 - rho) bisects the distribution and summ = 0.1 sits
on Retina's mean image mass -- both thresholds are on a cliff, and H0 at the defaults could fail
merely because they do not transfer (notes/status.md).

Tuning runs only: untimed, vision features cached, results under results/tune/ (enforced). Each
config is written as it finishes and skipped on restart, so an interrupted sweep resumes.

    python scripts/sweep_gate.py [--methods var var_attn] [--datasets ...] [--no-cache] [--redo]
"""
import argparse
import itertools
import time

import torch

from effvar import FLAGS
from effvar.evaluate import evaluate_dataset
from effvar.generation import MAX_NEW_TOKENS, SEED, WARMUP_RUNS
from effvar.methods import make_method
from effvar.model import load_model, load_processor
from effvar.runio import RESULTS_DIR, write_config, write_metrics, write_predictions
from effvar.vision_cache import VisionCache

RHOS = (0.1, 0.2, 0.5, 0.7)
SUMMS = (0.05, 0.1, 0.2)
TAU, P = 10.0, 0.4
SUBSET = "tune"


def config_name(rho: float, summ: float) -> str:
    return f"rho{rho:g}_summ{summ:g}"


def out_dir(method: str, rho: float, summ: float, flag: str):
    return RESULTS_DIR / "tune" / method / config_name(rho, summ) / flag


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["var", "var_attn"])
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="recompute vision features every time")
    ap.add_argument("--redo", action="store_true", help="recompute configs that already have results")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    processor = load_processor()
    model = load_model("var_eager")
    cache = None if args.no_cache else VisionCache.for_model(model, processor)

    jobs = [(m, rho, summ) for m in args.methods for rho, summ in itertools.product(RHOS, SUMMS)]
    done = [j for j in jobs if not args.redo
            and all((out_dir(j[0], j[1], j[2], f) / "metrics.json").exists() for f in args.datasets)]
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} configs ({len(args.methods)} methods x {len(RHOS)}x{len(SUMMS)} grid), "
          f"{len(done)} already done, {len(todo)} to run; vision_cache={cache is not None}")

    started = time.perf_counter()
    for k, (method_name, rho, summ) in enumerate(todo, 1):
        t0 = time.perf_counter()
        method = make_method(method_name, attn="var_eager", rho=rho, summ=summ, tau=TAU, p=P)
        scores = []
        with cache.attached(model) if cache else _null():
            for flag in args.datasets:
                path = out_dir(method_name, rho, summ, flag)
                if not args.redo and (path / "metrics.json").exists():
                    continue
                rows, metrics, extra, spec, subset, token_sets = evaluate_dataset(
                    model, processor, method, flag, SUBSET, limit=args.limit, cache=cache)
                path.mkdir(parents=True, exist_ok=True)
                write_predictions(path, rows)
                write_metrics(path, metrics)
                write_config(path, method=method, spec=spec, subset=subset, token_sets=token_sets,
                             max_new_tokens=MAX_NEW_TOKENS, seed=SEED, warmup_runs=WARMUP_RUNS,
                             limit=args.limit, extra=extra)
                scores.append(metrics["qwk"] if flag == "retinamnist" else metrics["macro_f1"])

        elapsed = time.perf_counter() - started
        remaining = (elapsed / k) * (len(todo) - k)
        score = f"{sum(scores) / len(scores):.3f}" if scores else "cached"
        print(f"[{k}/{len(todo)}] {method_name} rho={rho} summ={summ}: score={score} "
              f"({fmt(time.perf_counter() - t0)}; elapsed {fmt(elapsed)}, ~{fmt(remaining)} left)",
              flush=True)

    if cache:
        print("vision cache:", cache.stats())
    print(f"sweep finished in {fmt(time.perf_counter() - started)}")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
