"""`effvar_rank1_topm` m-curve on the TRAIN tune subsets, to fit m and settle the layer ordering.

`R` (var, tau 10) only ever edits layers 9-32 of 34 -- identically on all three datasets -- so the band
is 24 layers before any budget is imposed. Two candidate orderings come out of the H2 log
(`results/tune/R_delta`, written by `run_eval.py --log-delta`); on the **answer row** they nearly
coincide (Spearman +0.98, identical sets at m >= 12), but the choice is still made by measurement:

- `raw`   -- layers by mean ||delta_o|| on the answer row.
- `ratio` -- layers by ||delta_o|| / ||o|| on the answer row (scale-free; layer norms grow with depth).
- `control` -- 8 layers drawn uniformly from the band, seed 0. If the control matches the orderings at
  m = 8, the concentration argument does not hold and that must be reported plainly.

Ranking uses the **answer** phase: the prompt prefills "Answer:", so the prediction is the first
generated token and comes from the last prefill row. Decode-step edits cannot change it and are
excluded (`VarState.record_delta_o`).

m = 0 is `base` and is reused from `results/tune/base/`. m = 24 is the whole band, the same set for
both orderings, and should reproduce `R` (= `results/tune/var/rho0.5_summ0.1`) since layers 0-8 and 33
are never edited anyway -- that equality is itself a check on the layer restriction.

Caveat on the ranking: the answer-row `||delta_o||` is a layer's *direct* contribution to the row the
prediction is read from. Dropping a layer also changes the residual stream feeding later layers, so a
layer with ~0 direct contribution is not guaranteed to be free. That indirect path is exactly what the
curve measures and the ranking cannot.

Tuning only: untimed, vision cached, results under results/tune/ (enforced). Checkpointed per config.

    python scripts/sweep_topm.py [--datasets ...] [--redo]
"""
import argparse
import json
import random
import time

import torch

from effvar import FLAGS
from effvar.evaluate import evaluate_dataset
from effvar.generation import MAX_NEW_TOKENS, SEED, WARMUP_RUNS
from effvar.methods import make_method
from effvar.model import load_model, load_processor
from effvar.runio import RESULTS_DIR, write_config, write_metrics, write_predictions
from effvar.vision_cache import VisionCache

SUBSET = "tune"
TAU, P = 10.0, 0.4
MS = (2, 4, 8, 12, 16)      # m = 0 is base, m = 24 is the whole band; both reused, not re-run
FULL_M = 24
CONTROL_M = 8
DELTA_RUN = RESULTS_DIR / "tune" / "R_delta"


def orderings() -> dict[str, list[int]]:
    """Layer orders from the H2 log, averaged over datasets, ranked on the answer row."""
    per_layer: dict[int, dict[str, list[float]]] = {}
    for flag in FLAGS:
        by_layer = json.loads((DELTA_RUN / flag / "metrics.json").read_text())["var_delta_o_by_layer"]
        for layer, row in by_layer.items():
            acc = per_layer.setdefault(int(layer), {"raw": [], "ratio": []})
            acc["raw"].append(row["answer_delta_o_rms"])
            acc["ratio"].append(row["answer_ratio"])
    mean = {k: {n: sum(v) / len(v) for n, v in d.items()} for k, d in per_layer.items()}
    band = sorted(mean)
    out = {key: sorted(band, key=lambda ly: -mean[ly][key]) for key in ("raw", "ratio")}
    out["control"] = sorted(random.Random(SEED).sample(band, CONTROL_M))
    return out


def out_dir(tag: str, flag: str):
    return RESULTS_DIR / "tune" / "topm" / tag / flag


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    order = orderings()
    band = sorted(order["raw"])
    print(f"band: {len(band)} layers {band[0]}..{band[-1]}")
    for key in ("raw", "ratio"):
        print(f"  {key:7s} top-{max(MS)}: {order[key][:max(MS)]}")
    print(f"  control (seed {SEED}): {order['control']}")

    # On the answer row the two orderings mostly coincide, so identical layer sets are run once and
    # the tag records which orderings share them -- the curves are still reported separately.
    seen: dict[tuple[int, ...], str] = {}
    jobs: list[tuple[str, tuple[int, ...]]] = []
    for m in MS:
        for key in ("raw", "ratio"):
            layers = tuple(sorted(order[key][:m]))
            if layers in seen:
                print(f"  {key} m={m} == {seen[layers]} (same {m} layers), not re-run")
                continue
            other = "ratio" if key == "raw" else "raw"
            tag = (f"both_m{m}" if tuple(sorted(order[other][:m])) == layers else f"{key}_m{m}")
            seen[layers] = tag
            jobs.append((tag, layers))
    jobs.append((f"full_m{FULL_M}", tuple(band)))
    jobs.append((f"control_m{CONTROL_M}", tuple(order["control"])))

    torch.manual_seed(SEED)
    processor = load_processor()
    model = load_model("var_eager")
    cache = VisionCache.for_model(model, processor)

    todo = [j for j in jobs if args.redo
            or not all((out_dir(j[0], f) / "metrics.json").exists() for f in args.datasets)]
    print(f"{len(jobs)} configs, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    started = time.perf_counter()
    for k, (tag, layers) in enumerate(todo, 1):
        t0 = time.perf_counter()
        method = make_method("effvar_rank1_topm", attn="var_eager", tau=TAU, p=P, layers=layers)
        scores = []
        with cache.attached(model):
            for flag in args.datasets:
                path = out_dir(tag, flag)
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
        score = f"{sum(scores) / len(scores):.3f}" if scores else "cached"
        print(f"[{k}/{len(todo)}] {tag} (m={len(layers)}): score={score} "
              f"({fmt(time.perf_counter() - t0)}; elapsed {fmt(elapsed)}, "
              f"~{fmt((elapsed / k) * (len(todo) - k))} left)", flush=True)

    print("vision cache:", cache.stats())
    print(f"m-curve finished in {fmt(time.perf_counter() - started)}")


if __name__ == "__main__":
    main()
