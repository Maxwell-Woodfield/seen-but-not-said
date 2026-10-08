"""Interleaved A/B rounds over the timing ladder. TIMING ONLY -- ACCURACY IS NOT EVALUATED HERE.

Measurement 1 of notes/status.md's next-session list: corrected per-call-site ceilings. The existing
"~2.3 ms/site" figure came from removing detector + gate + eager attention TOGETHER in 8 layers, so it
over-attributes; the arms in `effvar.ablate.arms()` separate them.

Why a new runner rather than repeated `run_eval.py` calls: `results/ab/` was six hand-issued runs whose
CI was computed outside the repo, and two runs of identical code on full val differed by 92 ms of LM
time (notes/status.md). Run-level drift is the dominant error term, so every arm must be measured many
times, interleaved, inside one process against one loaded model.

Order schedule: mirrored pairs on a rotated base. Round 2p uses `rotate(arms, p)`, round 2p+1 uses its
exact reverse, so an arm sitting at position i in the forward round sits at A-1-i in the reverse one.
Any drift that is linear in within-round position therefore cancels exactly within each pair, and the
rotation decorrelates arm from position across pairs. `results/ab_drift.log` shows why this matters: on
a FIXED order, R rose 763.7 -> 768.0 ms while E fell 754.7 -> 738.3 ms across three rounds, which
inflates the R-E difference.

Refuses to start on a dirty tree (tracked files modified) unless --allow-dirty is passed: the m1 run
recorded `dirty: true`, and its provenance had to be argued after the fact (effvar/ and the ab_*
scripts unchanged between its commit a3e9d78 and the next HEAD) instead of read off the manifest.

    python scripts/ab_rounds.py --dry-run
    python scripts/ab_rounds.py --set ladder --rounds 10 --limit 20 --out results/timing/m1_ladder
    python scripts/ab_rounds.py --set m24 --rounds 10 --limit 20 --out results/timing/m24_remeasure
    python scripts/ab_rounds.py --set m3 --batch 4 --rounds 10 --limit 16 --out results/timing/m3_batch

Measurement 3 (`--batch B`): each generate() call takes B DISTINCT images of one dataset, and the var arms
run the per-row path (`Ablation(per_row=True)`: each image keeps its own sink set). Each row of
rows.jsonl is one batch; its timing fields are PER IMAGE (batch time / B, the throughput view), with
the raw batch figures kept in `batch_*` fields (the per-request latency view).

Writes <out>/<runid>/{manifest.yaml, rows.jsonl}. Aggregate with scripts/ab_report.py.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from effvar import FLAGS, MODEL_ID
from effvar.ablate import ARM_SETS, Ablation, arms, method_kwargs
from effvar.data import iter_samples, load_subset
from effvar.generation import DECODE_STEPS_D, MAX_NEW_TOKENS, SEED, run_batch, run_one
from effvar.methods import make_method
from effvar.model import ATTN_MODES, load_model, load_processor
from effvar.prompts import get_spec
from effvar.runio import git_state
from effvar.var import stats_summary

ROOT = Path(__file__).resolve().parent.parent


def gpu_state() -> dict:
    """SM clock, temperature and power, so monotone thermal drift can be told from noise."""
    q = "clocks.sm,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        sm, temp, power = (p.strip() for p in out.split(","))
        return {"sm_mhz": float(sm), "temp_c": float(temp), "power_w": float(power)}
    except Exception as exc:                                  # never fail a run over telemetry
        return {"error": str(exc)}


def schedule(names: list[str], rounds: int) -> list[list[str]]:
    out = []
    for r in range(rounds):
        p = r // 2
        base = names[p % len(names):] + names[:p % len(names)]
        out.append(base if r % 2 == 0 else list(reversed(base)))
    return out


def digest(gen) -> str:
    h = hashlib.sha1()
    h.update(repr(list(gen.new_ids)).encode())
    h.update(gen.first_logits.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def batch_row(gen) -> dict:
    """Timing fields for one batch, PER IMAGE (batch time / B), plus the raw batch figures."""
    B = gen.batch
    decode_total = None if gen.decode_ms_per_token is None else gen.decode_ms_per_token * (gen.n_steps - 1)
    lm_prefill = gen.prefill_ms - gen.vision_ms
    lm = None if gen.decode_ms_per_token is None else lm_prefill + DECODE_STEPS_D * gen.decode_ms_per_token
    return {"batch": B, "n_new": gen.n_steps,
            "vision_ms": gen.vision_ms / B, "prefill_ms": gen.prefill_ms / B, "lm_prefill_ms": lm_prefill / B,
            "decode_ms_per_token": None if gen.decode_ms_per_token is None else gen.decode_ms_per_token / B,
            "decode_total_ms": None if decode_total is None else decode_total / B,
            "lm_ms": None if lm is None else lm / B, "e2e_ms": gen.e2e_ms / B,
            "batch_lm_prefill_ms": lm_prefill, "batch_decode_total_ms": decode_total, "batch_e2e_ms": gen.e2e_ms,
            "peak_vram_gib": gen.peak_vram_gib}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=20, help="images per dataset per round")
    ap.add_argument("--subset", default="val")
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS))
    ap.add_argument("--out", default="results/timing/m1_ladder")
    ap.add_argument("--set", default="ladder", choices=sorted(ARM_SETS), help="named arm set")
    ap.add_argument("--arms", nargs="+", default=None, help="explicit arms instead of --set")
    ap.add_argument("--batch", type=int, default=None,
                    help="measurement 3: B distinct images per generate(); var arms use the per-row path")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even with modified tracked files; the manifest then records dirty: true")
    ap.add_argument("--allow-test", action="store_true",
                    help="permit a test-split subset: the approved final session (--set test) only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    git = git_state()
    if git["dirty"] and not (args.allow_dirty or args.dry_run):
        raise SystemExit("refusing to start: tracked files are modified, so this run's code could not be "
                         "recovered from its commit. Commit first, or pass --allow-dirty.")

    everything = arms()
    wanted = args.arms or ARM_SETS[args.set]
    unknown = [a for a in wanted if a not in everything]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    table = {k: dict(everything[k]) for k in wanted}
    B = args.batch
    if B is not None:
        if B < 1 or args.limit % B:
            raise SystemExit(f"--limit ({args.limit}) must be a positive multiple of --batch ({B})")
        for name, a in table.items():
            if a["method"] == "base":
                continue
            if a["ablate"] is not None or a.get("layers") is not None:
                raise SystemExit(f"--batch supports plain R arms only; {name} is ablated or layer-restricted")
            a["ablate"] = Ablation(per_row=True)
    names = list(table)
    orders = schedule(names, args.rounds)
    per_block = len(args.datasets) * args.limit // (B or 1) + 1                 # +1 warmup per block
    n_gen = args.rounds * len(names) * per_block

    print(f"TIMING ONLY -- accuracy is not evaluated.\n{len(names)} arms x {args.rounds} rounds x "
          f"{len(args.datasets)} datasets x {args.limit} img" + (f", batch {B}" if B else "")
          + f" = {n_gen} generate() calls")
    if B in (None, 1):
        print(f"estimated wall clock: {n_gen * 0.75 / 60:.0f} min at ~0.75 s/generation")
    else:
        print(f"estimated wall clock: under {n_gen * B * 0.75 / 60:.0f} min (B = 1 pace; batching is faster)")
    for r, o in enumerate(orders):
        print(f"  round {r + 1}: {' '.join(o)}")
    if args.dry_run:
        return 0

    subsets = {}
    for flag in args.datasets:
        sub = load_subset(flag, args.subset)
        if sub.split == "test" and not args.allow_test:  # mechanical, not promised: test stays untouched
            raise SystemExit(f"refusing to time on a test split ({flag}); --allow-test is for the approved "
                             "final session only")
        subsets[flag] = sub

    processor = load_processor()
    model = load_model("var_eager")
    methods = {name: make_method(a["method"], attn=a["attn"], **method_kwargs(a)) for name, a in table.items()}

    # Materialise images once: iter_samples reopens the mmap'd npz per call, and that must never land
    # inside a timed round.
    print("loading images...")
    images, specs = {}, {}
    for flag in args.datasets:
        specs[flag] = get_spec(flag)
        images[flag] = list(iter_samples(subsets[flag], limit=args.limit))

    runid = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out / runid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    manifest = {
        "purpose": "timing only", "accuracy_evaluated": False,
        "note": "Per-component timing ablations for notes/options_memo.md's corrected per-call-site "
                "ceilings. Arms that disable parts of VAR produce meaningless predictions by design.",
        "model": MODEL_ID, "dtype": "bfloat16", "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "decode_steps_D": DECODE_STEPS_D,
        "rounds": args.rounds, "limit": args.limit, "subset": args.subset, "batch": B,
        "timing_unit": "per image (batch time / B); raw batch_* fields kept" if B else "per image",
        "datasets": {f: {"split": subsets[f].split, "n": len(subsets[f].indices),
                         "indices": [int(i) for i, _, _ in images[f]]} for f in args.datasets},
        "arm_set": None if args.arms else args.set,
        "arms": {n: {"method": a["method"], "attn": a["attn"], "note": a["note"],
                     "bitwise_ref": a["bitwise"],
                     "ablate": None if a["ablate"] is None else a["ablate"].params(),
                     "layers": None if a.get("layers") is None else list(a["layers"]),
                     "tau": a.get("tau")}
                 for n, a in table.items()},
        "order_schedule": orders,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "cuda": str(torch.version.cuda), "git": git, "allow_dirty": args.allow_dirty,
        "gpu_state_start": gpu_state(), "started": runid,
    }

    t_start = time.perf_counter()
    per_round_gpu = []
    with rows_path.open("w", encoding="utf-8") as fh:
        for r, order in enumerate(orders):
            st = gpu_state()
            per_round_gpu.append({"round": r + 1, **st})
            print(f"round {r + 1}/{args.rounds}  sm={st.get('sm_mhz')}MHz temp={st.get('temp_c')}C "
                  f"({(time.perf_counter() - t_start) / 60:.1f} min elapsed)")
            for pos, name in enumerate(order):
                model.set_attn_implementation(ATTN_MODES[table[name]["attn"]])
                method = methods[name]
                flag0 = args.datasets[0]
                if B is not None:
                    run_batch(model, processor, method, specs[flag0], [im for _, _, im in images[flag0][:B]])
                    for flag in args.datasets:
                        for k in range(0, args.limit, B):
                            chunk = images[flag][k:k + B]
                            gen = run_batch(model, processor, method, specs[flag], [im for _, _, im in chunk])
                            state = getattr(method, "last_state", None)
                            if state is not None:
                                state.finalise_stats()
                            fh.write(json.dumps({
                                "round": r + 1, "pos": pos, "arm": name, "dataset": flag,
                                "index": int(chunk[0][0]), "indices": [int(i) for i, _, _ in chunk],
                                **batch_row(gen), "logits_sha1": digest(gen),
                                "var_stats": None if state is None else stats_summary(state.stats),
                            }) + "\n")
                    fh.flush()
                    continue
                run_one(model, processor, method, specs[flag0], images[flag0][0][2], MAX_NEW_TOKENS)
                for flag in args.datasets:
                    spec = specs[flag]
                    for idx, label, image in images[flag]:
                        gen = run_one(model, processor, method, spec, image, MAX_NEW_TOKENS)
                        state = getattr(method, "last_state", None)
                        if state is not None:
                            state.finalise_stats()
                        n_new = len(gen.new_ids)
                        fh.write(json.dumps({
                            "round": r + 1, "pos": pos, "arm": name, "dataset": flag, "index": int(idx),
                            "n_new": n_new, "vision_ms": gen.vision_ms, "prefill_ms": gen.prefill_ms,
                            "lm_prefill_ms": gen.lm_prefill_ms,
                            "decode_ms_per_token": gen.decode_ms_per_token,
                            # Raw decode time as well as lm_ms: lm_ms extrapolates the 2 measured decode
                            # steps (new_tokens is usually 3) to D = 7, which multiplies their noise by
                            # 3.5. The extrapolation is deliberate -- it stops a differing stop length
                            # moving the metric -- so lm_ms stays primary and this is the companion.
                            "decode_total_ms": (None if gen.decode_ms_per_token is None
                                                else gen.decode_ms_per_token * (n_new - 1)),
                            "lm_ms": gen.lm_ms, "e2e_ms": gen.e2e_ms,
                            "peak_vram_gib": gen.peak_vram_gib, "logits_sha1": digest(gen),
                            "var_stats": None if state is None else stats_summary(state.stats),
                        }) + "\n")
                fh.flush()

    manifest["gpu_state_by_round"] = per_round_gpu
    manifest["gpu_state_end"] = gpu_state()
    manifest["wall_minutes"] = round((time.perf_counter() - t_start) / 60, 2)
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"\nwrote {rows_path}  ({manifest['wall_minutes']:.1f} min)")
    shown = out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else str(out_dir)
    print(f"aggregate: python scripts/ab_report.py {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
