"""Phase 7 card V1: GPU runner for the noised runs and the clean per-id re-run (notes/vcd_plan.md, Amendment V1-1).

Presets (registered sequences, one model load per process):
    python scripts/run_vcd.py --preset smoke          # G3: base_identity, R_identity, base_s500, base_s999 (first 10)
    python scripts/run_vcd.py --preset smoke-rerun    # G3, a SECOND process: base_s500_rerun, base_s999_rerun
    python scripts/run_vcd.py --preset g4 --clean-rerun full|fallback [--max-new-tokens 8|1]
        # G4: base_s500, R_s500, base_s999, base_s500_rep2, clean_R, clean_base. --clean-rerun is chosen only by the
        # dated, timing-only V1- note after G3; --max-new-tokens 1 only by a dated V1- amendment before G4.
Single run (anything outside a preset; never a silent replacement):
    python scripts/run_vcd.py --arm base|var (--step 500|999 [--rep 1|2] [--no-cache] [--identity] | --clean)
        --exp-name tune/vcd/V1/<name> [--limit N] [--indices FLAG:I,J ...] [--datasets ...]
        [--replace "<reason>"]
Add --dry-run to any of them (the processor and saved configs only: no weights, no GPU).

What every run does:
- Refuses a dirty tree (no tracked edits, nothing changed or untracked under effvar/ scripts/ tests/), and code
  trees other than those of the code commit recorded in the plan index's G2 row (guard 6), before the model loads
  and again before each sub-run.
- Asserts, before the model loads: each letter's ids = the registered ones (guard 5); the frozen prompt, attn
  var_eager, option_token_sets and method params = the saved clean run's config.yaml for that arm.
- Noised mode: x' = c1*x + c2*eps on the fp32 pixel_values (effvar.vcd.build_noised_inputs), then the harness's
  generate call; eps from vcd_seed(flag, split, index, rep). With the noised cache, features are cached apart from
  the clean ones (effvar/vcd_cache.py). Uncached (the smoke only), a probe records max |delta| between the fresh
  projector output and the clean cached features (guard 8), and no noised cache is read or written (guard 3).
- Clean mode: the harness's own run_untimed on the clean cache, which refuses any noised directory (guard 4).
- Rows: the saved clean runs' fields plus logits_sha1, p_ids, logp_ids, logp_option (fp32), step, seed, rep, c1, c2,
  identity (and feat_maxabs_vs_clean_cache in the smoke). metrics.json holds label-free code-check quantities only.
- Each sub-run writes its own predictions.jsonl / config.yaml / metrics.json per dataset, with its own cache-counter
  deltas, the clean-cache state and the noised-cache directories' state at start and end, and git at start and end.
- Process record results/tune/vcd/V1/processes/<UTC>_<pid>_<label>.yaml: argv, git, model-load time, each
  sub-run's rows / wall time / s per image, rewritten after the load, every 50 rows and after each dataset (a crash
  still leaves the GPU time spent on record). Before each sub-run (and before the load), the GPU
  time recorded so far plus the projection (rows x the measured s/image of its rate class; else the largest measured
  rate; else 1.1 s) must stay within 1.0 GPU-h, or the process stops cleanly before that sub-run (exit 3).
- An existing run directory is never overwritten. A single run may replace one only with --replace "<reason>": the
  old directory moves to results/tune/vcd/V1/_replaced/<UTC>/..., and the reason is recorded. Presets never replace.
"""
import argparse
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from effvar import FLAGS, MODEL_ID, ROOT
from effvar.generation import MAX_NEW_TOKENS, SEED
from effvar.runio import RESULTS_DIR, write_metrics, write_predictions
from effvar.vcd import (ATTN, FALLBACK_S_PER_IMAGE, GPU_CAP_S, PRESETS, PROCESS_DIR, REPLACED_DIR, SAVED_RUNS,
                        SCHEDULE, STEPS, V1_EXP, SubRun, expected_code_trees, git_code_state, is_clean, preset,
                        selection, subrun_record, validate_subrun)

LOAD_S_FALLBACK = 60.0   # model load, used only before a load has been measured
PROGRESS_EVERY = 50      # rows between process-record rewrites
TIMING_KEYS = ("vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "e2e_ms")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# --------------------------------------------------------------------------- argument handling
def parse_indices(tokens) -> tuple:
    """["pneumoniamnist:291,4328", ...] -> (("pneumoniamnist", (291, 4328)), ...)."""
    out = {}
    for tok in tokens or ():
        flag, _, rest = tok.partition(":")
        if flag not in FLAGS or not rest:
            raise ValueError(f"--indices expects FLAG:I,J,...; got {tok!r}")
        out.setdefault(flag, set()).update(int(x) for x in rest.split(",") if x)
    return tuple((f, tuple(sorted(v))) for f, v in out.items())


def build_subruns(args) -> tuple[list[SubRun], str]:
    """(sub-runs, process label) from the CLI. Raises ValueError on any rule violation."""
    single = ("arm", "step", "rep", "limit", "indices", "exp_name", "replace")
    if args.preset:
        given = [a for a in single if getattr(args, a) not in (None, [], ())]
        given += [a for a in ("clean", "no_cache", "identity") if getattr(args, a)]
        if list(args.datasets) != list(FLAGS):
            given.append("datasets")
        if given:
            raise ValueError(f"--preset takes no single-run arguments (got {given}); presets run their registered "
                             "rows and never replace")
        if args.preset != "g4" and (args.clean_rerun or args.max_new_tokens != MAX_NEW_TOKENS):
            raise ValueError("--clean-rerun / --max-new-tokens apply to --preset g4 only (the smoke uses 8)")
        return preset(args.preset, clean_rerun=args.clean_rerun, max_new_tokens=args.max_new_tokens), args.preset
    if not args.arm or not args.exp_name:
        raise ValueError("a single run needs --arm and --exp-name (or use --preset)")
    name = args.exp_name.rstrip("/").split("/")[-1]
    indices = parse_indices(args.indices)
    if args.clean:
        if args.step is not None or args.rep is not None or args.no_cache or args.identity:
            raise ValueError("--clean takes no --step, --rep, --no-cache or --identity")
        if args.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError("clean re-runs keep MAX_NEW_TOKENS 8 (guard 7)")
        sub = SubRun(name=name, exp=args.exp_name, arm=args.arm, mode="clean", cache="clean", limit=args.limit,
                     indices=indices, clean_rerun=args.clean_rerun)
    else:
        if args.step is None:
            raise ValueError("a noised run needs --step 500|999 (or --clean)")
        if args.clean_rerun:
            raise ValueError("--clean-rerun applies to clean runs only")
        sub = SubRun(name=name, exp=args.exp_name, arm=args.arm, mode="noised",
                     cache="none" if args.no_cache else "noised", step=args.step, rep=args.rep or 1,
                     identity=args.identity, limit=args.limit, indices=indices, max_new_tokens=args.max_new_tokens)
    validate_subrun(sub)
    return [sub], f"single_{name}"


# --------------------------------------------------------------------------- process record and budget
class Ledger:
    """This process's record, plus the GPU time and per-image rates of every earlier Phase 7 process."""

    def __init__(self, results_dir: Path, label: str, argv, clock):
        self.dir = Path(results_dir) / PROCESS_DIR
        self.path = self.dir / f"{utc_stamp()}_{os.getpid()}_{label}.yaml"
        self.clock = clock
        self.t0 = clock()
        self.prior = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(self.dir.glob("*.yaml"))] \
            if self.dir.exists() else []
        self.rec = {"label": label, "argv": list(argv), "started_utc": utc_stamp(), "status": "starting",
                    "load_s": None, "elapsed_s": 0.0, "subruns": [], "cap_s": GPU_CAP_S}

    def prior_s(self) -> float:
        return float(sum(r.get("elapsed_s") or 0.0 for r in self.prior))

    def used_s(self) -> float:
        return self.prior_s() + (self.clock() - self.t0)

    def rate(self, key: str) -> tuple[float, str]:
        """The largest measured s/image of the rate class; a miss falls back to the uncached rate (both compute the
        vision tower); otherwise the largest measured rate; otherwise 1.1 s."""
        done = [s for r in self.prior + [self.rec] for s in r.get("subruns", [])
                if s.get("status") == "done" and s.get("rows")]
        for k in (key, "uncached") if key == "miss" else (key,):
            same = [s["s_per_image"] for s in done if s["rate_key"] == k]
            if same:
                return max(same), f"measured ({k})"
        if done:
            return max(s["s_per_image"] for s in done), "largest measured rate"
        return FALLBACK_S_PER_IMAGE, "fallback 1.1 s/image (card, Phase 4)"

    def load_estimate(self) -> float:
        loads = [r["load_s"] for r in self.prior if r.get("load_s")]
        return max(loads) if loads else LOAD_S_FALLBACK

    def write(self, **fields) -> None:
        self.rec.update(fields)
        self.rec["elapsed_s"] = self.clock() - self.t0
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(self.rec, sort_keys=False, allow_unicode=True), encoding="utf-8")


def budget_line(used: float, projected: float, rate: float, why: str, rows: int, what: str) -> dict:
    return {"before": what, "used_s": used, "rows": rows, "s_per_image": rate, "rate_source": why,
            "projected_s": projected, "total_s": used + projected, "cap_s": GPU_CAP_S}


# --------------------------------------------------------------------------- one sub-run
def code_check_metrics(rows: list[dict], wall_s: float, cache_delta: dict | None, rate_key: str) -> dict:
    """Label-free quantities only (no metric is computed before G4 ends: guard 6)."""
    n_new = [len(r["new_ids"]) for r in rows]
    vals, counts = np.unique(n_new, return_counts=True)
    out = {"n": len(rows), "compliance_rate": float(np.mean([r["first_token_in_options"] for r in rows])),
           "leakage_mean": float(np.mean([r["leakage"] for r in rows])),
           "leakage_max": float(np.max([r["leakage"] for r in rows])),
           "new_tokens_hist": {str(int(v)): int(c) for v, c in zip(vals, counts)},
           "wall_s": wall_s, "sec_per_image": wall_s / len(rows), "rate_key": rate_key,
           "peak_vram_gib_max": float(max(r["peak_vram_gib"] for r in rows)),
           "vision_cache_stats": cache_delta}
    probe = [r["feat_maxabs_vs_clean_cache"] for r in rows if "feat_maxabs_vs_clean_cache" in r]
    if probe:
        out["probe_maxabs_min"], out["probe_maxabs_max"] = float(min(probe)), float(max(probe))
    return out


def _counts(cache) -> dict | None:
    st = cache.stats() if cache is not None else None
    return None if st is None else {"hits": st["hits"], "misses": st["misses"], "key_digest": st["key_digest"]}


def check_code(state: dict, trees: dict, what: str) -> None:
    """Guard 6 at run time: a clean tree whose code equals the code commit recorded in the plan index."""
    if not is_clean(state):
        raise SystemExit(f"refusing {what}: the tree is not clean (guard 6). tracked edits: {state['dirty']}; "
                         f"code changes: {state['code_changes']}")
    if state["code_trees"] != trees:
        raise SystemExit(f"refusing {what}: effvar/ scripts/ tests/ differ from the code commit in the plan index "
                         "(guard 6)")


def execute_subrun(rt, sub: SubRun, sel: dict, results_dir: Path, clock, git_state, process_path: Path,
                   argv, extra: dict, trees: dict, progress=lambda: None) -> dict:
    git_start = git_state()
    check_code(git_start, trees, sub.name)
    summary = {"name": sub.name, "exp": sub.exp, "rate_key": sub.rate_key, "rows": 0, "wall_s": 0.0,
               "status": "running", "datasets": {}}
    with rt.open_cache(sub) as cache:
        for flag, indices in sel.items():
            out = Path(results_dir) / sub.exp / flag
            if out.exists():
                raise SystemExit(f"refusing to overwrite {out}")
            state0, noised0 = rt.clean_state(), rt.noised_state()
            before = _counts(cache)
            t0 = clock()
            rows = []
            for idx, label, image in rt.samples(flag, indices):
                rows.append(rt.row(sub, flag, idx, label, image, cache))
                if len(rows) % PROGRESS_EVERY == 0:
                    progress()
            wall = clock() - t0
            after = _counts(cache)
            delta = None if after is None else {"hits": after["hits"] - before["hits"],
                                                "misses": after["misses"] - before["misses"],
                                                "key_digest": after["key_digest"]}
            if [r["index"] for r in rows] != list(indices):
                raise RuntimeError(f"{sub.name} {flag}: rows do not follow the selection")
            out.mkdir(parents=True)
            write_predictions(out, rows)
            write_metrics(out, code_check_metrics(rows, wall, delta, sub.rate_key))
            cfg = {**rt.config(sub, flag, indices), "subrun": subrun_record(sub), "vision_cache_stats": delta,
                   "clean_cache_state": {"start": state0, "end": rt.clean_state()},
                   "noised_cache_state": {"start": noised0, "end": rt.noised_state()},
                   "process_record": Path(process_path).name, "argv": list(argv), **extra,
                   "git": git_start, "git_end": git_state()}
            (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8")
            summary["rows"] += len(rows)
            summary["wall_s"] += wall
            summary["datasets"][flag] = {"rows": len(rows), "wall_s": wall, "vision_cache_stats": delta}
            progress()
            print(f"{sub.name} {flag}: n={len(rows)} {wall / len(rows):.3f} s/image cache={delta}")
    summary["s_per_image"] = summary["wall_s"] / summary["rows"]
    summary["status"] = "done"
    return summary


def archive(paths: list[Path], results_dir: Path) -> list[dict]:
    stamp = utc_stamp()
    moved = []
    for p in paths:
        dest = Path(results_dir) / REPLACED_DIR / stamp / p.relative_to(results_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append({"from": p.relative_to(results_dir).as_posix(), "to": dest.relative_to(results_dir).as_posix()})
    return moved


# --------------------------------------------------------------------------- the sequence
def run_sequence(subs: list[SubRun], *, label: str, argv, make_runtime, subset_idx: dict, flags=FLAGS,
                 results_dir: Path = RESULTS_DIR, clock=time.perf_counter, git_state=git_code_state,
                 replace: str | None = None, extra: dict | None = None, code_trees: dict | None = None) -> int:
    """Run sub-runs in order under one model load. Returns 0 when done, 3 on a clean budget stop.
    `code_trees` defaults to the trees of the code commit recorded in the plan index."""
    results_dir = Path(results_dir)
    for s in subs:
        validate_subrun(s)
    trees = expected_code_trees() if code_trees is None else code_trees
    git0 = git_state()
    check_code(git0, trees, "to run")
    sels = {s.name: selection(s, subset_idx, flags) for s in subs}
    empty = [s.name for s in subs if not sum(len(v) for v in sels[s.name].values())]
    if empty:
        raise SystemExit(f"refusing to run: no rows selected for {empty}")
    existing = [results_dir / s.exp / f for s in subs for f in sels[s.name] if (results_dir / s.exp / f).exists()]
    if existing and (replace is None or len(subs) != 1):
        raise SystemExit("refusing to overwrite existing run directories "
                         f"{[p.relative_to(results_dir).as_posix() for p in existing]}"
                         + ("" if len(subs) == 1 else " (presets never replace)")
                         + ("; a single run may replace with --replace \"<reason>\"" if len(subs) == 1 else ""))
    extra = dict(extra or {})
    ledger = Ledger(results_dir, label, argv, clock)
    rows0 = sum(len(v) for v in sels[subs[0].name].values())
    rate0, why0 = ledger.rate(subs[0].rate_key)
    pre = budget_line(ledger.used_s() + ledger.load_estimate(), rows0 * rate0, rate0, why0, rows0,
                      f"load + {subs[0].name}")
    if pre["total_s"] > GPU_CAP_S:
        ledger.write(status="stopped_budget", git=git0, budget=[pre],
                     stop=f"projected {pre['total_s']:.0f} s > cap {GPU_CAP_S:.0f} s before the model load")
        print(f"BUDGET STOP before the model load: {pre}")
        return 3
    ledger.write(status="loading", git=git0, budget=[pre], **extra)
    t0 = clock()
    rt = make_runtime()
    ledger.write(status="loaded", load_s=clock() - t0)
    for sub in subs:
        rows = sum(len(v) for v in sels[sub.name].values())
        rate, why = ledger.rate(sub.rate_key)
        line = budget_line(ledger.used_s(), rows * rate, rate, why, rows, sub.name)
        ledger.rec["budget"].append(line)
        if line["total_s"] > GPU_CAP_S:
            ledger.write(status="stopped_budget",
                         stop=f"projected {line['total_s']:.0f} s > cap {GPU_CAP_S:.0f} s before {sub.name}")
            print(f"BUDGET STOP before {sub.name}: {line}")
            return 3
        sub_extra = dict(extra)
        if replace is not None and existing:
            sub_extra["replaces"] = {"reason": replace, "archived": archive(existing, results_dir)}
            ledger.rec["replaces"] = sub_extra["replaces"]
        ledger.rec["subruns"].append({"name": sub.name, "rate_key": sub.rate_key, "rows": rows, "status": "running"})
        ledger.write(status=f"running {sub.name}")
        summary = execute_subrun(rt, sub, sels[sub.name], results_dir, clock, git_state, ledger.path, argv,
                                 sub_extra, trees, progress=ledger.write)
        ledger.rec["subruns"][-1] = summary
        ledger.write(status=f"done {sub.name}")
    ledger.write(status="done", git_end=git_state())
    return 0


# --------------------------------------------------------------------------- the real runtime (GPU)
def saved_config(arm: str, flag: str) -> dict:
    return yaml.safe_load((RESULTS_DIR / SAVED_RUNS[arm] / flag / "config.yaml").read_text(encoding="utf-8"))


def preflight(subs: list[SubRun], processor, subset_idx: dict, flags=FLAGS) -> dict:
    """No model: guard 5 and the saved-config identity of the frozen inputs. Returns token sets and the clean digest."""
    from effvar.methods import make_method
    from effvar.prompts import get_spec
    from effvar.scoring import option_token_sets
    from effvar.vcd import check_registered_ids

    token_sets, digests = {}, set()
    used = {(s.arm, f) for s in subs for f in selection(s, subset_idx, flags)}
    for flag in FLAGS:
        spec = get_spec(flag)
        token_sets[flag] = option_token_sets(processor.tokenizer, spec)
        check_registered_ids(token_sets[flag], spec)
    for arm, flag in sorted(used):
        spec, saved, method = get_spec(flag), saved_config(arm, flag), make_method(arm, attn=ATTN)
        checks = {"prompt": saved["prompt"] == spec.text(), "attn": saved["attn"] == ATTN,
                  "option_token_sets": saved["option_token_sets"] == token_sets[flag],
                  "method": saved["method"] == method.name, "method_params": saved["method_params"] == method.params(),
                  "subset": saved["subset"]["name"] == "tune" and saved["subset"]["n"] == len(subset_idx[flag]),
                  "max_new_tokens": saved["decoding"]["max_new_tokens"] == MAX_NEW_TOKENS}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise SystemExit(f"{arm} {flag}: differs from the saved clean run {SAVED_RUNS[arm]} on {bad}")
        digests.add(saved["vision_cache_stats"]["key_digest"])
    for arm in ("base", "var"):
        for flag in FLAGS:
            digests.add(saved_config(arm, flag)["vision_cache_stats"]["key_digest"])
    if len(digests) != 1:
        raise SystemExit(f"the saved clean runs used more than one clean cache: {sorted(digests)}")
    return {"token_sets": token_sets, "clean_digest": digests.pop()}


class Runtime:
    """MedGemma (attn var_eager), both methods, the clean cache key. Built inside the timed load."""

    def __init__(self, processor, pre: dict):
        import torch
        from effvar.data import load_subset
        from effvar.methods import make_method
        from effvar.model import load_model
        from effvar.prompts import get_spec
        from effvar.vcd_cache import CACHE_DIR, check_clean_dir, key_digest
        from effvar.vision_cache import cache_key

        torch.manual_seed(SEED)
        self.processor = processor
        self.model = load_model(ATTN)
        self.methods = {a: make_method(a, attn=ATTN) for a in ("base", "var")}
        self.specs = {f: get_spec(f) for f in FLAGS}
        self.subsets = {f: load_subset(f, "tune") for f in FLAGS}
        self.token_sets = pre["token_sets"]
        self.clean_key = cache_key(self.model, processor)
        self.clean_digest = pre["clean_digest"]
        if key_digest(self.clean_key) != self.clean_digest:
            raise SystemExit(f"clean cache key digest {key_digest(self.clean_key)} != saved runs' {self.clean_digest}")
        self.clean_dir = CACHE_DIR / self.clean_digest
        check_clean_dir(self.clean_dir, self.clean_key)
        self.spans = {}

    @contextmanager
    def open_cache(self, sub: SubRun):
        from effvar.vcd_cache import CleanFeatureProbe, NoisedVisionCache, clean_cache_for_key
        if sub.cache == "noised":
            cache = NoisedVisionCache(self.clean_key, sub.step)
        elif sub.cache == "clean":
            cache = clean_cache_for_key(self.clean_key, self.clean_digest)
        else:
            cache = CleanFeatureProbe(self.clean_dir)
        with cache.attached(self.model):
            yield cache

    def clean_state(self) -> dict:
        from effvar.vcd_cache import dir_state
        return dir_state(self.clean_dir)

    def noised_state(self) -> dict:
        from effvar.vcd_cache import noised_dirs_state
        return noised_dirs_state()

    def samples(self, flag: str, indices: list[int]):
        from effvar.data import iter_samples
        subset = self.subsets[flag]
        label_of = dict(zip(subset.indices, subset.labels))
        chosen = dc_replace(subset, indices=list(indices), labels=[label_of[i] for i in indices])
        yield from iter_samples(chosen)

    def row(self, sub: SubRun, flag: str, idx: int, label: int, image, cache) -> dict:
        from effvar.generation import run_untimed
        from effvar.model import build_inputs, image_span
        from effvar.scoring import first_token_metrics, parse_answer
        from effvar.vcd import noise_coefficients, readout, run_noised, vcd_seed

        spec, method, sets = self.specs[flag], self.methods[sub.arm], self.token_sets[flag]
        split = self.subsets[flag].split
        if flag not in self.spans:
            self.spans[flag] = image_span(build_inputs(processor=self.processor, messages=spec.messages(image),
                                                       device=self.model.device)["input_ids"], self.model.config)
        noise = {"step": None, "seed": None, "rep": None, "c1": None, "c2": None, "identity": False}
        if sub.mode == "clean":
            with cache.sample(flag, split, idx):
                gen = run_untimed(self.model, self.processor, method, spec, image)
        else:
            seed = vcd_seed(flag, split, idx, sub.rep)
            c1, c2 = (1.0, 0.0) if sub.identity else noise_coefficients(sub.step)
            noise = {"step": sub.step, "seed": seed, "rep": sub.rep, "c1": c1, "c2": c2, "identity": sub.identity}
            scope = cache.sample(flag, split, idx, seed) if sub.cache == "noised" else cache.sample(flag, split, idx)
            with scope:
                gen = run_noised(self.model, self.processor, method, spec, image, c1, c2, seed, sub.max_new_tokens)
        ft = first_token_metrics(gen.first_logits, sets, label)
        pred, source = parse_answer(gen.text, spec, ft["p_option"])
        row = {"index": idx, "label": label, "pred": pred, "match_source": source, "text": gen.text,
               "new_ids": gen.new_ids, "input_len": gen.input_len, "logits_sha1": gen.digest(), **ft,
               **gen.timings(), **readout(gen.first_logits, sets), **noise}
        if sub.cache == "none":
            if cache.calls != 1:
                raise RuntimeError(f"probe saw {cache.calls} projector calls for {flag} {idx}; expected 1")
            row["feat_maxabs_vs_clean_cache"] = cache.max_abs
        return row

    def config(self, sub: SubRun, flag: str, indices: list[int]) -> dict:
        import torch
        import transformers
        spec, method, subset = self.specs[flag], self.methods[sub.arm], self.subsets[flag]
        return {"model": MODEL_ID, "dtype": "bfloat16", "card": "notes/vcd_plan.md card V1 + Amendment V1-1",
                "method": method.name, "method_params": method.params(), "attn": ATTN, "dataset": flag,
                "subset": {"name": subset.name, "split": subset.split, "n": len(subset.indices)},
                "selection": {"limit": sub.limit, "indices": {f: list(v) for f, v in sub.indices},
                              "n_selected": len(indices), "selected": list(indices)},
                "mode": sub.mode,
                "noise": None if sub.mode == "clean" else {
                    "schedule": SCHEDULE, "step": sub.step, "rep": sub.rep, "identity": sub.identity,
                    "eps": "N(0, I) fp32, torch.Generator('cpu').manual_seed(seed), added to the processor-"
                           "normalised fp32 pixel_values before the vision tower, unclipped, then cast to bf16",
                    "seed": "int.from_bytes(sha1(f'{flag}|{split}|{index}|{rep}')[:8], 'big') & (2**63 - 1)"},
                "prompt": spec.text(), "labels": list(spec.labels), "option_texts": list(spec.option_texts),
                "option_token_sets": self.token_sets[flag], "registered_ids": True,
                "readout": "p_ids / logp_ids / logp_option follow option_token_sets order; fp32 log_softmax over "
                           "the full vocabulary; logp_option = logsumexp of the letter's ids",
                "decoding": {"greedy": True, "max_new_tokens": sub.max_new_tokens}, "seed": SEED,
                "torch": str(torch.__version__), "transformers": str(transformers.__version__),
                "gpu": torch.cuda.get_device_name(0),
                "image_span": {"start": self.spans[flag][0], "P": self.spans[flag][1]},
                "vision_cache": sub.cache, "clean_rerun": sub.clean_rerun,
                "saved_reference": {"run": SAVED_RUNS[sub.arm], "clean_cache_digest": self.clean_digest}}


# --------------------------------------------------------------------------- dry run
def dry_run(subs, label, subset_idx, flags, processor, results_dir: Path = RESULTS_DIR) -> int:
    git = git_code_state()
    ok = True
    print(f"{label}: git HEAD {git['commit']}  tracked edits: {'YES' if git['dirty'] else 'none'}  "
          f"code changes/untracked under effvar/scripts/tests: {git['code_changes'] or 'none'}")
    if not is_clean(git):
        print("WOULD REFUSE: the tree is not clean (guard 6)")
        ok = False
    try:
        same = git["code_trees"] == expected_code_trees()
        print(f"code trees vs the code commit in the plan index: {'equal' if same else 'DIFFERENT'}")
        if not same:
            print("WOULD REFUSE: effvar/ scripts/ tests/ differ from the code commit (guard 6)")
            ok = False
    except SystemExit as err:
        print(f"WOULD REFUSE: {err}")
        ok = False
    for s in subs:
        if not sum(len(v) for v in selection(s, subset_idx, flags).values()):
            print(f"WOULD REFUSE: no rows selected for {s.name}")
            ok = False
    try:
        pre = preflight(subs, processor, subset_idx, flags)
        print(f"guard 5 ids and saved-config identity: OK; clean cache digest {pre['clean_digest']}")
    except SystemExit as err:
        print(f"WOULD REFUSE: {err}")
        ok = False
    ledger = Ledger(results_dir, label, [], time.perf_counter)
    used = ledger.prior_s() + ledger.load_estimate()
    print(f"GPU time recorded so far {ledger.prior_s():.0f} s; load estimate {ledger.load_estimate():.0f} s")
    for s in subs:
        sel = selection(s, subset_idx, flags)
        rows = sum(len(v) for v in sel.values())
        rate, why = ledger.rate(s.rate_key)
        used += rows * rate
        exists = [f for f in sel if (results_dir / s.exp / f).exists()]
        print(f"  {s.name:16s} {s.mode:6s} arm {s.arm:4s} step {s.step} rep {s.rep} cache {s.cache:6s} "
              f"mnt {s.max_new_tokens} rows {rows:4d} ({', '.join(f'{f}:{len(v)}' for f, v in sel.items())}) "
              f"expect {s.expect} -> cumulative {used / 3600:.3f} GPU-h [{why}]"
              + (f"  EXISTS {exists}" if exists else ""))
        if exists:
            ok = False
    if used > GPU_CAP_S:
        print(f"note: the projection {used / 3600:.3f} GPU-h exceeds the cap; the run would stop before the "
              "first sub-run that crosses it")
    print("DRY RUN OK" if ok else "DRY RUN: the real run would refuse (see above)")
    return 0 if ok else 1


# --------------------------------------------------------------------------- CLI
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS)
    ap.add_argument("--clean-rerun", choices=("full", "fallback"))
    ap.add_argument("--arm", choices=("base", "var"), help="var = R at its defaults")
    ap.add_argument("--step", type=int, choices=STEPS)
    ap.add_argument("--rep", type=int, choices=(1, 2))
    ap.add_argument("--clean", action="store_true", help="the clean per-id re-run on the clean cache")
    ap.add_argument("--limit", type=int, default=None, help="first N tune images per dataset")
    ap.add_argument("--indices", nargs="+", default=None, metavar="FLAG:I,J")
    ap.add_argument("--no-cache", action="store_true", help="the smoke: no vision cache (exp under smoke/)")
    ap.add_argument("--identity", action="store_true", help="guard 1: coefficients forced to (1, 0); uncached")
    ap.add_argument("--max-new-tokens", type=int, choices=(MAX_NEW_TOKENS, 1), default=MAX_NEW_TOKENS)
    ap.add_argument("--exp-name", default=None, help=f"must start with {V1_EXP}/")
    ap.add_argument("--datasets", nargs="+", default=list(FLAGS), choices=FLAGS)
    ap.add_argument("--replace", default=None, metavar="REASON", help="single runs only; archived and recorded")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    try:
        subs, label = build_subruns(args)
    except ValueError as err:
        ap.error(str(err))
    from effvar.data import load_subset
    from effvar.model import load_processor
    subset_idx = {f: list(load_subset(f, "tune").indices) for f in FLAGS}
    flags = [f for f in FLAGS if f in args.datasets]
    processor = load_processor()
    if args.dry_run:
        return dry_run(subs, label, subset_idx, flags, processor)
    pre = preflight(subs, processor, subset_idx, flags)
    extra = {"preset": args.preset, "preset_clean_rerun": args.clean_rerun,
             "preset_max_new_tokens": args.max_new_tokens if args.preset == "g4" else None}
    return run_sequence(subs, label=label, argv=sys.argv[1:] if argv is None else argv,
                        make_runtime=lambda: Runtime(processor, pre), subset_idx=subset_idx, flags=flags,
                        replace=args.replace, extra=extra)


if __name__ == "__main__":
    raise SystemExit(main())
