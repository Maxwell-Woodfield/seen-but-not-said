"""Aggregate an `ab_rounds.py` run into per-component ceilings. TIMING ONLY -- no accuracy here.

Statistics, and why these and not others:

- The unit of analysis is the (round, arm) MEAN, never the image. A within-run paired bootstrap cannot
  see run-level drift, and drift is the dominant error term here (92 ms between two runs of identical
  code, notes/status.md). So a contrast X-Y is the set of R round-level differences D_r, and its CI is
  the ordinary t interval on those R numbers.
- Means, not medians, because the ladder decomposition adds marginals and only means are additive. A
  10%-trimmed mean is reported alongside; if the two disagree the run had a stall and says so.
- A distribution-free companion: k/R rounds of the same sign. With 10 rounds, all-same-sign is an exact
  two-sided sign test at p = 2/2^10 = 0.002, which needs no normality and is the more defensible
  statement at this few degrees of freedom.
- Mirroring makes consecutive D_r anti-correlated, which INFLATES sd(D). The interval is therefore
  conservative, which is the right direction to err.
- No multiplicity adjustment: this is measurement, not hypothesis testing. `R` vs `R_dup` is the
  empirical calibration instead -- it must contain 0, and nothing smaller than its half-width is
  reported as resolved.

Two latency conventions (notes/validation_plan.md §2), both reportable with --metric:
- `lm_ms`: the PRE-REGISTERED D = 7 convention, `lm_prefill + 7 x decode/token`. H1's thresholds are
  stated in it. With 3 generated tokens it extrapolates 2 measured decode steps by 3.5x, and ~80% of
  R's overhead in it is that extrapolated decode.
- `lm_run_ms`: MEASURED AS RUN, `lm_prefill + decode_total` at the actual length -- the headline.
Per-call-site costs are always computed on `lm_run_ms`: the site counts are per forward actually run
(prefill + 2 decode steps), so dividing an `lm_ms` marginal by them mixes 7 decode steps with 2 (the
bug in the first m1 report, which inflated prep/gate/edit per-site costs ~2-3x).

    python scripts/ab_report.py results/timing/m1_ladder/<runid> [--metric lm_run_ms]
"""
import argparse
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# t_{0.975, df}; df = rounds - 1.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 29: 2.045}

LADDER = ["S", "Ev", "Hk", "Dt", "DtS", "DtG", "R"]
COMPONENT_OF = {"Ev": "eager vs SDPA", "Hk": "hook dispatch", "Dt": "detect", "DtS": "prep",
                "DtG": "gate", "R": "edit"}
# Per-image call sites for R over the forwards actually run (prefill + 2 decode steps), from the
# profiled run's `var_profile.calls / images` (results/profile/R_tau10).
SITES = {"Dt": ("detect", 32), "DtS": ("prep", 72), "DtG": ("gate", 72), "R": ("edit", 40)}
# (cheaper arm, reference arm): saving = reference - cheaper, as % of (reference - S). Name-based, so
# each run's manifest notes say what an arm meant at the time (in m1, `R` was the first port).
PAIRS = [("R", "R_legacy"), ("R_det1215", "R"), ("M24", "R"), ("M24_legacy", "R_legacy"),
         ("M24", "R_legacy"),
         # m1 (20260920_152202) names, before lean became R's implementation:
         ("R_lean", "R"), ("R_lean_det1215", "R_lean"), ("R_lean_det1215", "R_det1215"),
         ("R_lean_det1215", "R")]
PREP_STOP_MS = 15.0  # pre-committed 2026-09-21: prep below this as run on the lean R is not optimised
CONVENTION = {"lm_ms": "pre-registered D = 7 convention (2 measured decode steps extrapolated x3.5)",
              "lm_run_ms": "measured as run (prefill + the decode steps actually generated)"}


def tcrit(df: int) -> float:
    if df in T975:
        return T975[df]
    return 1.96 if df > 30 else T975[min(T975, key=lambda k: abs(k - df))]


def trimmed_mean(xs: list[float], frac: float = 0.1) -> float:
    xs = sorted(xs)
    k = int(len(xs) * frac)
    core = xs[k:len(xs) - k] or xs
    return statistics.fmean(core)


def round_means(rows: list[dict], key: str) -> dict[str, dict[int, float]]:
    """{arm: {round: mean of `key` over that round's images}}."""
    acc: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        if r.get(key) is None:
            continue
        acc.setdefault(r["arm"], {}).setdefault(r["round"], []).append(r[key])
    return {a: {rd: statistics.fmean(v) for rd, v in d.items()} for a, d in acc.items()}


def contrast(means: dict[str, dict[int, float]], x: str, y: str) -> dict | None:
    """Round-level difference x - y with a t interval and a sign count."""
    if x not in means or y not in means:
        return None
    rounds = sorted(set(means[x]) & set(means[y]))
    d = [means[x][r] - means[y][r] for r in rounds]
    if len(d) < 2:
        return None
    m, sd = statistics.fmean(d), statistics.stdev(d)
    half = tcrit(len(d) - 1) * sd / len(d) ** 0.5
    pos = sum(1 for v in d if v > 0)
    return {"mean": m, "sd": sd, "lo": m - half, "hi": m + half, "half": half, "n": len(d),
            "same_sign": max(pos, len(d) - pos), "trimmed": trimmed_mean(d), "per_round": d}


def fmt(c: dict | None, unit: str = "ms") -> str:
    if c is None:
        return "n/a"
    star = "" if (c["lo"] <= 0 <= c["hi"]) else "*"
    return f"{c['mean']:+.1f} [{c['lo']:+.1f},{c['hi']:+.1f}]{star} {c['same_sign']}/{c['n']}"


def adds(a: str, prev: str) -> str:
    """What rung `a` adds over the previous PRESENT rung: one component, or several when rungs between
    them are absent (in `lean_ladder`, Dt - S is eager + hook dispatch + detect together)."""
    parts = LADDER[LADDER.index(prev) + 1:LADDER.index(a) + 1]
    return " + ".join(COMPONENT_OF[x] for x in parts)


def add_run_metric(rows: list[dict]) -> None:
    """`lm_run_ms` = LM time as run: prefill minus vision, plus the decode steps actually generated."""
    for r in rows:
        ok = r.get("lm_prefill_ms") is not None and r.get("decode_total_ms") is not None
        r["lm_run_ms"] = r["lm_prefill_ms"] + r["decode_total_ms"] if ok else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--metric", default="lm_ms", choices=["lm_ms", "lm_run_ms", "e2e_ms", "lm_prefill_ms",
                                                          "decode_total_ms", "prefill_ms"])
    args = ap.parse_args()

    run = ROOT / args.run_dir
    rows = [json.loads(l) for l in (run / "rows.jsonl").open(encoding="utf-8")]
    add_run_metric(rows)
    manifest = yaml.safe_load((run / "manifest.yaml").read_text(encoding="utf-8"))
    present = [a for a in manifest["arms"] if any(r["arm"] == a for r in rows)]
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit(f"# Timing A/B -- {run.name}")
    emit()
    emit("**TIMING ONLY. ACCURACY IS NOT EVALUATED IN THIS TREE.** Arms that disable parts of VAR")
    emit("produce meaningless predictions by design; only their latency is interpretable.")
    emit()
    emit(f"metric `{args.metric}`" + (f" -- **{CONVENTION[args.metric]}**" if args.metric in CONVENTION else "")
         + f", {manifest['rounds']} rounds x {manifest['limit']} img x "
         f"{len(manifest['datasets'])} datasets, subset `{manifest['subset']}`, "
         + (f"**batch {manifest['batch']}, timings per image (batch / B)**, " if manifest.get("batch") else "")
         + f"git `{manifest['git']['commit'][:7]}`{' (dirty)' if manifest['git']['dirty'] else ''}, "
         f"{manifest.get('wall_minutes', '?')} min")
    emit()

    # ---- validity gates, printed before any result ----------------------------------------------
    emit("## Validity")
    emit()
    vis = round_means(rows, "vision_ms")
    vmeans = {a: statistics.fmean(list(v.values())) for a, v in vis.items()}
    spread = (max(vmeans.values()) - min(vmeans.values())) / statistics.fmean(list(vmeans.values()))
    emit(f"- `vision_ms` spread across arms **{spread * 100:.1f}%** "
         f"({'OK' if spread <= 0.05 else 'FAIL -- >5% invalidates the LM/vision split'}), "
         f"range {min(vmeans.values()):.0f}-{max(vmeans.values()):.0f} ms")

    ntok: dict[str, set] = {}
    for r in rows:
        ntok.setdefault(r["arm"], set()).add(r["n_new"])
    emit(f"- `n_new` per arm: " + ", ".join(f"{a}={sorted(ntok[a])}" for a in present))

    sha = {}
    for r in rows:
        sha.setdefault(r["arm"], {})[(r["dataset"], r["index"])] = r["logits_sha1"]
    groups: dict[str, list[str]] = {}
    for a in present:
        ref = manifest["arms"][a]["bitwise_ref"]
        if ref:
            groups.setdefault(ref, []).append(a)
    for ref, members in groups.items():
        anchor = ref if ref in sha else members[0]   # lean_ladder has Dt/DtS/DtG but not their anchor Ev
        bad = [a for a in members if sha.get(a) != sha.get(anchor)]
        emit(f"- bitwise group `{ref}`" + ("" if anchor == ref else f" (anchor absent; checked against `{anchor}`)")
             + f": {' '.join(members)} -> "
             f"{'ALL IDENTICAL' if not bad else 'SPLIT: ' + ' '.join(bad) + ' (timing is void)'}")

    means = round_means(rows, args.metric)
    null = contrast(means, "R_dup", "R")
    floor = null["half"] if null else float("nan")
    if null:
        emit(f"- **noise floor** `R_dup - R` = {fmt(null)} -> "
             f"{'OK, contains 0' if null['lo'] <= 0 <= null['hi'] else 'FAIL, excludes 0'}. "
             f"Any effect below **{floor:.1f} ms** is unresolved.")
    emit()

    # ---- denominators ----------------------------------------------------------------------------
    denom = {ref: contrast(means, ref, "S") for ref in present if ref != "S"}
    dR = denom.get("R")
    if dR:
        emit(f"**dR_LM = `R - S` = {fmt(dR)}**" + (f"; `R_legacy - S` = {fmt(denom['R_legacy'])}"
                                                    if denom.get("R_legacy") else "")
             + " (published, first port, fixed-order A/B: +245.7 ms on `lm_ms`).")
        emit()

    # ---- the ladder ------------------------------------------------------------------------------
    rungs = [a for a in LADDER if a in means]
    if len(rungs) > 2:
        emit("## Ladder (nested; marginals telescope to `R - Ev` BY CONSTRUCTION, not as a check)")
        emit()
        emit("| arm | what it adds | mean ms | marginal vs previous [95% CI] k/n |")
        emit("|---|---|---|---|")
        for i, a in enumerate(rungs):
            mv = statistics.fmean(list(means[a].values()))
            marg = fmt(contrast(means, a, rungs[i - 1])) if i else "-- (anchor)"
            emit(f"| `{a}` | {adds(a, rungs[i - 1]) if i else 'base SDPA'} | {mv:.1f} | {marg} |")
        emit()

    # ---- levers ----------------------------------------------------------------------------------
    pairs = [(x, y) for x, y in PAIRS if x in means and y in means]
    if pairs:
        emit("## Savings (reference - cheaper; positive = cheaper than the reference)")
        emit()
        emit("| cheaper | reference | saving ms [95% CI] k/n | % of (reference - S) | resolved |")
        emit("|---|---|---|---|---|")
        for x, y in pairs:
            c = contrast(means, y, x)
            d = denom.get(y)
            pct = f"{100 * c['mean'] / d['mean']:.1f}%" if d and d["mean"] else "n/a"
            res = "yes" if abs(c["mean"]) > floor and not (c["lo"] <= 0 <= c["hi"]) else "**no**"
            emit(f"| `{x}` | `{y}` | {fmt(c)} | {pct} | {res} |")
        emit()

    # ---- workload: how much editing each arm did, per image --------------------------------------
    work: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("var_stats"):
            work.setdefault(r["arm"], []).append(r["var_stats"])
    if work:
        def wmean(arm: str, key: str) -> str:
            xs = [v[key] for v in work[arm] if v.get(key) is not None]
            return f"{statistics.fmean(xs):.2f}" if xs else "n/a"
        run_all = round_means(rows, "lm_run_ms")
        emit("## Workload per image (a timing is only comparable if the work done is)")
        emit()
        emit("| arm | rows_edited | edit_events | n_vis | n_text | lm_run_ms |")
        emit("|---|---|---|---|---|---|")
        for a in present:
            if a in work:
                t_run = statistics.fmean(list(run_all[a].values())) if a in run_all else float("nan")
                emit(f"| `{a}` | {wmean(a, 'rows_edited')} | {wmean(a, 'edit_events')} | {wmean(a, 'n_vis')} | "
                     f"{wmean(a, 'n_text')} | {t_run:.1f} |")
        emit()

    # ---- sensitivity to the cold first round -----------------------------------------------------
    warm = round_means([r for r in rows if r["round"] > 1], args.metric)
    emit("### Sensitivity: round 1 dropped (the GPU starts at idle clocks)")
    emit()
    emit("| contrast | all rounds | rounds 2+ |")
    emit("|---|---|---|")
    for x, y in [("R", "S"), ("R_dup", "R")] + [(y, x) for x, y in pairs]:
        c_all, c_warm = contrast(means, x, y), contrast(warm, x, y)
        if c_all:
            emit(f"| `{x} - {y}` | {fmt(c_all)} | {fmt(c_warm)} |")
    emit()

    # ---- per-call-site, always on the as-run metric ----------------------------------------------
    run_means = round_means(rows, "lm_run_ms")
    run_null = contrast(run_means, "R_dup", "R")
    run_floor = run_null["half"] if run_null else float("nan")
    # A marginal is a component's cost only between ADJACENT ladder rungs; in an arm set without the
    # ladder (e.g. `m24`), `R - S` is the whole overhead, not "edit".
    site_rows = [(a, rungs[i - 1]) for i, a in enumerate(rungs)
                 if i and a in SITES and rungs[i - 1] == LADDER[LADDER.index(a) - 1]]
    if site_rows or "R_det1215" in run_means:
        emit("## Per-call-site (always on `lm_run_ms`, measured as run)")
        emit()
        emit("Sites are per image over the forwards actually run (prefill + 2 decode steps), from "
             "`results/profile/R_tau10`: detect 32, prep 72, gate 72, edit 40. They are divided into the "
             f"as-run marginal, never the D = 7 one. `lm_run_ms` noise floor: **{run_floor:.1f} ms**.")
        emit()
        emit("| component | as-run marginal ms [95% CI] | sites/img | ms/site |")
        emit("|---|---|---|---|")
        for a, prev in site_rows:
            c = contrast(run_means, a, prev)
            label, n = SITES[a]
            resolved = abs(c["mean"]) > run_floor
            per = f"{c['mean'] / n:.3f}" if resolved else f"(<{run_floor / n:.3f}, unresolved)"
            emit(f"| {label} | {fmt(c)} | {n} | {per} |")
        prep = contrast(run_means, "DtS", "Dt")
        if prep and manifest.get("arm_set") == "lean_ladder":
            verdict = ("below: do NOT optimise prep; report the composition and go to the test runs"
                       if prep["mean"] < PREP_STOP_MS else "at or above: one time-boxed, bitwise-safe session")
            emit(f"**Prep stop rule (pre-committed 2026-09-21, point estimate as run vs {PREP_STOP_MS:.0f} ms):** "
                 f"prep = {fmt(prep)} -> {verdict}.")
            emit()
        lev = contrast(run_means, "R", "R_det1215")
        if lev:
            per = f"{lev['mean'] / 20:.3f}" if abs(lev["mean"]) > run_floor else "unresolved"
            emit(f"| detector lever (20 of 32 sites) | {fmt(lev)} | 20 | {per} |")
        emit()

    # ---- drift -----------------------------------------------------------------------------------
    gs = manifest.get("gpu_state_by_round") or []
    if gs and "sm_mhz" in gs[0]:
        emit("## GPU state by round")
        emit()
        emit("| round | " + " | ".join(str(g["round"]) for g in gs) + " |")
        emit("|---|" + "---|" * len(gs))
        emit("| SM MHz | " + " | ".join(f"{g['sm_mhz']:.0f}" for g in gs) + " |")
        emit("| temp C | " + " | ".join(f"{g['temp_c']:.0f}" for g in gs) + " |")
        emit()

    (run / f"report_{args.metric}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    summary = {"metric": args.metric, "convention": CONVENTION.get(args.metric), "noise_floor_ms": floor,
               "arm_means": {a: statistics.fmean(list(v.values())) for a, v in means.items()},
               "contrasts": {f"{x}-{y}": contrast(means, x, y)
                             for x in present for y in present if x != y}}
    (run / f"summary_{args.metric}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {run / f'report_{args.metric}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
