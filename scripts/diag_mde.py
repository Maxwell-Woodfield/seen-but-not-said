"""Amendment 2(b): minimum detectable effects for the OvO log-ratio AUC. CPU only, saved outputs only.

    python scripts/diag_mde.py

For each dataset: the paired-bootstrap SD of dOvO-LR (variant - base) on tune (n = 300) and on val, for R
and for the 24 gate-sweep configs (tune), plus the SD of a single arm's OvO-LR (the scale of one probe
AUC in B2). MDE50 = 1.96 SD (the effect whose CI just excludes 0 at the point estimate, i.e. the CI
half-width); MDE80 = 2.80 SD (80% power, two-sided 5%). Sizes with no outputs yet are scaled by
sqrt(n_ref / n) from tune: the Retina confirmation set (val 120 + confirm 300 = 420) and D4 (Derma 1000
train). 10,000 resamples, seed 0. Writes results/diag/A1/mde_ovo.md.
"""
import numpy as np

from effvar import FLAGS, ROOT
from effvar.diag_stats import DATASETS, align, boot_idx, load_run, ovo_logratio_auc

SWEEP = [f"{fam}/rho{r}_summ{s}" for fam in ("var", "var_attn") for r in (0.1, 0.2, 0.5, 0.7) for s in (0.05, 0.1, 0.2)]


def sds(base_dir, variants, flag, B=10_000):
    k, _ = DATASETS[flag]
    runs = {"base": load_run(ROOT / base_dir, flag), **{v: load_run(ROOT / d, flag) for v, d in variants.items()}}
    _, y, arr = align(runs)
    idx = boot_idx(len(y), B=B)
    b = np.array([ovo_logratio_auc(y[ix], arr["base"]["p"][ix], k) for ix in idx])
    out = {"n": len(y), "single_arm_sd": float(np.nanstd(b))}
    for v in variants:
        d = np.array([ovo_logratio_auc(y[ix], arr[v]["p"][ix], k) for ix in idx]) - b
        out[v] = float(np.nanstd(d))
    return out


def main() -> None:
    lines = ["# OvO log-ratio AUC: minimum detectable effects (Amendment 2b), 2026-09-28", "",
             "SD = paired-bootstrap SD of dOvO-LR vs base (10,000, seed 0). MDE50 = 1.96 SD = CI half-width; "
             "MDE80 = 2.80 SD. Scaled rows use SD x sqrt(300 / n) from tune.", "",
             "| dataset | size | source | SD (R − base) | MDE50 | MDE80 | sweep SD median [max] | single-arm SD |",
             "|---|---|---|---|---|---|---|---|"]
    for flag in FLAGS:
        tune = sds("results/tune/base", {"R": "results/tune/var/rho0.5_summ0.1",
                                         **{c: f"results/tune/{c}" for c in SWEEP}}, flag)
        val = sds("results/val/base_text_eager", {"R": "results/val/var_tau10_v2"}, flag)
        sw = np.array([tune[c] for c in SWEEP])
        for size, src, sd, extra, single in (
                (tune["n"], "tune (measured)", tune["R"], f"{np.median(sw):.4f} [{sw.max():.4f}]", tune["single_arm_sd"]),
                (val["n"], "val (measured)", val["R"], "—", val["single_arm_sd"])):
            lines.append(f"| {flag} | {size} | {src} | {sd:.4f} | {1.96 * sd:.4f} | {2.80 * sd:.4f} | {extra} | {single:.4f} |")
        scaled = {"retinamnist": [(420, "Retina confirmation (val ∪ confirm), scaled")],
                  "dermamnist": [(1000, "D4 Derma resolution set, scaled")]}.get(flag, [])
        for n, src in scaled:
            sd = tune["R"] * np.sqrt(tune["n"] / n)
            sdm = np.median(sw) * np.sqrt(tune["n"] / n)
            lines.append(f"| {flag} | {n} | {src} | {sd:.4f} | {1.96 * sd:.4f} | {2.80 * sd:.4f} | {sdm:.4f} | "
                         f"{tune['single_arm_sd'] * np.sqrt(tune['n'] / n):.4f} |")
    out = ROOT / "results/diag/A1/mde_ovo.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
