"""EXPLORATORY (not pre-registered): does R's ranking gain survive a calibration-invariant AUC? CPU only.

    python scripts/diag_invariant_auc.py

Motivated by A1 (P0): the pre-registered macro OvR AUC on softmaxed `p_option` is NOT invariant to a
per-class prior correction for K > 2 (Derma: base_cal - base = +0.094 on val), so "AUC rises" cannot by
itself separate grounding from a prior shift. `ovo_logratio_auc` (effvar/diag_stats.py) is exactly
invariant to that correction. This reports it for the Phase 4 references vs `base` on tune and val, with
paired bootstrap CIs (10,000, seed 0), plus the sanity row base_cal - base, which must be exactly 0.
Hypothesis-generating only; a proposal for Amendment 2. Writes results/diag/A1/invariant_auc.md.
"""
import numpy as np

from effvar import FLAGS, ROOT
from effvar.diag_stats import (DATASETS, align, apply_bias, boot_idx, ci, fit_bias, fmt, load_run,
                               ovo_logratio_auc, ranking_auc)

SPLITS = {
    "tune": ("results/tune/base", {"R": "results/tune/var/rho0.5_summ0.1",
                                   "var_attn r0.5 s0.1": "results/tune/var_attn/rho0.5_summ0.1"}),
    "val": ("results/val/base_text_eager", {"R": "results/val/var_tau10_v2", "var_tau15": "results/val/var_tau15",
                                            "var_attn_tau10": "results/val/var_attn_tau10",
                                            "var_attn_tau15": "results/val/var_attn_tau15"}),
}


def main() -> None:
    lines = ["# Exploratory: calibration-invariant AUC (OvO on log-ratios) — P0, 2026-09-28", "",
             "NOT pre-registered; hypothesis-generating. OvR = the pre-registered macro OvR AUC; OvO-LR = "
             "macro one-vs-one AUC on log p_c − log p_j, exactly invariant to per-class prior correction. "
             "Paired bootstrap 10,000, seed 0; `*` = CI excludes 0.", ""]
    for flag in FLAGS:
        k, _ = DATASETS[flag]
        lines += [f"## {flag}", "", "| split | arm − base | Δ OvR AUC | Δ OvO-LR AUC | base OvO-LR |", "|---|---|---|---|---|"]
        for split, (base_dir, arms) in SPLITS.items():
            runs = {"base": load_run(ROOT / base_dir, flag), **{a: load_run(ROOT / d, flag) for a, d in arms.items()}}
            _, y, arr = align(runs)
            idx = boot_idx(len(y))
            b_ovr = np.array([ranking_auc(y[ix], arr["base"]["p"][ix], k) for ix in idx])
            b_ovo = np.array([ovo_logratio_auc(y[ix], arr["base"]["p"][ix], k) for ix in idx])
            base_ovo = ovo_logratio_auc(y, arr["base"]["p"], k)
            # sanity: a prior correction fitted on these same images leaves OvO-LR exactly unchanged
            cal = apply_bias(arr["base"]["p"], fit_bias(arr["base"]["p"], y, k))
            assert abs(ovo_logratio_auc(y, cal, k) - base_ovo) < 1e-12, flag
            for a in arms:
                p = arr[a]["p"]
                d_ovr = np.array([ranking_auc(y[ix], p[ix], k) for ix in idx]) - b_ovr
                d_ovo = np.array([ovo_logratio_auc(y[ix], p[ix], k) for ix in idx]) - b_ovo
                lines.append(f"| {split} (n = {len(y)}) | {a} | "
                             f"{fmt(ranking_auc(y, p, k) - ranking_auc(y, arr['base']['p'], k), *ci(d_ovr))} | "
                             f"{fmt(ovo_logratio_auc(y, p, k) - base_ovo, *ci(d_ovo))} | {base_ovo:.4f} |")
        lines.append("")
    lines.append("Sanity (asserted): base_cal − base = 0 exactly under OvO-LR on every split and dataset.")
    out = ROOT / "results/diag/A1/invariant_auc.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
