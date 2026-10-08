"""A2 — shift decomposition of existing VAR runs against `base` (notes/diagnosis_plan.md §3 A2). CPU only.

    python scripts/diag_a2.py

Per image, delta = centred log p_option(variant) - centred log p_option(base), centred over classes.
- CM share = ||mean delta||^2 / mean ||delta||^2: 1 = every image shifted identically (a bias).
- alignment = mean_i r_i[y_i], r = delta - mean delta: does the image-specific part push toward the
  true class? (log-prob units; > 0 = toward.)
Runs: tune = the 24 gate-sweep configs (var, var_attn), the topm curve (full_m24 is bitwise R), base =
results/tune/base; val = var_tau10_v2 (R), var_tau15, var_attn_tau10/15, effvar_topm24_v2, base =
base_text_eager. Paired bootstrap over images, 10,000 resamples, seed 0, the same resamples for every run
of a dataset. Spearman(alignment, dAUC) across the tune configs checks that alignment tracks ranking.
Writes results/diag/A2/{a2.json, a2_report.md}.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from effvar import FLAGS, ROOT
from effvar.diag_stats import (DATASETS, align, boot_idx, centred_logp, ci, fmt, load_run, primary, ranking_auc,
                               shift_stats)

OUT = ROOT / "results" / "diag" / "A2"
R_TUNE = "var/rho0.5_summ0.1"


def tune_runs(flag):
    runs = {}
    for fam in ("var", "var_attn"):
        for d in sorted((ROOT / "results/tune" / fam).iterdir()):
            if (d / flag / "predictions.jsonl").exists():
                runs[f"{fam}/{d.name}"] = d
    for d in sorted((ROOT / "results/tune/topm").iterdir()):
        if (d / flag / "predictions.jsonl").exists():
            runs[f"topm/{d.name}"] = d
    return {"base": ROOT / "results/tune/base", **runs}


def val_runs():
    names = ("var_tau10_v2", "var_tau15", "var_attn_tau10", "var_attn_tau15", "effvar_topm24_v2")
    return {"base": ROOT / "results/val/base_text_eager", **{n: ROOT / "results/val" / n for n in names}}


def analyse(flag, dirs):
    k, ordinal = DATASETS[flag]
    _, y, arr = align({name: load_run(d, flag) for name, d in dirs.items()})
    idx = boot_idx(len(y))
    zb = centred_logp(arr["base"]["p"])
    base_auc = np.array([ranking_auc(y[ix], arr["base"]["p"][ix], k) for ix in idx])
    base_prim = primary(y, arr["base"]["pred"], k, ordinal)
    base_auc0 = ranking_auc(y, arr["base"]["p"], k)
    rows = {}
    for name, a in arr.items():
        if name == "base":
            continue
        delta = centred_logp(a["p"]) - zb
        (cm0,), (al0,) = shift_stats(delta, y)
        cm_b, al_b = shift_stats(delta, y, idx)
        auc_b = np.array([ranking_auc(y[ix], a["p"][ix], k) for ix in idx]) - base_auc
        rows[name] = {"n": int(len(y)),
                      "d_primary": primary(y, a["pred"], k, ordinal) - base_prim,
                      "d_auc": [ranking_auc(y, a["p"], k) - base_auc0, *ci(auc_b)],
                      "cm": [float(cm0), *ci(cm_b)], "alignment": [float(al0), *ci(al_b)],
                      "mean_delta": delta.mean(0).round(4).tolist(),
                      "flips": int((a["pred"] != arr["base"]["pred"]).sum())}
    return rows


def table(rows):
    out = ["| run | Δ primary | Δ AUC | CM share | alignment | flips | mean Δ (by class) |", "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        cm, al = r["cm"], r["alignment"]
        out.append(f"| {name} | {r['d_primary']:+.4f} | {fmt(*r['d_auc'])} | {cm[0]:.3f} [{cm[1]:.3f},{cm[2]:.3f}] "
                   f"| {fmt(*al)} | {r['flips']} | {r['mean_delta']} |")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    lines = ["# A2 — shift decomposition of existing VAR runs vs base (P0, 2026-09-28)", "",
             "CM share = ||mean Δ||² / mean ||Δ||² (1 = a pure bias); alignment = image-specific push toward "
             "the true class (log-prob units). Paired bootstrap 10,000, seed 0; `*` = CI excludes 0.", ""]
    for flag in FLAGS:
        tune = analyse(flag, tune_runs(flag))
        val = analyse(flag, val_runs())
        configs = [n for n in tune if n != "topm/full_m24"]          # full_m24 is bitwise R
        al = [tune[n]["alignment"][0] for n in configs]
        cm = [tune[n]["cm"][0] for n in configs]
        dauc = [tune[n]["d_auc"][0] for n in configs]
        rho_al, p_al = spearmanr(al, dauc)
        rho_cm, p_cm = spearmanr(cm, dauc)
        report[flag] = {"tune": tune, "val": val,
                        "spearman_alignment_dauc": [float(rho_al), float(p_al), len(configs)],
                        "spearman_cm_dauc": [float(rho_cm), float(p_cm), len(configs)]}
        r = tune[R_TUNE]
        lines += [f"## {flag}", "",
                  f"**R (tune):** CM {r['cm'][0]:.3f} [{r['cm'][1]:.3f},{r['cm'][2]:.3f}], alignment "
                  f"{fmt(*r['alignment'])}, ΔAUC {fmt(*r['d_auc'])}. **R (val):** CM "
                  f"{val['var_tau10_v2']['cm'][0]:.3f} [{val['var_tau10_v2']['cm'][1]:.3f},"
                  f"{val['var_tau10_v2']['cm'][2]:.3f}], alignment {fmt(*val['var_tau10_v2']['alignment'])}, "
                  f"ΔAUC {fmt(*val['var_tau10_v2']['d_auc'])}.",
                  f"Across {len(configs)} tune configs: Spearman(alignment, ΔAUC) = {rho_al:+.3f} (p = {p_al:.2g}); "
                  f"Spearman(CM, ΔAUC) = {rho_cm:+.3f} (p = {p_cm:.2g}).", "", "### val", ""]
        lines += table(val) + ["", "### tune", ""] + table(tune) + [""]
    (OUT / "a2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "a2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
