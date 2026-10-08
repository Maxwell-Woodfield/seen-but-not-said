"""A1 — calibrated baselines (notes/diagnosis_plan.md §3 A1, Amendment 1). CPU only, saved outputs only.

    python scripts/diag_a1.py

Fits on tune: `base` = results/tune/base, `R` = results/tune/var/rho0.5_summ0.1, minus the tune images that
are pixel-identical to a val/test image (Amendment 1: data/subsets/<flag>_tune_overlap.json).
- base_cal / R_cal: NLL prior correction (the pre-registered calibrated base), fitted per model;
- base_calT / R_calT: decision-tuned (macro-F1 offsets; Retina: expected-grade cut-points for QWK).
Exploration: 5-fold stratified CV on tune (seed 0; each fold's fit applied to its held-out fold, pooled).
Confirmation: fits on all of tune, applied unchanged to val (results/val/base_text_eager, var_tau10_v2).
Raw arms use the saved text-parsed prediction; calibrated arms argmax the recalibrated p. AUC for a
decision-tuned arm is its raw model's (the rule changes decisions, not scores).
Paired bootstrap over images, 10,000 resamples, seed 0, with the fits held fixed (the CIs cover image
sampling, not fit variance). Writes results/diag/A1/{a1.json, a1_report.md}.
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from effvar import FLAGS, ROOT
from effvar.diag_stats import (DATASETS, align, apply_bias, apply_decision, boot_idx, ci, fit_bias, fit_decision,
                               fmt, load_run, primary, ranking_auc)

RUNS = {"tune": {"base": "results/tune/base", "R": "results/tune/var/rho0.5_summ0.1"},
        "val": {"base": "results/val/base_text_eager", "R": "results/val/var_tau10_v2"}}
ARMS = ("base", "base_cal", "base_calT", "R", "R_cal", "R_calT")
CONTRASTS = (("base_cal", "base"), ("R", "base"), ("R_cal", "base_cal"), ("base_cal", "R"),
             ("R_calT", "base_calT"), ("base_calT", "R"))
OUT = ROOT / "results" / "diag" / "A1"


def fit_all(p, y, k, ordinal):
    return {"bias": fit_bias(p, y, k), "decision": fit_decision(p, y, k, ordinal)}


def arms_from(fits, arr, k):
    """{arm: (pred, p_for_auc)} for one image set, given fits per model."""
    out = {}
    for m in ("base", "R"):
        p, f = arr[m]["p"], fits[m]
        pc = apply_bias(p, f["bias"])
        out[m] = (arr[m]["pred"], p)
        out[f"{m}_cal"] = (np.argmax(pc, 1), pc)
        out[f"{m}_calT"] = (apply_decision(f["decision"], p, k), p)
    return out


def evaluate(y, arms, k, ordinal, ix=None):
    ix = np.arange(len(y)) if ix is None else ix
    yb = y[ix]
    auc_cache = {}
    res = {}
    for name, (pred, p) in arms.items():
        key = id(p)
        if key not in auc_cache:
            auc_cache[key] = ranking_auc(yb, p[ix], k)
        res[name] = (primary(yb, pred[ix], k, ordinal), auc_cache[key])
    return res


def summarise(y, arms, k, ordinal):
    point = evaluate(y, arms, k, ordinal)
    idx = boot_idx(len(y))
    boot = {a: np.empty((len(idx), 2)) for a in arms}
    for b, ix in enumerate(idx):
        for a, v in evaluate(y, arms, k, ordinal, ix).items():
            boot[a][b] = v
    table = {a: {"primary": point[a][0], "auc": point[a][1],
                 "pred_share": np.bincount(arms[a][0], minlength=k).tolist()} for a in arms}
    contrasts = {}
    for a, b in CONTRASTS:
        d = boot[a] - boot[b]
        contrasts[f"{a} - {b}"] = {m: [point[a][j] - point[b][j], *ci(d[:, j])] for j, m in enumerate(("primary", "auc"))}
    return table, contrasts


def flips(y, arms, a="R_cal", b="base_cal"):
    pa, pb = arms[a][0], arms[b][0]
    moved = pa != pb
    trans = {}
    for u, v in zip(pb[moved], pa[moved]):
        trans[f"{'ABCDEFG'[u]}->{'ABCDEFG'[v]}"] = trans.get(f"{'ABCDEFG'[u]}->{'ABCDEFG'[v]}", 0) + 1
    cor = int(((pa == y) & (pb != y)).sum())
    bro = int(((pb == y) & (pa != y)).sum())
    n = int(moved.sum())
    return {"n": n, "corrected": cor, "broken": bro, "neutral": n - cor - bro,
            "dominant_share": max(trans.values()) / n if n else None,
            "transitions": dict(sorted(trans.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report, lines = {}, ["# A1 — calibrated baselines (P0, 2026-09-28)", "",
                         "Fit on tune (Amendment 1 exclusions applied), 5-fold CV on tune = exploration, "
                         "fits frozen from tune applied to val = confirmation. Paired bootstrap 10,000, seed 0. "
                         "`*` = CI excludes 0. primary = QWK (Retina) / macro-F1.", ""]
    for flag in FLAGS:
        k, ordinal = DATASETS[flag]
        excl = set(json.loads((ROOT / f"data/subsets/{flag}_tune_overlap.json").read_text())["exclude_tune_indices"])
        tune_idx, ty, tarr = align({m: load_run(ROOT / d, flag) for m, d in RUNS["tune"].items()}, exclude=excl)
        _, vy, varr = align({m: load_run(ROOT / d, flag) for m, d in RUNS["val"].items()})

        # exploration: 5-fold CV on tune
        cv_arms = {a: [np.empty(len(ty), dtype=int), np.empty((len(ty), k))] for a in ARMS}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(ty)), ty):
            fits = {m: fit_all(tarr[m]["p"][tr], ty[tr], k, ordinal) for m in ("base", "R")}
            sub = {m: {"pred": tarr[m]["pred"][te], "p": tarr[m]["p"][te]} for m in ("base", "R")}
            for a, (pred, p) in arms_from(fits, sub, k).items():
                cv_arms[a][0][te], cv_arms[a][1][te] = pred, p
        # raw-score arms must share one array so their AUC is computed once and identically
        cv_arms = {a: tuple(v) for a, v in cv_arms.items()}
        for m in ("base", "R"):
            cv_arms[f"{m}_calT"] = (cv_arms[f"{m}_calT"][0], cv_arms[m][1])

        # confirmation: fit on all of tune, apply to val
        fits = {m: fit_all(tarr[m]["p"], ty, k, ordinal) for m in ("base", "R")}
        val_arms = arms_from(fits, varr, k)

        report[flag] = {"n_tune": len(ty), "tune_excluded": sorted(excl), "n_val": len(vy),
                        "fits": {m: {"bias": fits[m]["bias"].tolist(),
                                     "decision": {kk: (v.tolist() if hasattr(v, "tolist") else v)
                                                  for kk, v in fits[m]["decision"].items()}} for m in fits}}
        lines += [f"## {flag} (tune n = {len(ty)}, excluded {sorted(excl) or 'none'}; val n = {len(vy)})", ""]
        for split, y, arms in (("tune 5-fold CV", ty, cv_arms), ("val (confirmation)", vy, val_arms)):
            table, contrasts = summarise(y, arms, k, ordinal)
            fl = flips(y, arms)
            report[flag][split] = {"arms": table, "contrasts": contrasts, "flips_Rcal_vs_basecal": fl}
            lines += [f"### {split}", "", "| arm | primary | AUC | predicted-class counts |", "|---|---|---|---|"]
            lines += [f"| {a} | {t['primary']:.4f} | {t['auc']:.4f} | {t['pred_share']} |" for a, t in table.items()]
            lines += ["", "| contrast | Δ primary | Δ AUC |", "|---|---|---|"]
            lines += [f"| {c} | {fmt(*v['primary'])} | {fmt(*v['auc'])} |" for c, v in contrasts.items()]
            lines += ["", f"Flips R_cal vs base_cal: {fl['n']} (corrected {fl['corrected']}, broken {fl['broken']}, "
                          f"neutral {fl['neutral']}; dominant share {fl['dominant_share']}) {fl['transitions']}", ""]
    (OUT / "a1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "a1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
