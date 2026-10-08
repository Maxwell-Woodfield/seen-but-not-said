"""Amendment 3 (f), POST HOC and descriptive (gates nothing): per-layer answer-row decodability, h0-h34, for
`base` and `R` on tune, from the saved B2 features. CPU only.

    python scripts/diag_depth.py [--jobs 6]

Probe = B2's (StandardScaler -> PCA(256) -> class-balanced L2 logistic; dim 443 excluded by B2's `feature()`),
each layer's C = the value B2's probe-set CV selected for that layer (b2.json `cv_all`), 5-fold stratified CV
on tune with B2's folds (seed 0; all 300 tune images, as in B2), OvO-LR on the out-of-fold predictions, OvR
beside. R - base per layer: paired bootstrap 10,000, seed 0. Where base and R features are identical (the
layers before R's band) the difference is 0 by construction; the first differing layer is found, not assumed.
Writes results/diag/B2_depth/{depth.json, depth_report.md}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_b2_probe import feature, fit_predict, gap_stats, load  # noqa: E402  (B2's probe, unchanged)

from effvar import FLAGS, ROOT  # noqa: E402
from effvar.diag_stats import DATASETS, boot_idx, fmt, ovo_logratio_auc, ranking_auc  # noqa: E402

B2 = ROOT / "results" / "diag" / "B2"
OUT = ROOT / "results" / "diag" / "B2_depth"
LAYERS = range(35)


def oof(X, y, k, C):
    cand = ("ans", None)   # only cand[0] matters to B2's make(): PCA on, as for every answer-row feature
    out = np.zeros((len(y), k))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        out[te] = fit_predict(cand, C, X[tr], y[tr], X[te])
    return out


def boot_gap(y, pa, pb, k):
    return gap_stats(y, pa, pb, k, boot_idx(len(y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    b2 = json.loads((B2 / "b2.json").read_text(encoding="utf-8"))
    report = {}
    lines = ["# Answer-row decodability by depth, h0-h34 (Amendment 3f; POST HOC, descriptive)", "",
             "B2's probe and folds; each layer's C from B2's probe-set CV; 5-fold CV on tune (n = 300), OvO-LR on "
             "out-of-fold predictions (OvR beside). R − base: paired bootstrap 10,000, seed 0; `*` = CI excludes 0. "
             "`probe CV` = B2's own probe-set CV score for that layer (base features). Gates nothing.", ""]
    summary = []
    for flag in FLAGS:
        k, _ = DATASETS[flag]
        trow, y = load(flag, "tune")
        rrow, yr = load(flag, "tune_R")
        assert [r["index"] for r in trow] == [r["index"] for r in rrow] and (y == yr).all()
        X = {m: [feature(rows, ("ans", ly)) for ly in LAYERS] for m, rows in (("base", trow), ("R", rrow))}
        same = [bool(np.array_equal(X["base"][ly], X["R"][ly])) for ly in LAYERS]
        first_diff = next((ly for ly in LAYERS if not same[ly]), None)
        cv_all = b2[flag]["cv_all"]
        Cs = [cv_all[f"answer row h{ly}"][0] for ly in LAYERS]
        jobs = [(m, ly) for m in ("base", "R") for ly in LAYERS if m == "base" or not same[ly]]
        res = Parallel(n_jobs=args.jobs)(delayed(oof)(X[m][ly], y, k, Cs[ly]) for m, ly in jobs)
        P = {"base": {}, "R": {}}
        for (m, ly), p in zip(jobs, res):
            P[m][ly] = p
        for ly in LAYERS:
            if same[ly]:
                P["R"][ly] = P["base"][ly]   # identical features, identical folds -> identical probe
        diff_layers = [ly for ly in LAYERS if not same[ly]]
        gaps = dict(zip(diff_layers, Parallel(n_jobs=args.jobs)(
            delayed(boot_gap)(y, P["R"][ly], P["base"][ly], k) for ly in diff_layers)))
        per = {}
        for ly in LAYERS:
            per[ly] = {"C": Cs[ly], "probe_cv_ovo": cv_all[f"answer row h{ly}"][1],
                       "base_ovo": ovo_logratio_auc(y, P["base"][ly], k), "R_ovo": ovo_logratio_auc(y, P["R"][ly], k),
                       "base_ovr": ranking_auc(y, P["base"][ly], k), "R_ovr": ranking_auc(y, P["R"][ly], k),
                       "R_minus_base_ovo": gaps.get(ly, [0.0, 0.0, 0.0]), "identical_features": same[ly]}
        peak = {m: max(LAYERS, key=lambda ly: per[ly][f"{m}_ovo"]) for m in ("base", "R")}
        sel = b2[flag]["selected"]
        report[flag] = {"n": len(y), "first_layer_R_differs": first_diff, "peak": peak, "by_layer": per,
                        "zero_shot_base_ovo": sel["zero-shot base"]["tune_ovo"], "opt_tune_ovo": sel["opt"]["tune_ovo"],
                        "b2_selected_ans": sel["ans"]["feature"]}
        summary.append(
            f"- **{flag}:** base peak h{peak['base']} {per[peak['base']]['base_ovo']:.4f}, final h34 "
            f"{per[34]['base_ovo']:.4f}; R peak h{peak['R']} {per[peak['R']]['R_ovo']:.4f}, final h34 {per[34]['R_ovo']:.4f} "
            f"(R − base at h34 {fmt(*per[34]['R_minus_base_ovo'])}). Zero-shot option readout "
            f"{report[flag]['zero_shot_base_ovo']:.4f}; B2 selected {sel['ans']['feature']}; R's features first "
            f"differ at h{first_diff}.")
        lines += [f"## {flag} (tune n = {len(y)}; R's answer-row features first differ at h{first_diff})", "",
                  "| layer | C | probe CV | base OvO | R OvO | R − base OvO | base OvR | R OvR |", "|---|---|---|---|---|---|---|---|"]
        for ly in LAYERS:
            r = per[ly]
            dtxt = "0 (identical)" if r["identical_features"] else fmt(*r["R_minus_base_ovo"])
            lines.append(f"| h{ly} | {r['C']:g} | {r['probe_cv_ovo']:.4f} | {r['base_ovo']:.4f} | {r['R_ovo']:.4f} | "
                         f"{dtxt} | {r['base_ovr']:.4f} | {r['R_ovr']:.4f} |")
        lines.append("")
    lines = lines[:4] + ["## Summary", ""] + summary + [""] + lines[4:]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "depth.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "depth_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + 2 + len(summary)]))


if __name__ == "__main__":
    main()
