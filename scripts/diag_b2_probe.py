"""B2 probes, gaps and gates (notes/diagnosis_plan.md §3 B2; Amendment 2). CPU only, on extracted features.

    python scripts/diag_b2_probe.py [--jobs 6]

Refuses to run unless results/diag/B2/extract_check.json shows the instrumented tune passes reproduced the
saved outputs (B2's stop rule).

Probe = StandardScaler -> PCA(256) -> L2 logistic regression, class-balanced. EVERY choice is made by 5-fold
stratified CV (seed 0) inside `<flag>_probe`, scored by OvO-LR AUC on the out-of-fold predictions: the
feature family, the pooling, the layer and C in {1e-4 ... 10}. Each gap term is then ONE pre-selected
feature, refitted on the whole probe set and scored ONCE on tune. There is no max over families on tune.
- img  = best image-token feature: projector output (mean, max) or a layer-2/5/8/9-32 pool (uniform,
         answer-row attention, sink-free attention);
- ans  = best answer-row residual, hidden states 0-34;
- opt  = matrix scaling of the model's own final option log-probs (no PCA);
- uni / sf = best uniform / sink-free pool among R's decision-row layers 15, 16, 19, 29, 31.
Gaps (OvO-LR, deciding): G_route = img - ans (gates C1 C2 C3 C5 D1 D3 at >= 0.02); G_read = ans - opt
(>= 0.03 -> A1, C4, D2); G_target = uni - sf (D1 at >= 0.02); G_info = img - opt (descriptive). Paired
bootstrap on tune, 10,000, seed 0. OvR is reported beside. Also: matched n = 480 refits (descriptive);
answer-row image mass -> base correctness (routing predicts AUC >= 0.60); R vs base answer-row decodability
(5-fold CV on tune, same folds, ans's layer and C); R's realised decision-row budget.
Writes results/diag/B2/{b2.json, b2_report.md}.
"""
import argparse
import json

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from effvar import FLAGS, ROOT
from effvar.data import stratified_indices
from effvar.diag_stats import DATASETS, binary_auc, boot_idx, ci, fmt, ovo_logratio_auc, ranking_auc

FEAT = ROOT / "results" / "tune" / "diag" / "B2"
OUT = ROOT / "results" / "diag" / "B2"
POOL_LAYERS = (2, 5, 8) + tuple(range(9, 33))
POOLS = ("uniform", "attn", "sinkfree")
TARGET_LAYERS = (15, 16, 19, 29, 31)
CS = np.logspace(-4, 1, 6)
THRESH = {"G_route": 0.02, "G_read": 0.03, "G_target": 0.02}


def load(flag, name):
    rows = torch.load(FEAT / flag / f"features_{name}.pt", weights_only=False)
    y = np.array([r["label"] for r in rows])
    return rows, y


DROP_DIM = 443  # deviation, see below


def feature(rows, cand, keep_443=False):
    """Probe features. DEVIATION (P1, 2026-09-28): the extraction stored hidden states as fp16, and the
    massive-activation dimension 443 overflowed fp16 (inf / saturated at 65504) in the image-token pools of
    layers 7-26 on nearly every image; nothing else overflowed. Dim 443 is therefore excluded from EVERY
    residual/projector feature, uniformly across gap terms -- the same dimension R's detector zeroes before
    its RMS (notes/medgemma_probe.md §B). `keep_443` exists only for the answer-row sensitivity check, where
    the stored values are finite."""
    kind = cand[0]
    if kind == "opt":
        return np.log(np.clip(np.stack([r["lens"][-1].numpy() for r in rows]), 1e-12, None))
    if kind == "proj":
        X = np.stack([r["proj"][cand[1]].float().numpy() for r in rows])
    elif kind == "pool":
        li = POOL_LAYERS.index(cand[1])
        X = np.stack([r["pools"][li, POOLS.index(cand[2])].float().numpy() for r in rows])
    elif kind == "ans":
        X = np.stack([r["ans"][cand[1]].float().numpy() for r in rows])
    else:
        raise ValueError(cand)
    if not keep_443:
        X = np.delete(X, DROP_DIM, axis=1)
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite values in {cand} after the dim-{DROP_DIM} exclusion")
    return X


def make(cand, C, n_train, dim):
    steps = [StandardScaler()]
    if cand[0] != "opt":
        steps.append(PCA(n_components=min(256, n_train - 1, dim), random_state=0))
    return steps, LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def fit_predict(cand, C, Xtr, ytr, Xte):
    steps, lr = make(cand, C, len(Xtr), Xtr.shape[1])
    for s in steps:
        Xtr = s.fit_transform(Xtr)
        Xte = s.transform(Xte)
    return lr.fit(Xtr, ytr).predict_proba(Xte)


def cv_score(cand, X, y, k):
    """OOF OvO-LR per C; the transforms are fitted once per fold and shared across C."""
    oof = {C: np.zeros((len(y), k)) for C in CS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        steps, _ = make(cand, 1.0, len(tr), X.shape[1])
        Xtr, Xte = X[tr], X[te]
        for s in steps:
            Xtr = s.fit_transform(Xtr)
            Xte = s.transform(Xte)
        for C in CS:
            oof[C][te] = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr]).predict_proba(Xte)
    scores = {float(C): ovo_logratio_auc(y, oof[C], k) for C in CS}
    best = max(scores, key=scores.get)
    return cand, best, scores[best]


def candidates():
    img = [("proj", 0), ("proj", 1)] + [("pool", ly, p) for ly in POOL_LAYERS for p in POOLS]
    ans = [("ans", ly) for ly in range(35)]
    return img, ans


def name(c):
    return {"proj": lambda c: f"projector {'mean' if c[1] == 0 else 'max'}",
            "pool": lambda c: f"L{c[1]} {c[2]} pool", "ans": lambda c: f"answer row h{c[1]}",
            "opt": lambda c: "matrix-scaled option logits"}[c[0]](c)


def gap_stats(y, pa, pb, k, idx):
    a0, b0 = ovo_logratio_auc(y, pa, k), ovo_logratio_auc(y, pb, k)
    d = np.array([ovo_logratio_auc(y[ix], pa[ix], k) - ovo_logratio_auc(y[ix], pb[ix], k) for ix in idx])
    return [a0 - b0, *ci(d)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    check = json.loads((OUT / "extract_check.json").read_text())
    for flag in FLAGS:
        for p in ("tune", "tune_R"):
            rec = check[f"{flag}/{p}"]
            n = rec["reproduces_saved"].split("/")
            if n[0] != n[1]:
                raise SystemExit(f"B2 stop rule: {flag}/{p} reproduces {rec['reproduces_saved']} -- probes not fitted")

    report = {}
    lines = ["# B2 — information ceiling and decision-row attention (P1, 2026-09-28)", "",
             "Every choice made by 5-fold CV inside the probe set (OvO-LR); each gap term is ONE pre-selected "
             "feature scored ONCE on tune (n = 300). Deciding metric OvO-LR (Amendment 2); OvR beside. Paired "
             "bootstrap 10,000, seed 0; `*` = CI excludes 0. Gates read the point estimate.",
             "**Deviation:** dim 443 is excluded from every probe feature (fp16 storage overflowed it in the "
             "image-token pools); see `feature()` and the per-dataset sensitivity line.", ""]
    for flag in FLAGS:
        k, _ = DATASETS[flag]
        prow, py = load(flag, "probe")
        trow, ty = load(flag, "tune")
        rrow, _ = load(flag, "tune_R")
        assert [r["index"] for r in trow] == [r["index"] for r in rrow]
        img_c, ans_c = candidates()
        scored = Parallel(n_jobs=args.jobs)(delayed(cv_score)(c, feature(prow, c), py, k) for c in img_c + ans_c)
        _, opt_C, opt_cv = cv_score(("opt",), feature(prow, ("opt",)), py, k)
        cv = {c: (C, s) for c, C, s in scored}
        pick = lambda pool: max(pool, key=lambda c: cv[c][1])  # noqa: E731
        sel = {"img": pick(img_c), "ans": pick(ans_c),
               "uni": pick([("pool", ly, "uniform") for ly in TARGET_LAYERS]),
               "sf": pick([("pool", ly, "sinkfree") for ly in TARGET_LAYERS])}
        cv[("opt",)] = (opt_C, opt_cv)
        sel["opt"] = ("opt",)

        def tune_probs(c, sub=None):
            rows, yy = (prow, py) if sub is None else ([prow[i] for i in sub], py[sub])
            return fit_predict(c, cv[c][0], feature(rows, c), yy, feature(trow, c))

        probs = {t: tune_probs(c) for t, c in sel.items()}
        model_p = np.stack([np.asarray(r["p_option"]) / np.sum(r["p_option"]) for r in trow])
        idx = boot_idx(len(ty))
        gaps = {"G_route": gap_stats(ty, probs["img"], probs["ans"], k, idx),
                "G_read": gap_stats(ty, probs["ans"], probs["opt"], k, idx),
                "G_target": gap_stats(ty, probs["uni"], probs["sf"], k, idx),
                "G_info": gap_stats(ty, probs["img"], probs["opt"], k, idx)}
        scores = {t: {"feature": name(c), "C": cv[c][0], "cv_ovo": cv[c][1],
                      "tune_ovo": ovo_logratio_auc(ty, probs[t], k), "tune_ovr": ranking_auc(ty, probs[t], k)}
                  for t, c in sel.items()}
        scores["zero-shot base"] = {"tune_ovo": ovo_logratio_auc(ty, model_p, k), "tune_ovr": ranking_auc(ty, model_p, k)}

        matched = None
        if len(py) > 480:
            sub, _ = stratified_indices(py, list(range(k)), 480, 2, 0)
            mp = {t: tune_probs(c, sub) for t, c in sel.items()}
            matched = {g: ovo_logratio_auc(ty, mp[a], k) - ovo_logratio_auc(ty, mp[b], k)
                       for g, (a, b) in {"G_route": ("img", "ans"), "G_read": ("ans", "opt"),
                                         "G_target": ("uni", "sf"), "G_info": ("img", "opt")}.items()}

        # answer-row image mass -> base correctness (binary AUC of a scalar)
        heads = np.stack([r["heads"].numpy() for r in trow])                       # [n, 34, H, 5]
        mass = heads[..., 0].mean((1, 2))
        sink_share = (heads[..., 1] / np.clip(heads[..., 0], 1e-12, None)).mean((1, 2))
        correct = np.argmax(model_p, 1) == ty
        mass_auc = [binary_auc(correct, mass), *ci([binary_auc(correct[ix], mass[ix]) for ix in idx])]
        share_auc = [binary_auc(correct, -sink_share), *ci([binary_auc(correct[ix], -sink_share[ix]) for ix in idx])]

        # R vs base answer-row decodability, 5-fold CV on tune, same folds, ans's layer and C
        c_ans = sel["ans"]
        oof = {m: np.zeros((len(ty), k)) for m in ("base", "R")}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(ty)), ty):
            for m, rows in (("base", trow), ("R", rrow)):
                X = feature(rows, c_ans)
                oof[m][te] = fit_predict(c_ans, cv[c_ans][0], X[tr], ty[tr], X[te])
        decod = gap_stats(ty, oof["R"], oof["base"], k, idx)

        # sensitivity of the dim-443 exclusion: the selected answer-row feature with 443 kept (finite there)
        sens = {"feature": name(c_ans), "cv_ovo_without": cv[c_ans][1],
                "tune_ovo_without": ovo_logratio_auc(ty, probs["ans"], k),
                "cv_ovo_with_443": float("nan"), "tune_ovo_with_443": float("nan")}
        try:
            Xp, Xt = feature(prow, c_ans, keep_443=True), feature(trow, c_ans, keep_443=True)
            sens_cv = cv_score(c_ans, Xp, py, k)
            sens.update(cv_ovo_with_443=sens_cv[2],
                        tune_ovo_with_443=ovo_logratio_auc(ty, fit_predict(c_ans, sens_cv[1], Xp, py, Xt), k))
        except ValueError:  # dim 443 overflowed in this feature too: no sensitivity number, reported as nan
            pass

        # R's realised decision-row budget
        bl = [r["budget"] for r in rrow]
        layers = sorted({int(ly) for b in bl for ly in b["by_layer"]})
        budget = {"moved_total_mean": float(np.mean([b["moved_total"] for b in bl])), "by_layer": {}}
        for ly in layers:
            ds = [b["by_layer"].get(str(ly)) for b in bl]
            ed = [d for d in ds if d and d["heads_edited"]]
            budget["by_layer"][ly] = {
                "images_edited": len(ed), "heads_edited_mean": float(np.mean([d["heads_edited"] if d else 0 for d in ds])),
                "moved_mean_all_heads": float(np.mean([d["moved"] if d else 0.0 for d in ds])),
                "moved_per_edited_head": float(np.mean([d["moved_edited"] for d in ed])) if ed else None,
                "S_v": float(np.mean([d["S_v"] for d in ed])) if ed else None,
                "S_t": float(np.mean([d["S_t"] for d in ed])) if ed else None,
                "bos": float(np.mean([d["bos"] for d in ed])) if ed else None}

        gates = {g: gaps[g][0] >= t for g, t in THRESH.items()}
        report[flag] = {"n_probe": len(py), "n_tune": len(ty), "selected": scores, "gaps": gaps, "gates": gates,
                        "matched_480": matched, "mass_to_correct_auc": mass_auc,
                        "low_sink_share_to_correct_auc": share_auc, "R_minus_base_decodability": decod,
                        "dim443_sensitivity": sens,
                        "budget": budget, "cv_all": {name(c): [cv[c][0], cv[c][1]] for c in cv}}
        lines += [f"## {flag} (probe n = {len(py)}, tune n = {len(ty)})", "",
                  "| term | selected feature (CV) | C | CV OvO | tune OvO | tune OvR |", "|---|---|---|---|---|---|"]
        for t, s in scores.items():
            lines.append(f"| {t} | {s.get('feature', '—')} | {s.get('C', '—')} | "
                         f"{s['cv_ovo']:.4f} | {s['tune_ovo']:.4f} | {s['tune_ovr']:.4f} |" if "C" in s else
                         f"| {t} | — | — | — | {s['tune_ovo']:.4f} | {s['tune_ovr']:.4f} |")
        lines += ["", "| gap | OvO-LR (tune) | threshold | gate | matched n = 480 |", "|---|---|---|---|---|"]
        for g, v in gaps.items():
            thr = THRESH.get(g)
            lines.append(f"| {g} | {fmt(*v)} | {thr if thr else 'descriptive'} | "
                         f"{('PASS' if gates[g] else 'fail') if thr else '—'} | "
                         f"{'—' if matched is None else f'{matched[g]:+.4f}'} |")
        top = [(ly, d) for ly, d in budget["by_layer"].items() if d["images_edited"]]
        top = sorted(top, key=lambda x: -x[1]["moved_mean_all_heads"])[:6]
        lines += ["", f"- Answer-row image mass → base correct: AUC {mass_auc[0]:.3f} [{mass_auc[1]:.3f}, {mass_auc[2]:.3f}] "
                      f"(routing predicts ≥ 0.60); low sink share → correct: {share_auc[0]:.3f} [{share_auc[1]:.3f}, {share_auc[2]:.3f}].",
                  f"- R − base answer-row decodability ({name(c_ans)}, 5-fold CV on tune): {fmt(*decod)}.",
                  f"- Dim-443 sensitivity ({sens['feature']}): tune OvO {sens['tune_ovo_without']:.4f} without, "
                  f"{sens['tune_ovo_with_443']:.4f} with (CV {sens['cv_ovo_without']:.4f} / {sens['cv_ovo_with_443']:.4f}).",
                  f"- R's realised decision-row budget: mass moved summed over layers (mean over heads) = "
                  f"{budget['moved_total_mean']:.4f} per image. Top layers: " +
                  "; ".join(f"L{ly}: {d['heads_edited_mean']:.2f} heads, {d['moved_per_edited_head']:.3f} per edited head "
                            f"(S_v {d['S_v']:.3f}, S_t {d['S_t']:.3f}, BOS {d['bos']:.3f})" for ly, d in top) + ".", ""]
    live = [f for f in FLAGS if report[f]["gates"]["G_route"]]
    lines += ["## Gates", "",
              f"- G_route ≥ 0.02 (attention cards live): {live or 'none'}; harm floor only: "
              f"{[f for f in FLAGS if f not in live]}.",
              f"- Attention line {'CLOSED (G_route < 0.02 on all three)' if not live else 'open'}.",
              f"- G_read ≥ 0.03 (→ A1, C4, D2): {[f for f in FLAGS if report[f]['gates']['G_read']] or 'none'}.",
              f"- D1 gate (G_route or G_target ≥ 0.02): "
              f"{[f for f in FLAGS if report[f]['gates']['G_route'] or report[f]['gates']['G_target']] or 'none'}.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b2.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "b2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
