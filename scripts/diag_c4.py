"""C4 — option order reversed (notes/diagnosis_plan.md §4 C4; Amendments 2 and 3). Tune only.

    python scripts/run_eval.py --method base --attn var_eager --subset tune --cache-vision --prompt reversed \
        --exp-name tune/diag/C4/base_rev
    python scripts/run_eval.py --method var --attn var_eager --subset tune --cache-vision --prompt reversed \
        --exp-name tune/diag/C4/R_rev
    python scripts/diag_c4.py      # GPU once: reversed content-free probe (18 passes); then CPU analysis

Arms: `base`, `R` = the saved frozen-order tune runs; `base_rev`, `R_rev` = the reversed prompt
(effvar/prompt_variants.py: options listed in reverse class order, lettered A.. in listing order, so A names
the last class). Letters are mapped back to classes at scoring, so every `p_option` / `pred` here is
class-aligned. Amendment 1's exclusions apply, as in A1 (Pneumonia n = 298).
- Deciding: macro OvO log-ratio AUC. Beside it: OvR (raw and calibrated), primary (QWK Retina / macro-F1),
  raw and calibrated. `<arm>_cal` = NLL per-class offsets (`fit_bias`), 5-fold stratified CV on tune with
  A1's folds (seed 0); `base_cal` is A1's tune-CV base_cal (asserted equal to a1.json).
- Guards (§2): leakage ≤ base + 2 pp and compliance ≥ 0.99, else the arm is broken and not interpreted.
- Card predictions: c -> |dOvO(base_rev - base)| ≥ 0.02; grounding -> R's dOvO is order-invariant (the
  interaction dOvO(R_rev - base_rev) - dOvO(R - base) has a CI containing 0 and a small point estimate);
  letter prior -> the excess follows letter A (the last class under reversal); class prior -> it stays on
  the class.
- EXPLORATORY (not pre-registered): the order-averaged comparator `<m>_avg` = softmax of the mean of the
  class-aligned log p over both orders, raw and 5-fold-CV calibrated.
- POST HOC, descriptive (Amendment 3f): B1's content-free probe under the reversed order.
Paired bootstrap over images, 10,000 resamples, seed 0, one set shared by every arm; the calibration fits
are held fixed. Writes results/diag/C4/{content_free_rev.json, c4.json, c4_report.md}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_b1 import CF_NAMES, content_free_images  # noqa: E402  (the same content-free images as B1)

from effvar import FLAGS, ROOT  # noqa: E402
from effvar.diag_stats import (DATASETS, align, apply_bias, boot_idx, ci, fit_bias, fmt, load_run,  # noqa: E402
                               ovo_logratio_auc, primary, ranking_auc)

OUT = ROOT / "results" / "diag" / "C4"
RUNS = {"base": "results/tune/base", "R": "results/tune/var/rho0.5_summ0.1",
        "base_rev": "results/tune/diag/C4/base_rev", "R_rev": "results/tune/diag/C4/R_rev"}
ORDER = {"base": "frozen", "R": "frozen", "base_rev": "reversed", "R_rev": "reversed"}
METRICS = ("ovo", "ovr", "ovr_cal", "primary", "primary_cal")
LETTERS = "ABCDEFG"


# --------------------------------------------------------------------------- GPU: content-free, reversed
def run_gpu():
    from effvar.generation import run_untimed
    from effvar.methods import make_method
    from effvar.model import load_model, load_processor
    from effvar.prompt_variants import reversed_spec
    from effvar.scoring import first_token_metrics, option_token_sets

    processor = load_processor()
    model = load_model("var_eager")
    methods = {"base": make_method("base", attn="var_eager"), "R": make_method("var", attn="var_eager")}
    out = {}
    for flag in FLAGS:
        spec = reversed_spec(flag)
        sets = option_token_sets(processor.tokenizer, spec)   # class-aligned
        out[flag] = {"prompt": spec.text(), "labels_by_class": list(spec.labels)}
        for name, img in content_free_images(flag).items():
            for m, method in methods.items():
                gen = run_untimed(model, processor, method, spec, img)
                ft = first_token_metrics(gen.first_logits, sets, 0)
                out[flag].setdefault(m, {})[name] = {"p_option": ft["p_option"], "leakage": ft["leakage"],
                                                     "text": gen.text}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "content_free_rev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- CPU analysis
def letter_of(c, k, order):
    return LETTERS[c] if order == "frozen" else LETTERS[k - 1 - c]


def check_provenance(flag):
    """The reversed runs used the reversed prompt, the full tune subset and R's exact parameters."""
    cfg = {a: yaml.safe_load((ROOT / d / flag / "config.yaml").read_text(encoding="utf-8")) for a, d in RUNS.items()}
    k = DATASETS[flag][0]
    for a in ("base_rev", "R_rev"):
        assert cfg[a].get("prompt_variant") == "reversed", (flag, a)
        assert cfg[a]["labels"] == [LETTERS[k - 1 - c] for c in range(k)], (flag, a)
        assert cfg[a]["subset"] == cfg["base"]["subset"] and cfg[a]["limit"] is None, (flag, a)
        assert cfg[a]["attn"] == cfg["base"]["attn"] == "var_eager", (flag, a)
    assert cfg["R_rev"]["method"] == cfg["R"]["method"] and cfg["R_rev"]["method_params"] == cfg["R"]["method_params"]
    assert cfg["base_rev"]["method"] == "base"
    return {a: cfg[a]["git"] for a in ("base_rev", "R_rev")}


def guards(run_dir, flag, keep):
    rows = {}
    with open(ROOT / run_dir / flag / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[int(r["index"])] = (r["leakage"], r["first_token_in_options"])
    return float(np.mean([rows[i][0] for i in keep])), float(np.mean([rows[i][1] for i in keep]))


def cv_calibrate(p, y, k):
    """5-fold stratified CV (A1's folds, seed 0): each fold's NLL offsets applied to its held-out fold."""
    out = np.empty_like(p)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        out[te] = apply_bias(p[te], fit_bias(p[tr], y[tr], k))
    return out


def metric_row(y, pred, p, pred_cal, p_cal, k, ordinal, ix):
    yy = y[ix]
    return (ovo_logratio_auc(yy, p[ix], k), ranking_auc(yy, p[ix], k), ranking_auc(yy, p_cal[ix], k),
            primary(yy, pred[ix], k, ordinal), primary(yy, pred_cal[ix], k, ordinal))


def boot_arm(y, arm, k, ordinal, idx):
    return np.array([metric_row(y, *arm, k, ordinal, ix) for ix in idx])


def flips(y, a, b, k):
    """Report-only (Amendment 2c): images whose class prediction moved from arm b to arm a."""
    moved = a != b
    trans = {}
    for u, v in zip(b[moved], a[moved]):
        trans[f"{u}->{v}"] = trans.get(f"{u}->{v}", 0) + 1
    n = int(moved.sum())
    cor, bro = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    return {"n": n, "corrected": cor, "broken": bro, "neutral": n - cor - bro,
            "dominant_share": max(trans.values()) / n if n else None,
            "transitions": dict(sorted(trans.items(), key=lambda kv: -kv[1]))}


def analyse(cf_rev):
    cf_fwd = json.loads((ROOT / "results/diag/B1/content_free.json").read_text(encoding="utf-8"))
    a1 = json.loads((ROOT / "results/diag/A1/a1.json").read_text(encoding="utf-8"))
    report = {}
    L = ["# C4 — option order reversed (P2, 2026-09-28)", "",
         "Tune, {base, R} × {frozen, reversed}; letters mapped back to classes before scoring. OvO-LR decides "
         "(Amendment 2); `_cal` = NLL offsets, 5-fold CV on tune (A1's folds). Paired bootstrap 10,000, seed 0; "
         "`*` = CI excludes 0. `_avg` arms are EXPLORATORY; the content-free section is POST HOC (Amendment 3f).",
         ""]
    for flag in FLAGS:
        k, ordinal = DATASETS[flag]
        git = check_provenance(flag)
        excl = set(json.loads((ROOT / f"data/subsets/{flag}_tune_overlap.json").read_text())["exclude_tune_indices"])
        keep, y, arr = align({a: load_run(ROOT / d, flag) for a, d in RUNS.items()}, exclude=excl)
        n = len(y)
        for m in ("base", "R"):   # exploratory order-averaged comparator
            z = 0.5 * (np.log(np.clip(arr[m]["p"], 1e-12, None)) + np.log(np.clip(arr[f"{m}_rev"]["p"], 1e-12, None)))
            pa = np.exp(z - z.max(1, keepdims=True))
            pa /= pa.sum(1, keepdims=True)
            arr[f"{m}_avg"] = {"pred": pa.argmax(1), "p": pa}
        arms = {}
        for a, v in arr.items():
            pc = cv_calibrate(v["p"], y, k)
            arms[a] = (v["pred"], v["p"], pc.argmax(1), pc)
        a1_bc = a1[flag]["tune 5-fold CV"]["arms"]["base_cal"]["primary"]
        assert abs(primary(y, arms["base"][2], k, ordinal) - a1_bc) < 1e-12, (flag, "base_cal differs from A1")

        idx = boot_idx(n)
        point = {a: np.array(metric_row(y, *arm, k, ordinal, np.arange(n))) for a, arm in arms.items()}
        boot = dict(zip(arms, Parallel(n_jobs=len(arms))(delayed(boot_arm)(y, arm, k, ordinal, idx)
                                                         for arm in arms.values())))
        # base_cal as an arm: base's calibrated columns in the raw slots
        cal_cols = [0, 2, 2, 4, 4]
        point["base_cal"], boot["base_cal"] = point["base"][cal_cols], boot["base"][:, cal_cols]

        def d(a, b, j):
            return [float(point[a][j] - point[b][j]), *ci(boot[a][:, j] - boot[b][:, j])]

        def dd(a, b, c, e, j):   # (a - b) - (c - e)
            x = boot[a][:, j] - boot[b][:, j] - boot[c][:, j] + boot[e][:, j]
            return [float(point[a][j] - point[b][j] - point[c][j] + point[e][j]), *ci(x)]

        g = {a: guards(RUNS[a], flag, keep) for a in RUNS}
        broken = {a: bool(g[a][1] < 0.99 or (a != "base" and g[a][0] > g["base"][0] + 0.02)) for a in g}
        contrasts = {}
        for v in ("R", "base_rev", "R_rev", "base_avg", "R_avg"):
            contrasts[f"{v} − base"] = {"ovo": d(v, "base", 0), "ovr": d(v, "base", 1), "primary": d(v, "base", 3)}
            contrasts[f"{v} − base_cal"] = {"ovo": d(v, "base_cal", 0), "ovr": d(v, "base_cal", 1),
                                           "primary": d(v, "base_cal", 3)}
            contrasts[f"{v}_cal − base_cal"] = {"ovo": d(v, "base", 0), "ovr": d(v, "base", 2),
                                               "primary": d(v, "base", 4)}
        contrasts["R_rev − base_rev"] = {"ovo": d("R_rev", "base_rev", 0), "ovr": d("R_rev", "base_rev", 1),
                                         "primary": d("R_rev", "base_rev", 3)}
        contrasts["R_rev_cal − base_rev_cal"] = {"ovo": d("R_rev", "base_rev", 0), "ovr": d("R_rev", "base_rev", 2),
                                                 "primary": d("R_rev", "base_rev", 4)}
        contrasts["R_avg − base_avg"] = {"ovo": d("R_avg", "base_avg", 0), "ovr": d("R_avg", "base_avg", 1),
                                         "primary": d("R_avg", "base_avg", 3)}
        inter = {"ovo": dd("R_rev", "base_rev", "R", "base", 0), "ovr": dd("R_rev", "base_rev", "R", "base", 1),
                 "primary": dd("R_rev", "base_rev", "R", "base", 3),
                 "primary_cal": dd("R_rev", "base_rev", "R", "base", 4)}
        shift = d("base_rev", "base", 0)

        counts = {a: np.bincount(arr[a]["pred"], minlength=k).tolist() for a in RUNS}
        letters = {a: [counts[a][[c for c in range(k) if letter_of(c, k, ORDER[a]) == LETTERS[i]][0]]
                       for i in range(k)] for a in RUNS}
        fl = {"R vs base": flips(y, arr["R"]["pred"], arr["base"]["pred"], k),
              "R_rev vs base_rev": flips(y, arr["R_rev"]["pred"], arr["base_rev"]["pred"], k),
              "base_rev vs base": flips(y, arr["base_rev"]["pred"], arr["base"]["pred"], k)}

        cf = {}
        for order, src in (("frozen", cf_fwd), ("reversed", cf_rev)):
            for m in ("base", "R"):
                ps = []
                for name in CF_NAMES:
                    p = np.asarray(src[flag][m][name]["p_option"])
                    ps.append(p / p.sum())
                ps.append(np.mean(ps, 0))
                for name, p in zip((*CF_NAMES, "p_cf"), ps):
                    c = int(p.argmax())
                    cf[f"{order}/{m}/{name}"] = {"p_class": np.round(p, 3).tolist(), "argmax_class": c,
                                                 "argmax_letter": letter_of(c, k, order)}

        report[flag] = {"n": n, "excluded": sorted(excl), "git": git, "class_counts_true": np.bincount(y, minlength=k).tolist(),
                        "arms": {a: dict(zip(METRICS, map(float, point[a]))) for a in point},
                        "guards": {a: {"leakage": g[a][0], "compliance": g[a][1], "broken": broken[a]} for a in g},
                        "contrasts": contrasts, "interaction_R_effect_rev_minus_frozen": inter,
                        "base_order_shift_ovo": shift, "c_threshold_met": abs(shift[0]) >= 0.02,
                        "pred_class_counts": counts, "pred_letter_counts": letters, "flips": fl, "content_free": cf}

        # ---- report
        L += [f"## {flag} (n = {n}; excluded {sorted(excl) or 'none'}; true class counts "
              f"{report[flag]['class_counts_true']})", "",
              "| arm | OvO | OvR | OvR cal | primary | primary cal | leakage | compliance | guard |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a in ("base", "R", "base_rev", "R_rev", "base_avg", "R_avg"):
            gg = report[flag]["guards"].get(a)
            gtxt = ("—", "—", "exploratory") if gg is None else (
                f"{gg['leakage']:.4f}", f"{gg['compliance']:.3f}", "BROKEN" if gg["broken"] else "ok")
            L.append(f"| {a} | " + " | ".join(f"{x:.4f}" for x in point[a]) + f" | {' | '.join(gtxt)} |")
        L += ["", "| contrast | ΔOvO (deciding) | ΔOvR | Δprimary |", "|---|---|---|---|"]
        L += [f"| {c} | {fmt(*v['ovo'])} | {fmt(*v['ovr'])} | {fmt(*v['primary'])} |" for c, v in contrasts.items()]
        L += [f"| **interaction** (R_rev − base_rev) − (R − base) | {fmt(*inter['ovo'])} | {fmt(*inter['ovr'])} | "
              f"{fmt(*inter['primary'])} (cal {fmt(*inter['primary_cal'])}) |", "",
              f"- **c:** dOvO(base_rev − base) = {fmt(*shift)}; |shift| ≥ 0.02: "
              f"{'YES' if report[flag]['c_threshold_met'] else 'no'}.",
              f"- **grounding (order-invariance of R's effect):** dOvO(R − base) {fmt(*contrasts['R − base']['ovo'])}; "
              f"dOvO(R_rev − base_rev) {fmt(*contrasts['R_rev − base_rev']['ovo'])}; interaction {fmt(*inter['ovo'])}.", "",
              "| arm | order | predicted CLASS counts (class 0..K-1) | predicted LETTER counts (A..) |", "|---|---|---|---|"]
        L += [f"| {a} | {ORDER[a]} | {counts[a]} | {letters[a]} |" for a in RUNS]
        L += ["", "Flips (report-only, class indices): " + "; ".join(
            f"{name}: {f['n']} (corrected {f['corrected']}, broken {f['broken']}, neutral {f['neutral']}, "
            f"dominant {f['dominant_share'] if f['dominant_share'] is None else round(f['dominant_share'], 2)}) "
            f"{dict(list(f['transitions'].items())[:4])}" for name, f in fl.items()), "",
            "Content-free (POST HOC): argmax as class (letter) under each order.", "",
            "| model | image | frozen: p by class | argmax | reversed: p by class | argmax |", "|---|---|---|---|---|---|"]
        for m in ("base", "R"):
            for name in (*CF_NAMES, "p_cf"):
                f, r = cf[f"frozen/{m}/{name}"], cf[f"reversed/{m}/{name}"]
                L.append(f"| {m} | {name} | {f['p_class']} | class {f['argmax_class']} ({f['argmax_letter']}) | "
                         f"{r['p_class']} | class {r['argmax_class']} ({r['argmax_letter']}) |")
        L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c4.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "c4_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    missing = [f"{d}/{f}" for d in (RUNS["base_rev"], RUNS["R_rev"]) for f in FLAGS
               if not (ROOT / d / f / "predictions.jsonl").exists()]
    if missing:
        raise SystemExit("C4 runs missing (see the module docstring for the commands): " + ", ".join(missing))
    path = OUT / "content_free_rev.json"
    cf_rev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else run_gpu()
    analyse(cf_rev)


if __name__ == "__main__":
    main()
