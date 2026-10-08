"""B1 — content-free prior and visual reliance (notes/diagnosis_plan.md §3 B1; Amendment 2).

    python scripts/diag_b1.py            # GPU: 3 content-free images x 3 datasets x {base, R}, then CPU analysis

Content-free inputs per dataset: mid-gray, black, and the pixel mean of the 300 tune images, with the
frozen prompt. p_cf = the mean of their renormalised p_option, per model. Then, on the saved tune outputs:
- reliance_i = JS(p_i || p_cf) (nats); AUC of base reliance for predicting base correctness (binary AUC);
- R's reliance minus base's, paired, per image (grounding predicts > 0 on Derma).
Contextual calibration p / p_cf (label-free) is applied to val `base` and `R` and set beside A1's `base_cal`.
It is a per-class multiplicative factor, so under OvO-LR it is the identity by construction (Amendment 2a);
it can move the primary metric and OvR only. Paired bootstrap 10,000, seed 0.
Writes results/diag/B1/{content_free.json, b1.json, b1_report.md}.
"""
import json

import numpy as np
from PIL import Image

from effvar import FLAGS, ROOT
from effvar.data import load_split, load_subset
from effvar.diag_stats import (DATASETS, align, apply_bias, binary_auc, boot_idx, ci, fmt, load_run, ovo_logratio_auc,
                               primary, ranking_auc)

OUT = ROOT / "results" / "diag" / "B1"
CF_NAMES = ("gray", "black", "tune_mean")


def content_free_images(flag):
    tune = load_subset(flag, "tune")
    ds = load_split(flag, "train")
    mean = np.mean([np.asarray(ds[i][0], dtype=np.float64) for i in tune.indices], axis=0)
    return {"gray": Image.new("RGB", (224, 224), (128, 128, 128)), "black": Image.new("RGB", (224, 224), (0, 0, 0)),
            "tune_mean": Image.fromarray(np.round(mean).astype(np.uint8))}


def run_gpu():
    from effvar.generation import run_untimed
    from effvar.methods import make_method
    from effvar.model import load_model, load_processor
    from effvar.prompts import get_spec
    from effvar.scoring import first_token_metrics, option_token_sets

    processor = load_processor()
    model = load_model("var_eager")
    methods = {"base": make_method("base", attn="var_eager"), "R": make_method("var", attn="var_eager")}
    out = {}
    for flag in FLAGS:
        spec = get_spec(flag)
        sets = option_token_sets(processor.tokenizer, spec)
        out[flag] = {}
        for name, img in content_free_images(flag).items():
            for m, method in methods.items():
                gen = run_untimed(model, processor, method, spec, img)
                ft = first_token_metrics(gen.first_logits, sets, 0)
                out[flag].setdefault(m, {})[name] = {"p_option": ft["p_option"], "leakage": ft["leakage"],
                                                     "text": gen.text}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "content_free.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * (np.log(np.clip(a, 1e-12, None)) - np.log(np.clip(b, 1e-12, None))), -1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyse(cf):
    a1 = json.loads((ROOT / "results/diag/A1/a1.json").read_text())
    report = {}
    lines = ["# B1 — content-free prior and visual reliance (P1, 2026-09-28)", "",
             "p_cf = mean renormalised p_option over gray / black / tune-mean images. Reliance = JS(p_i ‖ p_cf). "
             "Paired bootstrap 10,000, seed 0; `*` = CI excludes 0. OvO-LR decides (Amendment 2); contextual "
             "calibration leaves it unchanged by construction.", ""]
    for flag in FLAGS:
        k, ordinal = DATASETS[flag]
        letters = "ABCDEFG"[:k]
        pcf = {}
        lines += [f"## {flag}", "", "| model | image | p_option (renormalised) | argmax |", "|---|---|---|---|"]
        for m in ("base", "R"):
            ps = []
            for name in CF_NAMES:
                p = np.asarray(cf[flag][m][name]["p_option"])
                p = p / p.sum()
                ps.append(p)
                lines.append(f"| {m} | {name} | {np.round(p, 3).tolist()} | {letters[int(p.argmax())]} |")
            pcf[m] = np.mean(ps, 0)
            lines.append(f"| {m} | **p_cf** | {np.round(pcf[m], 3).tolist()} | **{letters[int(pcf[m].argmax())]}** |")

        # reliance on tune
        _, y, arr = align({"base": load_run(ROOT / "results/tune/base", flag),
                           "R": load_run(ROOT / "results/tune/var/rho0.5_summ0.1", flag)})
        rel = {m: js(arr[m]["p"], pcf[m][None, :]) for m in ("base", "R")}
        correct = arr["base"]["pred"] == y
        idx = boot_idx(len(y))
        auc0 = binary_auc(correct, rel["base"])
        auc_b = np.array([binary_auc(correct[ix], rel["base"][ix]) for ix in idx])
        d = rel["R"] - rel["base"]
        d_b = d[idx].mean(1)
        # contextual calibration on val, beside A1's base_cal
        _, vy, varr = align({"base": load_run(ROOT / "results/val/base_text_eager", flag),
                             "R": load_run(ROOT / "results/val/var_tau10_v2", flag)})
        arms = {"base": varr["base"]["p"], "R": varr["R"]["p"],
                "base_cal (A1)": apply_bias(varr["base"]["p"], np.array(a1[flag]["fits"]["base"]["bias"])),
                "base_cal_cf": varr["base"]["p"] / pcf["base"], "R_cal_cf": varr["R"]["p"] / pcf["R"]}
        arms = {a: p / p.sum(1, keepdims=True) for a, p in arms.items()}
        preds = {a: (varr[a]["pred"] if a in ("base", "R") else p.argmax(1)) for a, p in arms.items()}
        vidx = boot_idx(len(vy))
        prim = {a: primary(vy, preds[a], k, ordinal) for a in arms}
        prim_b = {a: np.array([primary(vy[ix], preds[a][ix], k, ordinal) for ix in vidx]) for a in arms}
        contr = {f"{a} − {b}": [prim[a] - prim[b], *ci(prim_b[a] - prim_b[b])]
                 for a, b in (("base_cal_cf", "base"), ("base_cal_cf", "base_cal (A1)"), ("R_cal_cf", "base_cal_cf"))}
        report[flag] = {"p_cf": {m: pcf[m].tolist() for m in pcf}, "argmax_p_cf": {m: letters[int(pcf[m].argmax())] for m in pcf},
                        "reliance_mean": {m: float(rel[m].mean()) for m in rel},
                        "auc_reliance_correct_base": [auc0, *ci(auc_b)], "d_reliance_R_minus_base": [float(d.mean()), *ci(d_b)],
                        "val_primary": prim, "val_ovr": {a: ranking_auc(vy, p, k) for a, p in arms.items()},
                        "val_ovo": {a: ovo_logratio_auc(vy, p, k) for a, p in arms.items()}, "val_contrasts": contr,
                        "val_pred_counts": {a: np.bincount(preds[a], minlength=k).tolist() for a in arms}}
        r = report[flag]
        lines += ["", f"- Reliance (tune, n = {len(y)}): base mean {r['reliance_mean']['base']:.3f}, R mean "
                      f"{r['reliance_mean']['R']:.3f}; **R − base {fmt(*r['d_reliance_R_minus_base'])}**.",
                  f"- AUC(base reliance → base correct) = {auc0:.3f} [{r['auc_reliance_correct_base'][1]:.3f}, "
                  f"{r['auc_reliance_correct_base'][2]:.3f}].", "",
                  "| val arm | primary | OvO-LR | OvR | predicted counts |", "|---|---|---|---|---|"]
        lines += [f"| {a} | {prim[a]:.4f} | {r['val_ovo'][a]:.4f} | {r['val_ovr'][a]:.4f} | {r['val_pred_counts'][a]} |"
                  for a in arms]
        lines += [""] + [f"- {c}: primary {fmt(*v)}" for c, v in contr.items()] + [""]
    (OUT / "b1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "b1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    path = OUT / "content_free.json"
    cf = json.loads(path.read_text()) if path.exists() else run_gpu()
    analyse(cf)


if __name__ == "__main__":
    main()
