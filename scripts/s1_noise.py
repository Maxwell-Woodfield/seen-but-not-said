"""Measurement-noise check behind the S1 promotion-screen thresholds (notes/gates.md). READ-ONLY.

Reads cached tune predictions only (`base`, `R` = var tau10 rho0.5 summ0.1, and the H1 m-curve variants
already evaluated in results/tune/topm/); runs no model. For each quantity S1 thresholds, it prints the
point estimate and a paired-bootstrap 95% CI (images resampled jointly across methods, per dataset):

- `R`'s gain over `base` per dataset, mean-of-3, and Derma ranking-AUC gain -- the gain-kept denominators;
- gain kept per dataset, mean-of-3 and on Derma dAUC, for each m-curve variant;
- variant - base per dataset (the "not worse than base" floor) and Derma dAUC vs base and vs `R`.

These numbers calibrated S1 on 2026-09-20: per-dataset gain kept is unresolvable at n = 300 wherever
`R`'s own gain CI straddles 0, while Derma dAUC kept resolves and rejects the random-layer control.
2,000 resamples, seed 0 (H1 itself uses 10,000); a few Derma resamples miss a class and give a NaN AUC,
which the percentiles skip.

    python scripts/s1_noise.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "results" / "tune"
DATASETS = {"retinamnist": (5, True), "pneumoniamnist": (2, False), "dermamnist": (7, False)}
VARIANTS = ["full_m24", "both_m16", "both_m12", "raw_m8", "control_m8", "both_m4"]
RUNS = {"base": TUNE / "base", "R": TUNE / "var" / "rho0.5_summ0.1"} | {v: TUNE / "topm" / v for v in VARIANTS}
B = 2000
SEED = 0


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in open(path / "predictions.jsonl", encoding="utf-8")]
    return sorted(rows, key=lambda r: r["index"])


def primary(flag: str, y, yhat) -> float:
    k, ordinal = DATASETS[flag]
    if ordinal:
        return cohen_kappa_score(y, yhat, labels=list(range(k)), weights="quadratic")
    return f1_score(y, yhat, labels=list(range(k)), average="macro", zero_division=0)


def auc(flag: str, y, p) -> float:
    k = DATASETS[flag][0]
    if k == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(k)))


def ci(a) -> np.ndarray:
    return np.nanpercentile(a, [2.5, 97.5])


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    rng = np.random.default_rng(SEED)
    boot, point, pauc = {}, {}, {}
    for flag in DATASETS:
        rows = {k: load(p / flag) for k, p in RUNS.items()}
        idx = [r["index"] for r in rows["base"]]
        assert all([r["index"] for r in v] == idx for v in rows.values()), f"{flag}: image sets differ"
        y = np.array([r["label"] for r in rows["base"]])
        preds = {k: np.array([r["pred"] for r in v]) for k, v in rows.items()}
        # Ranking AUC from p_option renormalised over the options (validation_plan.md §4).
        probs = {k: np.array([np.array(r["p_option"]) / sum(r["p_option"]) for r in v]) for k, v in rows.items()}
        draws = [rng.integers(0, len(y), len(y)) for _ in range(B)]
        boot[flag] = {k: np.array([primary(flag, y[i], preds[k][i]) for i in draws]) for k in RUNS}
        point[flag] = {k: primary(flag, y, preds[k]) for k in RUNS}
        if flag == "dermamnist":
            boot["auc"] = {k: np.array([auc(flag, y[i], probs[k][i]) for i in draws]) for k in RUNS}
            pauc = {k: auc(flag, y, probs[k]) for k in RUNS}

    mean_b = {k: np.mean([boot[f][k] for f in DATASETS], axis=0) for k in RUNS}
    mean_p = {k: np.mean([point[f][k] for f in DATASETS]) for k in RUNS}

    print(f"R gain over base (tune, n per dataset as saved; {B} resamples, seed {SEED}):")
    for flag in DATASETS:
        d = boot[flag]["R"] - boot[flag]["base"]
        lo, hi = ci(d)
        print(f"  {flag:15s} {point[flag]['R'] - point[flag]['base']:+.4f} CI [{lo:+.4f},{hi:+.4f}]"
              f"  P(boot gain <= 0) = {np.mean(d <= 0):.2f}")
    lo, hi = ci(mean_b["R"] - mean_b["base"])
    print(f"  {'mean-of-3':15s} {mean_p['R'] - mean_p['base']:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    lo, hi = ci(boot["auc"]["R"] - boot["auc"]["base"])
    print(f"  {'derma dAUC':15s} {pauc['R'] - pauc['base']:+.4f} CI [{lo:+.4f},{hi:+.4f}]")

    def kept(pt, b):
        """Point estimate and CI of (v - base) / (R - base), for one variant's point dict and boot dict."""
        return pt, ci(b)

    print("\nGain kept, point [95% CI]: per dataset | mean-of-3 | Derma dAUC")
    for v in VARIANTS:
        cells = []
        for flag in DATASETS:
            p, b = point[flag], boot[flag]
            pt, (lo, hi) = kept((p[v] - p["base"]) / (p["R"] - p["base"]), (b[v] - b["base"]) / (b["R"] - b["base"]))
            cells.append(f"{flag[:4]} {pt:5.0%} [{lo:6.0%},{hi:6.0%}]")
        pt, (lo, hi) = kept((mean_p[v] - mean_p["base"]) / (mean_p["R"] - mean_p["base"]),
                            (mean_b[v] - mean_b["base"]) / (mean_b["R"] - mean_b["base"]))
        cells.append(f"mean {pt:5.0%} [{lo:6.0%},{hi:6.0%}]")
        a = boot["auc"]
        pt, (lo, hi) = kept((pauc[v] - pauc["base"]) / (pauc["R"] - pauc["base"]), (a[v] - a["base"]) / (a["R"] - a["base"]))
        cells.append(f"dAUC {pt:5.0%} [{lo:6.0%},{hi:6.0%}]")
        print(f"  {v:11s} " + " | ".join(cells))
        floor = []
        for flag in DATASETS:
            lo, hi = ci(boot[flag][v] - boot[flag]["base"])
            floor.append(f"{flag[:4]} v-base {point[flag][v] - point[flag]['base']:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print("              " + " | ".join(floor))

    print(f"\nDerma ranking AUC (NaN share among resamples: {np.mean(np.isnan(boot['auc']['R'])):.4f})")
    a = boot["auc"]
    for v in ["R"] + VARIANTS:
        (lo, hi), (lo_r, hi_r) = ci(a[v] - a["base"]), ci(a[v] - a["R"])
        print(f"  {v:11s} dAUC vs base {pauc[v] - pauc['base']:+.4f} CI [{lo:+.4f},{hi:+.4f}]"
              f"  vs R {pauc[v] - pauc['R']:+.4f} CI [{lo_r:+.4f},{hi_r:+.4f}]")


if __name__ == "__main__":
    main()
