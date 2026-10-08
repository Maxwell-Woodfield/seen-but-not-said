"""The test-split report (notes/validation_plan.md section 5), from the committed summary.csv files.

    python scripts/test_report.py                      # all four test arms
    python scripts/test_report.py --runs base=results/rehearsal/base_sdpa R=...

Post-hoc only: reads `<run>/summary.csv` (written by scripts/export_summary.py from the gitignored
predictions.jsonl), never a model. Every number below is reproducible from files on record.

Bootstrap: images resampled jointly across methods, 10,000 resamples, seed 0, per dataset (section 2).

Duplicates: where `data/subsets/<flag>_test_duplicates.json` lists exact-duplicate images
(scripts/find_duplicates.py), the dataset is reported BOTH at its raw n and at the effective n with
each duplicate group collapsed to its first index. PneumoniaMNIST test has 3 duplicate pairs, so
n = 624 raw and n = 621 effective.

Metrics are computed with vectorised implementations (confusion-matrix macro-F1 and QWK, rank-based
OvR macro AUC) because sklearn's per-call overhead dominates 10,000 resamples. They are checked
against sklearn on every point estimate before the bootstrap runs; a disagreement above 1e-9 aborts.

Latency here is DESCRIPTIVE ONLY: the four test runs were sequential, so their latency columns carry
run-level drift. Latency CONTRASTS come from the interleaved A/B session via scripts/ab_report.py.
Conventions: `lm_ms` = LM prefill + D x decode_ms_per_token, D = 7 (pre-registered);
`lm_run_ms` = LM prefill + (n_new - 1) x decode_ms_per_token (as run).
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {"retinamnist": (5, True), "pneumoniamnist": (2, False), "dermamnist": (7, False)}
DEFAULT_RUNS = {
    "base": "results/test/base_sdpa",
    "R": "results/test/var_tau10",
    "var_attn t15": "results/test/var_attn_tau15",
    "m24": "results/test/effvar_topm24",
}
B = 10_000
SEED = 0
LATENCY = ["vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "lm_run_ms", "e2e_ms"]
SCALARS = ["target_p", "entropy", "leakage"]


# --------------------------------------------------------------------------- metrics (vectorised)
def confusion(y, yhat, k):
    return np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)


def macro_f1(y, yhat, k):
    c = confusion(y, yhat, k)
    tp = np.diag(c).astype(float)
    denom = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)))


def qwk(y, yhat, k):
    """Quadratic weighted kappa; identical to cohen_kappa_score(weights='quadratic', labels=range(k))."""
    c = confusion(y, yhat, k).astype(float)
    i = np.arange(k)
    w = (i[:, None] - i[None, :]) ** 2
    e = np.outer(c.sum(1), c.sum(0)) / c.sum()
    den = (w * e).sum()
    return float(1.0 - (w * c).sum() / den) if den > 0 else 0.0


def _binary_auc(pos, score):
    """Mann-Whitney AUC with average ranks for ties; NaN if one class is absent."""
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    _, inv, counts = np.unique(score[order], return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = (starts + (counts + 1) / 2.0)[inv]          # 1-based average ranks
    return float((ranks[pos[order]].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranking_auc(y, p, k):
    """Ranking AUC from p_option renormalised over the options (validation_plan.md section 4):
    binary AUC for 2 classes, else the macro average of the one-vs-rest AUCs."""
    if k == 2:
        return _binary_auc(y == 1, p[:, 1])
    return float(np.mean([_binary_auc(y == c, p[:, c]) for c in range(k)]))


def check_against_sklearn(flag, y, yhat, p):
    """Abort unless the vectorised metrics match sklearn on this point estimate."""
    k, ordinal = DATASETS[flag]
    pairs = [("macro_f1", macro_f1(y, yhat, k),
              f1_score(y, yhat, labels=list(range(k)), average="macro", zero_division=0)),
             ("auc", ranking_auc(y, p, k),
              roc_auc_score(y, p[:, 1]) if k == 2 else
              roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(k))))]
    if ordinal:
        pairs.append(("qwk", qwk(y, yhat, k),
                      cohen_kappa_score(y, yhat, labels=list(range(k)), weights="quadratic")))
    for name, fast, ref in pairs:
        if np.isnan(fast) and np.isnan(ref):  # a class with no positives; both agree it is undefined
            continue
        if not (abs(fast - ref) < 1e-9):
            raise SystemExit(f"{flag}: {name} fast {fast!r} != sklearn {ref!r}")


# --------------------------------------------------------------------------- data
def load_summary(run: Path) -> dict[str, list[dict]]:
    by_flag: dict[str, list[dict]] = {f: [] for f in DATASETS}
    with (run / "summary.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for k in ("index", "label", "pred", "correct", "n_new", "first_token_in_options"):
                row[k] = int(r[k])
            row["p_option"] = np.array(json.loads(r["p_option"]))
            for k in SCALARS + ["vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "e2e_ms"]:
                row[k] = float(r[k]) if r[k] != "" else np.nan
            row["lm_ms"] = row["lm_prefill_ms"] + 7 * row["decode_ms_per_token"]
            row["lm_run_ms"] = row["lm_prefill_ms"] + (row["n_new"] - 1) * row["decode_ms_per_token"]
            by_flag[row["dataset"]].append(row)
    return {f: sorted(v, key=lambda r: r["index"]) for f, v in by_flag.items() if v}


def duplicates_dropped(flag: str) -> set[int]:
    path = ROOT / "data" / "subsets" / f"{flag}_test_duplicates.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text())["keep_first_drop"])


# --------------------------------------------------------------------------- bootstrap
def arrays_for(runs, flag, drop: set[int]):
    ref = next(iter(runs.values()))[flag]
    keep = [i for i, r in enumerate(ref) if r["index"] not in drop]
    idx = [ref[i]["index"] for i in keep]
    out = {}
    for name, r in runs.items():
        rows = [r[flag][i] for i in keep]
        assert [x["index"] for x in rows] == idx, f"{flag}: image set differs in {name}"
        out[name] = {
            "yhat": np.array([x["pred"] for x in rows]),
            "p": np.array([x["p_option"] / x["p_option"].sum() for x in rows]),
            "correct": np.array([x["correct"] for x in rows]),
            **{k: np.array([x[k] for x in rows]) for k in SCALARS + LATENCY},
        }
    y = np.array([ref[i]["label"] for i in keep])
    return y, out


def evaluate(flag, y, a, k, ordinal):
    m = {"accuracy": float(np.mean(y == a["yhat"])),
         "macro_f1": macro_f1(y, a["yhat"], k)}
    if ordinal:
        m["qwk"] = qwk(y, a["yhat"], k)
    m["auc"] = ranking_auc(y, a["p"], k)
    for key in SCALARS + LATENCY:
        m[key] = float(np.nanmean(a[key]))
    return m


def compute(runs, flag, drop: set[int]):
    k, ordinal = DATASETS[flag]
    y, arr = arrays_for(runs, flag, drop)
    for name, a in arr.items():
        check_against_sklearn(flag, y, a["yhat"], a["p"])
    point = {n: evaluate(flag, y, a, k, ordinal) for n, a in arr.items()}
    rng = np.random.default_rng(SEED)
    boot = {n: {m: np.empty(B) for m in point[n]} for n in arr}
    for b in range(B):
        i = rng.integers(0, len(y), len(y))
        yb = y[i]
        for n, a in arr.items():
            ab = {key: a[key][i] for key in a}
            for m, v in evaluate(flag, yb, ab, k, ordinal).items():
                boot[n][m][b] = v
    return len(y), point, boot


# --------------------------------------------------------------------------- formatting
def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(pt, lo, hi, dec=4):
    mark = "*" if lo > 0 or hi < 0 else ""
    return f"{pt:+.{dec}f} [{lo:+.{dec}f},{hi:+.{dec}f}]{mark}"


def flips(runs, flag, base, name, drop: set[int]):
    b = [r for r in runs[base][flag] if r["index"] not in drop]
    v = [r for r in runs[name][flag] if r["index"] not in drop]
    trans, cor, bro, neu = {}, 0, 0, 0
    for rb, rv in zip(b, v):
        if rb["pred"] == rv["pred"]:
            continue
        trans[f"{rb['pred_letter']}->{rv['pred_letter']}"] = trans.get(f"{rb['pred_letter']}->{rv['pred_letter']}", 0) + 1
        if rv["correct"] and not rb["correct"]:
            cor += 1
        elif rb["correct"] and not rv["correct"]:
            bro += 1
        else:
            neu += 1
    return trans, cor, bro, neu


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, help="name=path pairs; default = the four test arms")
    ap.add_argument("--out", default="results/test/report.md")
    args = ap.parse_args()

    spec = DEFAULT_RUNS if args.runs is None else dict(p.split("=", 1) for p in args.runs)
    runs = {name: load_summary(ROOT / path) for name, path in spec.items()}
    names = list(runs)
    base = names[0]
    L: list[str] = []

    def emit(s=""):
        print(s)
        L.append(s)

    emit("# Test-split report")
    emit("")
    emit("Runs: " + ", ".join(f"`{n}` = `{p}`" for n, p in spec.items()))
    emit(f"Paired bootstrap: {B} resamples, seed {SEED}, images resampled jointly across methods.")
    emit(f"`*` marks a 95% CI excluding 0. Deltas are against `{base}`.")
    emit("")

    for flag in DATASETS:
        if flag not in runs[base]:
            continue
        k, ordinal = DATASETS[flag]
        drop = duplicates_dropped(flag)
        variants = [("raw", set())] + ([("duplicates collapsed", drop)] if drop else [])
        for tag, d in variants:
            n, point, boot = compute(runs, flag, d)
            emit(f"## {flag} -- n = {n} ({tag}), primary = {'QWK' if ordinal else 'macro-F1'}")
            emit("")
            metrics = [m for m in point[base] if m not in LATENCY]
            emit("| metric | " + " | ".join(f"`{x}`" for x in names) + " |")
            emit("|---|" + "---|" * len(names))
            for m in metrics:
                cells = [f"{point[base][m]:.4f}"]
                for x in names[1:]:
                    lo, hi = ci(boot[x][m] - boot[base][m])
                    cells.append(f"{point[x][m]:.4f} ({fmt(point[x][m] - point[base][m], lo, hi)})")
                emit(f"| {m} | " + " | ".join(cells) + " |")
            emit("")
            emit("Latency ms/image -- DESCRIPTIVE ONLY (sequential runs carry run-level drift; "
                 "contrasts come from the interleaved A/B session):")
            emit("")
            emit("| metric | " + " | ".join(f"`{x}`" for x in names) + " |")
            emit("|---|" + "---|" * len(names))
            for m in LATENCY:
                cells = [f"{point[base][m]:.1f}"]
                for x in names[1:]:
                    lo, hi = ci(boot[x][m] - boot[base][m])
                    cells.append(f"{point[x][m]:.1f} ({fmt(point[x][m] - point[base][m], lo, hi, 1)})")
                emit(f"| {m} | " + " | ".join(cells) + " |")
            emit("")
            emit("Flip-direction counts vs `" + base + "`:")
            emit("")
            emit("| run | flips | corrected | broken | neutral | transitions |")
            emit("|---|---|---|---|---|---|")
            for x in names[1:]:
                trans, cor, bro, neu = flips(runs, flag, base, x, d)
                top = ", ".join(f"{a} {c}" for a, c in sorted(trans.items(), key=lambda t: -t[1]))
                emit(f"| `{x}` | {sum(trans.values())} | {cor} | {bro} | {neu} | {top or '-'} |")
            emit("")

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
