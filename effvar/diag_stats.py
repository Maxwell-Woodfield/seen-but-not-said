"""Phase 6 statistics on saved outputs (notes/diagnosis_plan.md §2, A1, A2). CPU only, no model.

- Vectorised metrics, the same definitions as scripts/test_report.py (confusion-matrix macro-F1 and QWK,
  rank-based OvR macro AUC on `p_option` renormalised over the options); checked against sklearn in tests.
- A1 calibrations, each fitted on tune and applied unchanged elsewhere:
  - `fit_bias`: per-class additive offsets on log p (class 0 fixed at 0), minimising NLL -- THE base_cal;
  - `fit_decision`: offsets (argmax) maximising macro-F1, or for an ordinal task cut-points on the
    expected grade maximising QWK -- the decision-tuned sensitivity form.
- A2 shift decomposition of a variant against `base`, per image on centred log-probabilities.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATASETS = {"retinamnist": (5, True), "pneumoniamnist": (2, False), "dermamnist": (7, False)}
EPS = 1e-12


# --------------------------------------------------------------------------- data
def load_run(run_dir: Path, flag: str) -> dict[int, dict]:
    """index -> {label, pred, p (renormalised over the options)} from a run's predictions.jsonl."""
    out = {}
    with open(Path(run_dir) / flag / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = np.asarray(r["p_option"], dtype=float)
            out[int(r["index"])] = {"label": int(r["label"]), "pred": int(r["pred"]), "p": p / p.sum()}
    return out


def align(runs: dict[str, dict[int, dict]], exclude: set[int] = frozenset()):
    """Common image order across runs: (indices, y, {name: {"pred": [n], "p": [n, K]}})."""
    idx = sorted(set.intersection(*(set(r) for r in runs.values())) - set(exclude))
    first = next(iter(runs.values()))
    y = np.array([first[i]["label"] for i in idx])
    out = {}
    for name, r in runs.items():
        assert all(r[i]["label"] == first[i]["label"] for i in idx), name
        out[name] = {"pred": np.array([r[i]["pred"] for i in idx]), "p": np.stack([r[i]["p"] for i in idx])}
    return np.array(idx), y, out


# --------------------------------------------------------------------------- metrics
def confusion(y, yhat, k):
    return np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)


def macro_f1(y, yhat, k):
    c = confusion(y, yhat, k)
    tp = np.diag(c).astype(float)
    denom = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)))


def qwk(y, yhat, k):
    c = confusion(y, yhat, k).astype(float)
    i = np.arange(k)
    w = (i[:, None] - i[None, :]) ** 2
    e = np.outer(c.sum(1), c.sum(0)) / c.sum()
    den = (w * e).sum()
    return float(1.0 - (w * c).sum() / den) if den > 0 else 0.0


def binary_auc(pos, score):
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    _, inv, counts = np.unique(score[order], return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = (starts + (counts + 1) / 2.0)[inv]
    return float((ranks[pos[order]].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranking_auc(y, p, k):
    if k == 2:
        return binary_auc(y == 1, p[:, 1])
    return float(np.nanmean([binary_auc(y == c, p[:, c]) for c in range(k)]))


def ovo_logratio_auc(y, p, k):
    """Macro one-vs-one AUC on log-ratios: for each class pair (c, j), rank the images of classes c and j
    by log p_c - log p_j. A per-class additive offset on log p shifts every score of a pair by the same
    constant, so this metric is EXACTLY invariant to prior correction (fit_bias); for 2 classes it is the
    ordinary AUC. Exploratory (P0 finding: the OvR AUC above is not calibration-invariant for K > 2)."""
    z = np.log(np.clip(p, EPS, None))
    aucs = []
    for c in range(k):
        for j in range(c + 1, k):
            m = (y == c) | (y == j)
            aucs.append(binary_auc(y[m] == c, z[m, c] - z[m, j]))
    return float(np.nanmean(aucs))


def primary(y, yhat, k, ordinal):
    return qwk(y, yhat, k) if ordinal else macro_f1(y, yhat, k)


# --------------------------------------------------------------------------- A1 calibration
def fit_bias(p, y, k) -> np.ndarray:
    """Offsets b (b[0] = 0) minimising mean NLL of softmax(log p + b): a prior correction, convex."""
    z = np.log(np.clip(p, EPS, None))
    onehot = np.eye(k)[y]

    def f(b_free):
        b = np.concatenate(([0.0], b_free))
        s = z + b
        s = s - s.max(1, keepdims=True)
        q = np.exp(s)
        q /= q.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip((q * onehot).sum(1), EPS, None)))
        return nll, (q - onehot).mean(0)[1:]

    res = minimize(f, np.zeros(k - 1), jac=True, method="L-BFGS-B")
    return np.concatenate(([0.0], res.x))


def apply_bias(p, b):
    s = np.log(np.clip(p, EPS, None)) + b
    s = s - s.max(1, keepdims=True)
    q = np.exp(s)
    return q / q.sum(1, keepdims=True)


GRID = np.linspace(-8.0, 8.0, 321)


def fit_decision(p, y, k, ordinal, rounds: int = 6):
    """Decision-tuned rule. Nominal: argmax(log p + b) with b maximising macro-F1 by coordinate ascent
    from the NLL offsets. Ordinal: cut-points on E[grade] = sum_g g p_g maximising QWK. Ties keep the
    current value, so the rule moves only on a strict improvement."""
    if ordinal:
        e = p @ np.arange(k)
        cand = np.unique(np.concatenate([np.arange(k - 1) + 0.5, (np.sort(e)[1:] + np.sort(e)[:-1]) / 2]))
        t = np.arange(k - 1) + 0.5
        best = qwk(y, np.searchsorted(t, e), k)
        for _ in range(rounds):
            moved = False
            for j in range(k - 1):
                lo = t[j - 1] if j else -np.inf
                hi = t[j + 1] if j < k - 2 else np.inf
                for c in cand[(cand > lo) & (cand < hi)]:
                    tt = t.copy()
                    tt[j] = c
                    v = qwk(y, np.searchsorted(tt, e), k)
                    if v > best + 1e-12:
                        best, t, moved = v, tt, True
            if not moved:
                break
        return {"kind": "cutpoints", "t": t}
    b = fit_bias(p, y, k)
    z = np.log(np.clip(p, EPS, None))
    best = macro_f1(y, np.argmax(z + b, 1), k)
    for _ in range(rounds):
        moved = False
        for j in range(1, k):
            for c in b[j] + GRID:
                bb = b.copy()
                bb[j] = c
                v = macro_f1(y, np.argmax(z + bb, 1), k)
                if v > best + 1e-12:
                    best, b, moved = v, bb, True
        if not moved:
            break
    return {"kind": "offsets", "b": b}


def apply_decision(rule, p, k):
    if rule["kind"] == "cutpoints":
        return np.searchsorted(rule["t"], p @ np.arange(k))
    return np.argmax(np.log(np.clip(p, EPS, None)) + rule["b"], 1)


# --------------------------------------------------------------------------- A2 shift decomposition
def centred_logp(p):
    z = np.log(np.clip(p, EPS, None))
    return z - z.mean(1, keepdims=True)


def shift_stats(delta, y, idx=None):
    """(CM share, alignment) of per-image shifts `delta` [n, K] (centred over classes).

    CM share = ||mean_i delta_i||^2 / mean_i ||delta_i||^2 -- 1 means every image moved identically.
    Alignment = mean_i r_i[y_i], r = delta - mean delta -- the image-specific part's push toward the true
    class (log-prob units). `idx` [B, n] evaluates both on bootstrap resamples at once.
    """
    if idx is None:
        idx = np.arange(len(y))[None, :]
    sq = (delta ** 2).sum(-1)                               # [n]
    dy = np.take_along_axis(delta, y[:, None], 1)[:, 0]     # [n]
    onehot = np.eye(delta.shape[1])[y]                      # [n, K]
    cms, aligns = [], []
    for s in range(0, len(idx), 500):                       # chunks keep [B, n, K] small
        ix = idx[s:s + 500]
        m = delta[ix].mean(1)                               # [b, K]
        cms.append((m ** 2).sum(-1) / sq[ix].mean(1))
        aligns.append(dy[ix].mean(1) - (onehot[ix].mean(1) * m).sum(-1))
    return np.concatenate(cms), np.concatenate(aligns)


def boot_idx(n, B=10_000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(pt, lo, hi, dec=4):
    mark = "*" if lo > 0 or hi < 0 else ""
    return f"{pt:+.{dec}f} [{lo:+.{dec}f},{hi:+.{dec}f}]{mark}"
