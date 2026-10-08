"""Model-free checks of effvar/diag_stats.py (A1 calibration, A2 shift decomposition, metrics)."""
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from effvar.diag_stats import (apply_bias, apply_decision, boot_idx, centred_logp, fit_bias, fit_decision, macro_f1,
                               ovo_logratio_auc, qwk, ranking_auc, shift_stats)

rng = np.random.default_rng(0)


def fake(n=600, k=4, shift=None, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    logits = r.normal(size=(n, k)) + 2.0 * np.eye(k)[y]
    if shift is not None:
        logits = logits + shift
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), y


@pytest.mark.parametrize("k", [2, 5, 7])
def test_metrics_match_sklearn(k):
    p, y = fake(k=k)
    yhat = p.argmax(1)
    assert abs(macro_f1(y, yhat, k) - f1_score(y, yhat, labels=list(range(k)), average="macro")) < 1e-12
    assert abs(qwk(y, yhat, k) - cohen_kappa_score(y, yhat, labels=list(range(k)), weights="quadratic")) < 1e-12
    ref = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
    assert abs(ranking_auc(y, p, k) - ref) < 1e-12


def test_fit_bias_undoes_a_prior_shift():
    shift = np.array([3.0, 0.0, -1.0, 0.0])        # class 0 massively over-predicted
    p, y = fake(shift=shift)
    b = fit_bias(p, y, 4)
    assert b[0] == 0.0
    # the fitted offsets cancel the shift up to a constant
    assert np.allclose(b - b.mean(), -(shift - shift.mean()), atol=0.35)
    before = macro_f1(y, p.argmax(1), 4)
    after = macro_f1(y, apply_bias(p, b).argmax(1), 4)
    assert after > before + 0.1
    assert np.allclose(apply_bias(p, b).sum(1), 1.0)


def test_binary_bias_leaves_auc_unchanged():
    p, y = fake(k=2, shift=np.array([2.0, 0.0]))
    q = apply_bias(p, fit_bias(p, y, 2))
    assert abs(ranking_auc(y, p, 2) - ranking_auc(y, q, 2)) < 1e-12


def test_ovo_logratio_auc_is_invariant_to_prior_correction_while_ovr_is_not():
    p, y = fake(k=5, shift=np.array([2.5, 0.0, -0.5, 0.0, 1.0]))
    q = apply_bias(p, fit_bias(p, y, 5))
    assert abs(ovo_logratio_auc(y, p, 5) - ovo_logratio_auc(y, q, 5)) < 1e-12
    assert abs(ranking_auc(y, p, 5) - ranking_auc(y, q, 5)) > 1e-4   # OvR on softmax moves with the prior
    p2, y2 = fake(k=2)
    assert abs(ovo_logratio_auc(y2, p2, 2) - ranking_auc(y2, p2, 2)) < 1e-12


def test_decision_tuned_is_at_least_as_good_on_its_fit_set():
    p, y = fake(shift=np.array([2.0, 0.0, 0.0, 0.0]))
    b = fit_bias(p, y, 4)
    rule = fit_decision(p, y, 4, ordinal=False)
    assert macro_f1(y, apply_decision(rule, p, 4), 4) >= macro_f1(y, apply_bias(p, b).argmax(1), 4)
    p5, y5 = fake(k=5)
    cut = fit_decision(p5, y5, 5, ordinal=True)
    assert np.all(np.diff(cut["t"]) > 0)
    assert qwk(y5, apply_decision(cut, p5, 5), 5) >= qwk(y5, np.searchsorted(np.arange(4) + 0.5, p5 @ np.arange(5)), 5)


def test_shift_stats_separates_common_mode_from_image_specific():
    n, k = 400, 3
    y = rng.integers(0, k, n)
    const = np.tile(np.array([1.0, -0.5, -0.5]), (n, 1))
    cm, al = shift_stats(const, y)
    assert abs(cm[0] - 1.0) < 1e-12 and abs(al[0]) < 1e-12
    toward = np.eye(k)[y] - 1.0 / k                       # every image pushed toward its own class
    cm, al = shift_stats(toward, y)
    assert cm[0] < 0.1 and al[0] > 0.5
    away = -toward
    assert shift_stats(away, y)[1][0] < -0.5


def test_shift_stats_bootstrap_shapes_and_centring():
    p, y = fake(k=3)
    d = centred_logp(p) - centred_logp(np.roll(p, 1, axis=0))
    assert np.allclose(d.sum(1), 0.0)
    cm, al = shift_stats(d, y, boot_idx(len(y), B=1200))
    assert cm.shape == al.shape == (1200,) and np.all((cm >= 0) & (cm <= 1))
