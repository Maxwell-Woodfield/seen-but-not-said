"""Phase 7 card V1 (notes/vcd_plan.md, Amendment V1-1): noise, seed, input path, readout, option-level VCD, the noised
cache, the run layout and presets, the guard checkers and the analysis gate. No model and no GPU; the processor and
tokenizer run on CPU."""
import importlib.util
import inspect
import json
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from effvar import FLAGS
from effvar.diag_stats import EPS, apply_bias, fit_bias, ovo_logratio_auc
from effvar.model import build_inputs, load_processor
from effvar.prompts import get_spec
from effvar.scoring import first_token_metrics, option_token_sets
from effvar.vcd import (ALPHAS, MISMATCH_ROWS, SMOKE_EXP, V1_EXP, SubRun, a1_folds, add_noise, boot_p,
                        build_noised_inputs, check_registered_ids, code_commit_from_index, decide, draw_eps,
                        fold_sha1, g4_specs, holm, lex_score, log_clean, log_noised, noise_coefficients,
                        ovo_scores_auc, per_token_score, preset, readout, registered_sets, renorm_rows, renyi,
                        selection, smoke_specs, softmax, split_logw, subrun_record, v_head, validate_subrun, vcd_score,
                        vcd_seed)
from effvar.vcd_cache import (CleanFeatureProbe, NoisedVisionCache, check_clean_dir, clean_cache_for_key, clean_file,
                              dir_state, key_digest, noised_key)
from effvar.vision_cache import CACHE_ATTR, VisionCache, _VisionOutput

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
V = 240_000          # > every option id, so synthetic logits can address them
TREES = {"effvar": "t-effvar", "scripts": "t-scripts", "tests": "t-tests"}
CLEAN_GIT = {"commit": "c0de", "dirty": False, "code_changes": [], "code_trees": TREES}
# Synthetic tune subsets of 300, like the real ones, holding the registered rows where the real subsets have them
# (Retina 258 beyond the first 50, Derma 209 within them), so the registered counts (900, 150, 905) apply.
SUBSETS = {"retinamnist": list(range(300)),
           "pneumoniamnist": sorted(set(range(299)) | {4328}),
           "dermamnist": sorted(set(range(49)) | {209} | set(range(400, 647)) | {2086, 2353, 6009})}
assert all(len(v) == 300 for v in SUBSETS.values()) and SUBSETS["dermamnist"].index(209) == 49


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.fixture(scope="module")
def run_vcd():
    return _script("run_vcd")


@pytest.fixture(scope="module")
def check():
    return _script("check_vcd_regression")


@pytest.fixture(scope="module")
def diag():
    return _script("diag_vcd")


# --------------------------------------------------------------------------- noise schedule and seed (guard 2)
def test_noise_coefficients_match_the_card():
    c500, c999 = noise_coefficients(500), noise_coefficients(999)
    assert abs(c500[0] - 0.863) < 1e-3 and abs(c500[1] - 0.505) < 1e-3
    assert abs(c999[0] - 0.285) < 1e-3 and abs(c999[1] - 0.959) < 1e-3
    for (c1, c2), exact in ((c500, (0.86302, 0.50517)), (c999, (0.28504, 0.95851))):   # the card's 5-digit values
        assert abs(c1 - exact[0]) < 1e-5 and abs(c2 - exact[1]) < 1e-5
        assert abs(c1 ** 2 + c2 ** 2 - 1) < 1e-6
    betas = 1 / (1 + np.exp(-np.linspace(-6, 6, 1000))) * (5e-3 - 1e-5) + 1e-5          # float64 re-derivation
    abar = np.cumprod(1 - betas)
    for t in (500, 999):
        assert abs(noise_coefficients(t)[0] - np.sqrt(abar[t])) < 1e-6
        assert abs(noise_coefficients(t)[1] - np.sqrt(1 - abar[t])) < 1e-6
    with pytest.raises(ValueError):
        noise_coefficients(1000)


def test_seed_formula_on_a_fixed_example():
    assert vcd_seed("retinamnist", "train", 1, 1) == 2404348969288661602
    assert vcd_seed("retinamnist", "train", 1, 2) == 795034973876479703
    assert vcd_seed("retinamnist", "train", np.int64(1), 1) == 2404348969288661602
    assert 0 <= vcd_seed("dermamnist", "train", 6009, 2) < 2 ** 63


def test_eps_same_seed_same_draw_different_index_or_rep_differs():
    shape = (1, 3, 32, 32)
    s = vcd_seed("dermamnist", "train", 209, 1)
    assert torch.equal(draw_eps(s, shape), draw_eps(s, shape))
    assert draw_eps(s, shape).dtype == torch.float32
    assert not torch.equal(draw_eps(s, shape), draw_eps(vcd_seed("dermamnist", "train", 210, 1), shape))
    assert not torch.equal(draw_eps(s, shape), draw_eps(vcd_seed("dermamnist", "train", 209, 2), shape))


def test_eps_does_not_depend_on_the_step():
    assert "step" not in inspect.signature(draw_eps).parameters
    assert "step" not in inspect.signature(vcd_seed).parameters
    x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    eps = draw_eps(vcd_seed("retinamnist", "train", 5, 1), x.shape)
    rec = []
    for t in (500, 999):
        c1, c2 = noise_coefficients(t)
        rec.append((add_noise(x, c1, c2, eps) - c1 * x) / c2)     # both strengths recover the same eps
    assert torch.allclose(rec[0], eps, atol=1e-5) and torch.allclose(rec[1], eps, atol=1e-5)


# --------------------------------------------------------------------------- input path
def test_identity_coefficients_leave_pixel_values_bitwise_unchanged():
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 3
    x[0, 0, 0, :4] = torch.tensor([0.0, 1.0, -1.0, 1e-30])
    eps = draw_eps(123, x.shape)
    out = add_noise(x, 1.0, 0.0, eps)
    assert torch.equal(out.view(torch.int32), x.view(torch.int32))
    assert not torch.equal(add_noise(x, *noise_coefficients(500), eps), x)


def test_add_noise_refuses_non_fp32_or_mismatched_eps():
    x = torch.zeros(1, 3, 4, 4)
    with pytest.raises(TypeError):
        add_noise(x.to(torch.bfloat16), 1.0, 0.0, torch.zeros(1, 3, 4, 4))
    with pytest.raises(TypeError):
        add_noise(x, 1.0, 0.0, torch.zeros(1, 3, 4, 5))


def _images():
    grad = np.stack(np.meshgrid(np.arange(224), np.arange(224)), -1).sum(-1) % 256
    return [Image.new("RGB", (224, 224), (128, 128, 128)),
            Image.fromarray(np.stack([grad, 255 - grad, grad // 2], -1).astype(np.uint8))]


def test_noised_inputs_at_identity_equal_build_inputs_bitwise(processor):
    spec = get_spec("retinamnist")
    seed = vcd_seed("retinamnist", "train", 1, 1)
    for image in _images():
        ref = build_inputs(processor, spec.messages(image), "cpu")
        got = build_noised_inputs(processor, spec.messages(image), "cpu", 1.0, 0.0, seed)
        assert set(ref.keys()) == set(got.keys())
        for key in ref:
            a, b = ref[key], got[key]
            assert a.dtype == b.dtype and a.shape == b.shape, key
            if a.dtype == torch.bfloat16:
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), key
            else:
                assert torch.equal(a, b), key
        noised = build_noised_inputs(processor, spec.messages(image), "cpu", *noise_coefficients(500), seed)
        assert torch.equal(noised["input_ids"], ref["input_ids"])
        assert noised["pixel_values"].dtype == torch.bfloat16
        assert not torch.equal(noised["pixel_values"], ref["pixel_values"])


# --------------------------------------------------------------------------- ids and readout (guard 5, change 1)
@pytest.mark.parametrize("flag", FLAGS)
def test_option_ids_are_the_registered_ones(processor, flag):
    spec = get_spec(flag)
    sets = option_token_sets(processor.tokenizer, spec)
    assert sets == registered_sets(spec)
    check_registered_ids(sets, spec)
    with pytest.raises(SystemExit):
        check_registered_ids([s[::-1] for s in sets[:-1]] + [[1, 2]], spec)


def test_readout_matches_first_token_metrics():
    sets = registered_sets(get_spec("dermamnist"))
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        logits = torch.randn(V, generator=g) * 3
        logits[[i for s in sets for i in s]] += torch.randn(14, generator=g) * 4 + 6
        ft = first_token_metrics(logits, sets, 0)
        ro = readout(logits, sets)
        assert len(ro["p_ids"]) == len(sets) and all(len(s) == 2 for s in ro["p_ids"])
        for c, s in enumerate(sets):
            p = torch.softmax(logits.float(), -1)
            assert ro["p_ids"][c] == pytest.approx([float(p[i]) for i in s], rel=1e-5)
            assert sum(ro["p_ids"][c]) == pytest.approx(ft["p_option"][c], rel=1e-5)
            assert np.exp(ro["logp_option"][c]) == pytest.approx(ft["p_option"][c], rel=1e-5)
            assert np.exp(ro["logp_ids"][c]) == pytest.approx(ro["p_ids"][c], rel=1e-5)


def test_noised_side_is_never_clipped_where_p_underflows():
    sets = registered_sets(get_spec("pneumoniamnist"))
    logits = torch.zeros(V)
    logits[7] = 200.0                      # all mass elsewhere: every option probability underflows to 0 in fp32
    logits[sets[0]] = torch.tensor([-100.0, -100.0])
    logits[sets[1]] = torch.tensor([-50.0, -60.0])
    ro = readout(logits, sets)
    assert all(p == 0.0 for s in ro["p_ids"] for p in s)
    assert all(np.isfinite(ro["logp_option"]))
    assert ro["logp_option"][0] == pytest.approx(-300.0 + np.log(2), abs=1e-3)
    zp = log_noised([ro["logp_option"]])
    assert np.isfinite(zp).all() and zp[0, 0] < np.log(EPS)        # far below log EPS, unclipped
    assert zp[0, 0] - zp[0, 1] == pytest.approx(-50.0 + np.log(2) - np.log1p(np.exp(-10.0)), abs=1e-2)
    zc = log_clean(np.array([[1e-30, 1.0]]))
    assert zc[0, 0] == pytest.approx(np.log(EPS))                   # the clean side clips, as Phase 6


# --------------------------------------------------------------------------- decisions and V_head
def _rows():
    """Raw P (summing to 1 - leakage), leakage: normal, tied, flagged non-empty, flagged empty."""
    P = np.array([[0.70, 0.20, 0.08], [0.40, 0.40, 0.18], [0.20, 0.10, 0.05], [0.03, 0.02, 0.01]])
    return P, 1.0 - P.sum(1)


def test_v_head_exact_rule_leakage_stand_in_and_empty_rows():
    P, leak = _rows()
    mask, flagged, empty = v_head(P, leak, 0.1)
    assert flagged.tolist() == [False, False, True, True]
    assert mask[0].tolist() == [True, True, True]                   # 0.08 >= 0.1 * 0.70
    assert v_head(P, leak, 0.2)[0][0].tolist() == [True, True, False]   # 0.08 < 0.2 * 0.70
    assert mask[2].tolist() == [True, True, False]                  # threshold 0.1 * leakage 0.65 = 0.065
    assert empty.tolist() == [False, False, False, True]            # threshold 0.094 > every P
    u = np.array([[0, 0, 9.0], [0, 0, 9.0], [0, 0, 9.0], [0, 0, 9.0]])
    pred = decide(u, mask, empty, P.argmax(1))
    assert pred.tolist() == [2, 2, 0, 0]          # row 2 masks C; row 3 (empty) keeps the clean argmax
    m02, _, e02 = v_head(P, leak, 0.2)
    assert decide(u, m02, e02, P.argmax(1))[0] == 0   # at beta 0.2 row 0 masks C
    R = np.array([[0.1, 0.8, 0.05]] * 4) * np.array([[1], [1], [1], [0.05]])
    m2, _, e2 = v_head(R, 1 - R.sum(1), 0.1)
    assert e2.tolist() == [False, False, False, True]
    assert decide(u, m2, e2, R.argmax(1))[3] == 1   # VCD + R's empty row falls back to R's argmax


@pytest.mark.parametrize("beta", [0.1, 0.2])
def test_alpha0_reproduces_base_and_R_including_flagged_and_empty_rows(beta):
    r = np.random.default_rng(3)
    P, leak = _rows()
    P = np.vstack([P, softmax(r.normal(0, 2, (200, 3))) * 0.97])
    leak = np.concatenate([leak, np.full(200, 0.03)])
    PR = np.vstack([P[:4][:, ::-1], softmax(r.normal(0, 2, (200, 3))) * 0.99])
    leakR = np.concatenate([1 - PR[:4].sum(1), np.full(200, 0.01)])
    zp = r.normal(0, 5, P.shape)
    for Pb, lb in ((P, leak), (PR, leakR)):
        mask, flagged, empty = v_head(Pb, lb, beta)
        assert flagged.any() and empty.any()
        z = log_clean(renorm_rows(Pb))
        pred = decide(vcd_score(z, zp, 0.0), mask, empty, Pb.argmax(1))
        assert np.array_equal(pred, Pb.argmax(1))
        assert np.array_equal(vcd_score(z, zp, 0.0), z)


# --------------------------------------------------------------------------- ranking invariances
def _ranking_data(k, n=300, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    p = softmax(r.normal(0, 1, (n, k)) + 0.8 * np.eye(k)[y])
    return y, p


@pytest.mark.parametrize("k", [2, 5, 7])
def test_constant_and_affine_zprime_leave_ovo_identical(k):
    y, p = _ranking_data(k)
    z = log_clean(p)
    base = ovo_logratio_auc(y, p, k)
    assert ovo_scores_auc(y, z, k) == base
    r = np.random.default_rng(1)
    const = r.normal(0, 3, k)
    for alpha in ALPHAS:
        assert abs(ovo_scores_auc(y, vcd_score(z, const, alpha), k) - base) < 1e-12
        a = 0.9 * (1 + alpha) / alpha
        assert abs(ovo_scores_auc(y, vcd_score(z, a * z + const, alpha), k) - base) < 1e-12
    if k == 2:
        assert abs(ovo_scores_auc(y, vcd_score(z, 3.0 * z + const, 1.0), k) - (1 - base)) < 1e-12   # a > (1+α)/α


@pytest.mark.parametrize("k", [2, 5, 7])
def test_ovo_of_base_cal_equals_base_and_offsets_leave_vcd_ovo(k):
    y, p = _ranking_data(k, seed=k)
    assert abs(ovo_logratio_auc(y, apply_bias(p, fit_bias(p, y, k)), k) - ovo_logratio_auc(y, p, k)) < 1e-12
    u = vcd_score(log_clean(p), np.random.default_rng(2).normal(0, 1, p.shape), 1.0)
    b = fit_bias(softmax(u), y, k)
    assert abs(ovo_scores_auc(y, u + b, k) - ovo_scores_auc(y, u, k)) < 1e-12


def test_lex_score_is_finite_and_ranks_masked_below_plausible():
    r = np.random.default_rng(4)
    u = r.normal(0, 20, (100, 5))
    z = np.minimum(r.normal(-3, 4, (100, 5)), 0.0)
    mask = r.random((100, 5)) < 0.5
    mask[np.arange(100), r.integers(0, 5, 100)] = True
    s = lex_score(u, z, mask)
    assert np.isfinite(s).all()
    assert s[mask].min() > s[~mask].max()                  # across all rows, so OvO ranks them that way too
    assert np.array_equal(s[mask], u[mask])
    with pytest.raises(ValueError):
        lex_score(u, z, mask, M=float(np.abs(u).max()))


def test_per_token_vcd_equals_option_level_plus_renyi():
    r = np.random.default_rng(5)
    p_ids = r.uniform(1e-6, 1, (50, 4, 2)) / 3
    pp_ids = r.uniform(1e-6, 1, (50, 4, 2)) / 3
    P, Pp = p_ids.sum(-1), pp_ids.sum(-1)
    for alpha in ALPHAS:
        lhs = per_token_score(np.log(p_ids), np.log(pp_ids), alpha)
        D = renyi(split_logw(np.log(p_ids)), split_logw(np.log(pp_ids)), 1 + alpha)
        assert np.allclose(lhs, vcd_score(np.log(P), np.log(Pp), alpha) + alpha * D, atol=1e-10)
        assert (D >= -1e-12).all()
        same = p_ids * (Pp / P)[..., None]                   # the same split in both views
        assert np.allclose(renyi(split_logw(np.log(p_ids)), split_logw(np.log(same)), 1 + alpha), 0, atol=1e-12)


def test_ovo_scores_auc_equals_ovo_logratio_auc():
    for k in (2, 5, 7):
        y, p = _ranking_data(k, seed=10 + k)
        assert ovo_scores_auc(y, np.log(np.clip(p, EPS, None)), k) == ovo_logratio_auc(y, p, k)


def test_a1_folds_match_diag_a1_and_the_sha_is_int64_bytes():
    y = np.repeat(np.arange(5), 30)
    splits, folds = a1_folds(y)
    ref = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(splits, ref))
    import hashlib
    assert fold_sha1(folds) == hashlib.sha1(folds.astype(np.int64).tobytes()).hexdigest()[:12]


def test_holm_and_bootstrap_p():
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert holm([0.5]) == [0.5]
    assert boot_p(np.full(999, 0.3)) == pytest.approx(2 / 1000)
    assert boot_p(np.linspace(-1, 1, 1001)) == pytest.approx(1.0)


def test_code_commit_from_index():
    text = "| G1 | CPU pre-checks | done `1aded69` |\n| G2 | code + tests: noised path | done `abc1234` |\n"
    assert code_commit_from_index(text) == "abc1234"
    with pytest.raises(SystemExit):
        code_commit_from_index("| G2 | code + tests | — |\n")


# --------------------------------------------------------------------------- caches (guards 4 and 8)
CLEAN_KEY = {"model_id": "m", "revision": "r", "dtype": "torch.bfloat16", "vision_attn": "sdpa",
             "image_processor_sha256": "0" * 64, "mm_tokens_per_image": 256}


class _Inner:
    def __init__(self):
        self.calls = 0

    def get_image_features(self, pixel_values, **kwargs):
        self.calls += 1
        return _VisionOutput((pixel_values.float().mean() + torch.arange(6.0)).reshape(2, 3).to(torch.bfloat16))


class _Model:
    def __init__(self):
        self.model = _Inner()
        self.dtype = torch.bfloat16


def test_noised_key_differs_and_digest_matches_vision_cache(tmp_path):
    clean = key_digest(CLEAN_KEY)
    assert clean == VisionCache(CLEAN_KEY, root=tmp_path).digest
    k500, k999 = noised_key(CLEAN_KEY, 500), noised_key(CLEAN_KEY, 999)
    assert k500["noise"] == {"schedule": "vcd_sigmoid_1000", "step": 500}
    assert len({clean, key_digest(k500), key_digest(k999)}) == 3
    with pytest.raises(ValueError):
        noised_key(k500, 999)
    assert clean_file(tmp_path, "dermamnist", "train", 7).name == VisionCache(CLEAN_KEY, tmp_path).path(
        "dermamnist", "train", 7).name


def test_noised_cache_refuses_a_clean_directory(tmp_path):
    d = tmp_path / key_digest(noised_key(CLEAN_KEY, 500))
    d.mkdir()
    (d / "key.json").write_text(json.dumps(CLEAN_KEY))
    with pytest.raises(RuntimeError, match="noised cache refuses"):
        NoisedVisionCache(CLEAN_KEY, 500, root=tmp_path)


def test_clean_run_refuses_a_noised_directory(tmp_path):
    digest = key_digest(CLEAN_KEY)
    with pytest.raises(RuntimeError, match="no clean cache"):
        clean_cache_for_key(CLEAN_KEY, digest, root=tmp_path)
    d = tmp_path / digest
    d.mkdir()
    (d / "key.json").write_text(json.dumps(noised_key(CLEAN_KEY, 500)))
    with pytest.raises(RuntimeError, match="refuses a noised cache directory"):
        clean_cache_for_key(CLEAN_KEY, digest, root=tmp_path)
    (d / "key.json").write_text(json.dumps(CLEAN_KEY))
    (d / "retinamnist_train_1.pt").write_bytes(b"")
    assert clean_cache_for_key(CLEAN_KEY, digest, root=tmp_path).dir == d
    (d / "retinamnist_train_1_s2404348969288661602.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="seed-named"):
        check_clean_dir(d, CLEAN_KEY)
    with pytest.raises(RuntimeError, match="digest"):
        clean_cache_for_key(CLEAN_KEY, "0" * 16, root=tmp_path)
    with pytest.raises(RuntimeError):
        clean_cache_for_key(noised_key(CLEAN_KEY, 500), digest, root=tmp_path)


def test_noised_cache_round_trip_counters_and_payload_checks(tmp_path):
    model, cache = _Model(), NoisedVisionCache(CLEAN_KEY, 500, root=tmp_path)
    pv = torch.ones(1, 3, 4, 4)
    seed = vcd_seed("dermamnist", "train", 209, 1)
    with cache.attached(model):
        assert getattr(model, CACHE_ATTR) is cache
        with pytest.raises(RuntimeError, match="outside a sample"):
            model.model.get_image_features(pv)
        with cache.sample("dermamnist", "train", 209, seed):
            first = model.model.get_image_features(pv).pooler_output
        path = cache.path("dermamnist", "train", 209, seed)
        assert path.name == f"dermamnist_train_209_s{seed}.pt" and path.exists()
        payload = torch.load(path, weights_only=True)
        assert set(payload) == {"features", "step", "seed"} and payload["step"] == 500 and payload["seed"] == seed
        with cache.sample("dermamnist", "train", 209, seed):
            again = model.model.get_image_features(pv * 7).pooler_output        # served from disk
        assert torch.equal(first, again) and model.model.calls == 1
        seed2 = vcd_seed("dermamnist", "train", 209, 2)
        with cache.sample("dermamnist", "train", 209, seed2):                     # another seed: another file
            model.model.get_image_features(pv)
        assert cache.stats() == {"key_digest": cache.digest, "step": 500, "hits": 1, "misses": 2}
        torch.save({"features": first, "step": 500, "seed": 1}, cache.path("dermamnist", "train", 5, 99))
        with cache.sample("dermamnist", "train", 5, 99), pytest.raises(RuntimeError, match="payload"):
            model.model.get_image_features(pv)
        torch.save(first, cache.path("dermamnist", "train", 6, 99))               # a bare (clean-style) tensor
        with cache.sample("dermamnist", "train", 6, 99), pytest.raises(RuntimeError, match="not a noised payload"):
            model.model.get_image_features(pv)
    assert getattr(model, CACHE_ATTR, None) is None
    assert NoisedVisionCache(CLEAN_KEY, 500, root=tmp_path).dir == cache.dir      # reopens its own directory
    assert NoisedVisionCache(CLEAN_KEY, 999, root=tmp_path).dir != cache.dir


def test_clean_rerun_cache_serves_hits_and_never_writes(tmp_path):
    digest = key_digest(CLEAN_KEY)
    d = tmp_path / digest
    d.mkdir()
    (d / "key.json").write_text(json.dumps(CLEAN_KEY))
    model, pv = _Model(), torch.ones(1, 3, 4, 4)
    stored = model.model.get_image_features(pv).pooler_output
    torch.save(stored, clean_file(d, "retinamnist", "train", 1))
    torch.save({"features": stored, "step": 500, "seed": 1}, clean_file(d, "retinamnist", "train", 2))
    before = dir_state(d)
    cache = clean_cache_for_key(CLEAN_KEY, digest, root=tmp_path)
    with cache.attached(model):
        assert getattr(model, CACHE_ATTR) is cache
        with cache.sample("retinamnist", "train", 1):
            got = model.model.get_image_features(pv * 5).pooler_output
        assert torch.equal(got, stored) and model.model.calls == 1        # served, not computed
        with cache.sample("retinamnist", "train", 3), pytest.raises(RuntimeError, match="never written"):
            model.model.get_image_features(pv)
        with cache.sample("retinamnist", "train", 2), pytest.raises(RuntimeError, match="not a clean entry"):
            model.model.get_image_features(pv)
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1
    assert dir_state(d) == before and model.model.calls == 1


def test_noised_dirs_state_lists_only_noised_directories(tmp_path):
    from effvar.vcd_cache import noised_dirs_state
    VisionCache(CLEAN_KEY, root=tmp_path)
    assert noised_dirs_state(tmp_path) == {}
    cache = NoisedVisionCache(CLEAN_KEY, 999, root=tmp_path)
    assert noised_dirs_state(tmp_path) == {cache.digest: {"step": 999, "n_files": 0}}
    torch.save({"features": torch.zeros(1), "step": 999, "seed": 3}, cache.path("dermamnist", "train", 1, 3))
    assert noised_dirs_state(tmp_path)[cache.digest]["n_files"] == 1
    assert noised_dirs_state(tmp_path / "absent") == {}


def test_probe_reads_the_clean_file_and_never_writes(tmp_path):
    d = tmp_path / key_digest(CLEAN_KEY)
    d.mkdir()
    (d / "key.json").write_text(json.dumps(CLEAN_KEY))
    model = _Model()
    pv = torch.ones(1, 3, 4, 4)
    torch.save(model.model.get_image_features(pv).pooler_output, clean_file(d, "retinamnist", "train", 1))
    before = dir_state(d)
    probe = CleanFeatureProbe(d)
    with probe.attached(model):
        assert getattr(model, CACHE_ATTR, None) is None
        with probe.sample("retinamnist", "train", 1):
            model.model.get_image_features(pv)
        assert probe.max_abs == 0.0 and probe.calls == 1
        with probe.sample("retinamnist", "train", 1):
            model.model.get_image_features(pv * 3)
        assert probe.max_abs > 0
    assert dir_state(d) == before and before["n_files"] == 1


# --------------------------------------------------------------------------- run layout
def test_presets_are_the_registered_sequences():
    assert [s.name for s in preset("smoke")] == ["base_identity", "R_identity", "base_s500", "base_s999"]
    assert [s.arm for s in preset("smoke")] == ["base", "var", "base", "base"]
    assert [s.name for s in preset("smoke-rerun")] == ["base_s500_rerun", "base_s999_rerun"]
    g4 = preset("g4", clean_rerun="full")
    assert [s.name for s in g4] == ["base_s500", "R_s500", "base_s999", "base_s500_rep2", "clean_R", "clean_base"]
    assert [s.expect for s in g4] == [(0, 900), (900, 0), (0, 900), (0, 150), (5, 0), (900, 0)]
    assert [s.rate_key for s in g4] == ["miss", "hit", "miss", "miss", "hit", "hit"]
    fb = {s.name: s for s in preset("g4", clean_rerun="fallback", max_new_tokens=1)}
    assert fb["clean_base"].expect == (151, 0) and fb["clean_base"].clean_rerun == "fallback"
    assert {s.max_new_tokens for s in fb.values() if s.mode == "noised"} == {1}
    assert {s.max_new_tokens for s in fb.values() if s.mode == "clean"} == {8}
    assert all(s.max_new_tokens == 8 for s in preset("smoke") + preset("smoke-rerun"))
    for s in preset("smoke") + preset("smoke-rerun") + g4 + list(fb.values()):
        validate_subrun(s)
        yaml.safe_dump(subrun_record(s))
    with pytest.raises(ValueError):
        preset("g4")
    sel = selection(fb["clean_base"], SUBSETS)
    assert sel["retinamnist"] == list(range(50)) + [258] and 209 in sel["dermamnist"]
    assert selection(fb["clean_R"], SUBSETS) == {"pneumoniamnist": [291, 4328], "dermamnist": [2086, 2353, 6009]}
    assert {f: len(v) for f, v in selection(g4[0], SUBSETS).items()} == {f: len(v) for f, v in SUBSETS.items()}


def test_selection_and_subrun_refusals():
    sub = SubRun(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1, limit=3,
                 indices=(("retinamnist", (258,)),))
    assert selection(sub, SUBSETS)["retinamnist"] == [0, 1, 2, 258]
    assert selection(sub, SUBSETS)["dermamnist"] == [0, 1, 2]
    only = SubRun(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1,
                  indices=(("retinamnist", (258,)),))
    assert selection(only, SUBSETS) == {"retinamnist": [258]}
    with pytest.raises(ValueError):
        selection(SubRun(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1,
                         indices=(("retinamnist", (999999,)),)), SUBSETS)
    bad = [dict(cache="noised", exp=f"{SMOKE_EXP}/x"),                 # the smoke is uncached
           dict(cache="none", exp=f"{V1_EXP}/x"),                       # only the smoke is uncached
           dict(identity=True),                                         # identity needs no cache
           dict(exp="tune/other/x"), dict(step=250), dict(rep=3), dict(max_new_tokens=4)]
    for over in bad:
        args = dict(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1)
        args.update(over)
        with pytest.raises(ValueError):
            validate_subrun(SubRun(**args))
    with pytest.raises(ValueError):
        validate_subrun(SubRun(name="c", exp=f"{V1_EXP}/c", arm="base", mode="clean", cache="clean", step=500))


def test_run_vcd_argument_refusals(run_vcd):
    ap = run_vcd.make_parser()

    def build(*argv):
        return run_vcd.build_subruns(ap.parse_args(list(argv)))

    assert [s.name for s in build("--preset", "smoke")[0]] == ["base_identity", "R_identity", "base_s500", "base_s999"]
    for argv in (("--preset", "smoke", "--arm", "base"), ("--preset", "smoke", "--clean-rerun", "full"),
                 ("--preset", "g4"), ("--preset", "smoke", "--max-new-tokens", "1"),
                 ("--preset", "g4", "--clean-rerun", "full", "--datasets", "dermamnist"),
                 ("--preset", "g4", "--clean-rerun", "full", "--replace", "why"),
                 ("--arm", "base", "--step", "500"),                                         # no exp name
                 ("--arm", "base", "--clean", "--step", "500", "--exp-name", f"{V1_EXP}/c"),
                 ("--arm", "base", "--clean", "--max-new-tokens", "1", "--exp-name", f"{V1_EXP}/c"),
                 ("--arm", "base", "--exp-name", f"{V1_EXP}/n"),                             # no step
                 ("--arm", "base", "--step", "500", "--identity", "--exp-name", f"{V1_EXP}/n"),   # identity, cached
                 ("--arm", "base", "--step", "500", "--no-cache", "--exp-name", f"{V1_EXP}/n"),   # uncached off smoke
                 ("--arm", "base", "--step", "500", "--exp-name", f"{SMOKE_EXP}/n"),              # cached in smoke
                 ("--arm", "base", "--step", "500", "--clean-rerun", "full", "--exp-name", f"{V1_EXP}/n"),
                 ("--arm", "base", "--step", "500", "--exp-name", "tune/elsewhere/n")):
        with pytest.raises(ValueError):
            build(*argv)
    (sub,), label = build("--arm", "var", "--clean", "--indices", "pneumoniamnist:291,4328", "dermamnist:2086",
                          "--exp-name", f"{V1_EXP}/clean_R2")
    assert label == "single_clean_R2" and sub.indices == (("pneumoniamnist", (291, 4328)), ("dermamnist", (2086,)))
    (sub,), _ = build("--arm", "base", "--step", "999", "--rep", "2", "--no-cache", "--identity",
                      "--exp-name", f"{SMOKE_EXP}/x")
    assert sub.cache == "none" and sub.identity and sub.rep == 2


# --------------------------------------------------------------------------- presets with a stub runtime
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _StubCache:
    def __init__(self, sub):
        self.sub, self.hits, self.misses = sub, 0, 0

    def stats(self):
        return {"key_digest": f"{self.sub.cache}-{self.sub.step}", "hits": self.hits, "misses": self.misses}


class _StubRuntime:
    def __init__(self, clock, log, row_s=1.0):
        self.clock, self.log, self.row_s = clock, log, row_s

    @contextmanager
    def open_cache(self, sub):
        yield None if sub.cache == "none" else _StubCache(sub)

    def clean_state(self):
        return {"dir": "clean0", "n_files": 3, "key_json_sha256": "abc"}

    def noised_state(self):
        return {}

    def samples(self, flag, indices):
        for i in indices:
            yield i, i % 2, None

    def row(self, sub, flag, idx, label, image, cache):
        self.log.append((sub.name, flag, idx))
        self.clock.t += self.row_s
        if cache is not None:
            if sub.expect and sub.expect[0] > 0:
                cache.hits += 1
            else:
                cache.misses += 1
        return {"index": idx, "label": label, "new_ids": [1, 2, 3], "first_token_in_options": True,
                "leakage": 0.01, "peak_vram_gib": 8.0}

    def config(self, sub, flag, indices):
        return {"mode": sub.mode, "decoding": {"max_new_tokens": sub.max_new_tokens}, "clean_rerun": sub.clean_rerun}


def _sequence(run_vcd, tmp_path, subs, *, load_s=10.0, row_s=1.0, replace=None, git_state=lambda: CLEAN_GIT,
              extra=None, label="t"):
    clock, log, loads = _Clock(), [], []

    def make_runtime():
        loads.append(1)
        clock.t += load_s
        return _StubRuntime(clock, log, row_s)

    code = run_vcd.run_sequence(subs, label=label, argv=["x"], make_runtime=make_runtime, subset_idx=SUBSETS,
                                results_dir=tmp_path, clock=clock, git_state=git_state, replace=replace, extra=extra,
                                code_trees=TREES)
    return code, log, loads


def _records(tmp_path):
    return [yaml.safe_load(p.read_text()) for p in sorted((tmp_path / V1_EXP / "processes").glob("*.yaml"))]


def test_preset_order_outputs_and_per_subrun_counters(run_vcd, tmp_path):
    code, log, loads = _sequence(run_vcd, tmp_path / "a", preset("smoke"))
    assert code == 0 and loads == [1]
    order = [name for name, _, _ in log]
    assert list(dict.fromkeys(order)) == ["base_identity", "R_identity", "base_s500", "base_s999"]
    for s in preset("smoke"):
        for flag in FLAGS:
            d = tmp_path / "a" / s.exp / flag
            assert {p.name for p in d.iterdir()} == {"predictions.jsonl", "config.yaml", "metrics.json"}
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            assert cfg["subrun"]["name"] == s.name and cfg["git"] == CLEAN_GIT and cfg["vision_cache_stats"] is None
    tmp_path = tmp_path / "b"
    code, log, _ = _sequence(run_vcd, tmp_path, preset("g4", clean_rerun="fallback"), row_s=0.1,
                             extra={"preset": "g4", "preset_clean_rerun": "fallback"}, label="g4")
    assert code == 0
    assert list(dict.fromkeys(name for name, _, _ in log)) == [s.name for s in preset("g4", clean_rerun="fallback")]
    for s in preset("g4", clean_rerun="fallback"):
        sel = selection(s, SUBSETS)
        tot = [0, 0]
        for flag, idx in sel.items():
            cfg = yaml.safe_load((tmp_path / s.exp / flag / "config.yaml").read_text())
            st = cfg["vision_cache_stats"]
            assert st["hits"] + st["misses"] == len(idx)                  # this sub-run's own delta, per dataset
            tot = [tot[0] + st["hits"], tot[1] + st["misses"]]
            assert cfg["clean_rerun"] == s.clean_rerun and cfg["preset_clean_rerun"] == "fallback"
            rows = [json.loads(line) for line in (tmp_path / s.exp / flag / "predictions.jsonl").open()]
            assert [r["index"] for r in rows] == idx
        assert tuple(tot) == s.expect                                      # the registered totals
    rec = _records(tmp_path)[-1]
    assert rec["status"] == "done" and rec["load_s"] == 10.0 and rec["preset_clean_rerun"] == "fallback"
    assert [s["name"] for s in rec["subruns"]] == [s.name for s in preset("g4", clean_rerun="fallback")]
    assert all(s["status"] == "done" and s["s_per_image"] == pytest.approx(0.1) for s in rec["subruns"])


def test_budget_stop_before_the_crossing_subrun(run_vcd, tmp_path):
    proc = tmp_path / V1_EXP / "processes"
    proc.mkdir(parents=True)
    (proc / "00000000T000000Z_1_prior.yaml").write_text(yaml.safe_dump(
        {"elapsed_s": 3500.0, "load_s": 10.0,
         "subruns": [{"status": "done", "rows": 30, "rate_key": "uncached", "s_per_image": 1.0}]}))
    code, log, loads = _sequence(run_vcd, tmp_path, preset("smoke"))
    assert code == 3 and loads == [1]
    ran = list(dict.fromkeys(name for name, _, _ in log))
    assert ran == ["base_identity", "R_identity", "base_s500"]       # 3510 + 3 x 30 = 3600; base_s999 would cross
    assert not (tmp_path / SMOKE_EXP / "base_s999").exists()
    rec = [r for r in _records(tmp_path) if r.get("label") == "t"][0]
    assert rec["status"] == "stopped_budget" and "base_s999" in rec["stop"]
    assert rec["budget"][-1]["total_s"] == pytest.approx(3630.0)
    assert sorted(p.name for p in (tmp_path / SMOKE_EXP).iterdir()) == ["R_identity", "base_identity", "base_s500"]


def test_budget_stop_before_the_load(run_vcd, tmp_path):
    proc = tmp_path / V1_EXP / "processes"
    proc.mkdir(parents=True)
    (proc / "0_prior.yaml").write_text(yaml.safe_dump({"elapsed_s": 3590.0, "load_s": 30.0, "subruns": []}))
    code, log, loads = _sequence(run_vcd, tmp_path, preset("smoke"))
    assert code == 3 and loads == [] and log == []
    assert not (tmp_path / SMOKE_EXP).exists()


def test_refuses_to_overwrite_before_loading(run_vcd, tmp_path):
    (tmp_path / SMOKE_EXP / "base_s500" / "retinamnist").mkdir(parents=True)

    def never():
        raise AssertionError("the model must not load")

    def attempt(root, subs=None, git_state=lambda: CLEAN_GIT, trees=TREES, replace=None):
        run_vcd.run_sequence(subs or preset("smoke"), label="t", argv=[], make_runtime=never, subset_idx=SUBSETS,
                             results_dir=root, clock=_Clock(), git_state=git_state, replace=replace, code_trees=trees)

    for replace in (None, "presets never replace"):
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            attempt(tmp_path, replace=replace)
    with pytest.raises(SystemExit, match="not clean"):
        attempt(tmp_path / "other", git_state=lambda: {**CLEAN_GIT, "code_changes": ["?? effvar/x.py"]})
    with pytest.raises(SystemExit, match="differ from the code commit"):          # guard 6 at run start
        attempt(tmp_path / "other", trees={**TREES, "effvar": "another-tree"})
    with pytest.raises(SystemExit, match="no rows selected"):
        attempt(tmp_path / "other", subs=[SubRun(name="e", exp=f"{V1_EXP}/e", arm="base", mode="noised",
                                                 cache="noised", step=500, rep=1, indices=(("retinamnist", ()),))])
    assert not (tmp_path / V1_EXP / "processes").exists() and not (tmp_path / "other").exists()


def test_subrun_refuses_if_the_code_changes_between_subruns(run_vcd, tmp_path):
    states = iter([CLEAN_GIT] * 9 + [{**CLEAN_GIT, "code_trees": {**TREES, "scripts": "edited"}}] * 50)
    with pytest.raises(SystemExit, match="differ from the code commit"):
        _sequence(run_vcd, tmp_path, preset("smoke"), git_state=lambda: next(states))
    done = [p.parent.name for p in (tmp_path / SMOKE_EXP).glob("*/*/config.yaml")]
    assert done and not (tmp_path / SMOKE_EXP / "base_s999").exists()


def test_a_crash_leaves_the_gpu_time_on_record(run_vcd, tmp_path):
    sub = SubRun(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1,
                 expect=(0, 900))
    clock = _Clock()

    class Crashing(_StubRuntime):
        def row(self, *a):
            if len(self.log) == 120:
                raise RuntimeError("simulated crash")
            return super().row(*a)

    def make_runtime():
        clock.t += 10.0
        return Crashing(clock, [])

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_vcd.run_sequence([sub], label="crash", argv=[], make_runtime=make_runtime, subset_idx=SUBSETS,
                             results_dir=tmp_path, clock=clock, git_state=lambda: CLEAN_GIT, code_trees=TREES)
    rec = _records(tmp_path)[-1]
    assert rec["status"] == "running x" and rec["elapsed_s"] >= 110.0      # load + 100 rows, rewritten every 50


def test_miss_rate_falls_back_to_the_uncached_rate(run_vcd, tmp_path):
    proc = tmp_path / V1_EXP / "processes"
    proc.mkdir(parents=True)
    (proc / "0_smoke.yaml").write_text(yaml.safe_dump({"elapsed_s": 100.0, "load_s": 30.0, "subruns": [
        {"status": "done", "rows": 30, "rate_key": "uncached", "s_per_image": 1.3},
        {"status": "done", "rows": 30, "rate_key": "hit", "s_per_image": 0.4}]}))
    ledger = run_vcd.Ledger(tmp_path, "t", [], _Clock())
    assert ledger.rate("miss") == (1.3, "measured (uncached)")
    assert ledger.rate("hit") == (0.4, "measured (hit)")
    assert ledger.load_estimate() == 30.0 and ledger.prior_s() == 100.0


def test_single_run_replace_archives_and_records(run_vcd, tmp_path):
    sub = SubRun(name="x", exp=f"{V1_EXP}/x", arm="base", mode="noised", cache="noised", step=500, rep=1, limit=2)
    old = tmp_path / sub.exp / "retinamnist"
    old.mkdir(parents=True)
    (old / "marker.txt").write_text("old run")
    with pytest.raises(SystemExit, match="--replace"):
        _sequence(run_vcd, tmp_path, [sub])
    code, _, _ = _sequence(run_vcd, tmp_path, [sub], replace="re-run after a crash")
    assert code == 0
    cfg = yaml.safe_load((old / "config.yaml").read_text())
    assert cfg["replaces"]["reason"] == "re-run after a crash"
    moved = tmp_path / cfg["replaces"]["archived"][0]["to"]
    assert (moved / "marker.txt").read_text() == "old run" and "_replaced" in moved.as_posix()
    assert _records(tmp_path)[-1]["replaces"]["reason"] == "re-run after a crash"


# --------------------------------------------------------------------------- guard checkers on a synthetic tree
def _p(flag, i, salt=""):
    r = np.random.default_rng(zlib.crc32(f"{flag}|{i}|{salt}".encode()))
    k = len(get_spec(flag).labels)
    return [float(x) for x in softmax(r.normal(0, 2, k)) * 0.99]


def _saved_row(flag, i, arm):
    P = _p(flag, i, arm)
    k = len(P)
    return {"index": i, "label": i % k, "pred": int(np.argmax(P)), "match_source": "label", "text": " A\n",
            "new_ids": [562, 107, 106], "input_len": 372, "p_option": P, "target_p": P[i % k], "entropy": 0.1,
            "leakage": 0.01, "first_token_in_options": True, "vision_ms": None, "prefill_ms": None,
            "lm_prefill_ms": None, "decode_ms_per_token": None, "lm_ms": None, "e2e_ms": None, "peak_vram_gib": 8.1}


def _new_fields(P, step=None, seed=None, rep=None, c=(None, None), identity=False):
    return {"peak_vram_gib": 8.2, "logits_sha1": f"sha{sum(P)}", "p_ids": [[p * 0.9, p * 0.1] for p in P],
            "logp_ids": [[float(np.log(p * 0.9)), float(np.log(p * 0.1))] for p in P],
            "logp_option": [float(np.log(p)) for p in P], "step": step, "seed": seed, "rep": rep, "c1": c[0],
            "c2": c[1], "identity": identity}


def _write(root, exp, flag, rows, cfg):
    d = Path(root) / exp / flag
    d.mkdir(parents=True, exist_ok=True)
    with (d / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


CSTATE = {"dir": "clean0", "n_files": 5, "key_json_sha256": "k"}


def _cfg(sub, stats, proc="proc_g4"):
    return {"subrun": subrun_record(sub), "mode": sub.mode, "decoding": {"max_new_tokens": sub.max_new_tokens},
            "subset": {"split": "train"}, "vision_cache_stats": stats, "clean_rerun": sub.clean_rerun,
            "clean_cache_state": {"start": CSTATE, "end": CSTATE}, "noised_cache_state": {"start": {}, "end": {}},
            "process_record": proc, "git": CLEAN_GIT, "git_end": CLEAN_GIT}


def _noised_rows(flag, idx, sub, feat=None):
    rows = []
    for i in idx:
        P = _p(flag, i, f"n{sub.step}{sub.rep}{sub.arm}")
        r = {**_saved_row(flag, i, "base"), "p_option": P,
             **_new_fields(P, sub.step, vcd_seed(flag, "train", i, sub.rep), sub.rep, noise_coefficients(sub.step))}
        if feat is not None:
            r["feat_maxabs_vs_clean_cache"] = feat
        rows.append(r)
    return rows


def build_tree(root, layout="full", g4=True):
    """A complete, consistent G3 + G4 results tree (saved runs, smoke, re-runs, G4) under `root`; G3 only with
    g4=False."""
    for arm, run in (("base", "tune/base"), ("var", "tune/var/rho0.5_summ0.1")):
        for flag in FLAGS:
            _write(root, run, flag, [_saved_row(flag, i, arm) for i in SUBSETS[flag]],
                   {"vision_cache_stats": {"key_digest": "clean0"}})
    for name, sub in smoke_specs().items():
        proc = "proc_rerun" if name.endswith("_rerun") else "proc_smoke"
        for flag, idx in selection(sub, SUBSETS).items():
            if sub.identity:
                arm = "base" if sub.arm == "base" else "var"
                rows = [{**_saved_row(flag, i, arm), **_new_fields(_saved_row(flag, i, arm)["p_option"], 500,
                         vcd_seed(flag, "train", i, 1), 1, (1.0, 0.0), True), "feat_maxabs_vs_clean_cache": 0.0}
                        for i in idx]
            else:
                pair = sub if not name.endswith("_rerun") else smoke_specs()[name[:-6]]
                rows = _noised_rows(flag, idx, pair, feat=0.25)
            _write(root, sub.exp, flag, rows, _cfg(sub, None, proc))
    for name, sub in (g4_specs(layout) if g4 else {}).items():
        digest = {"noised": f"n{sub.step}", "clean": "clean0"}[sub.cache]
        for flag, idx in selection(sub, SUBSETS).items():
            n = len(idx)
            stats = {"hits": n if sub.expect[0] else 0, "misses": 0 if sub.expect[0] else n, "key_digest": digest}
            if sub.mode == "clean":
                arm = sub.arm
                rows = [{**_saved_row(flag, i, arm), **_new_fields(_saved_row(flag, i, arm)["p_option"])} for i in idx]
            else:
                rows = _noised_rows(flag, idx, sub)
            _write(root, sub.exp, flag, rows, _cfg(sub, stats))


def _set_cfg(root, exp, flag, **changes):
    p = Path(root) / exp / flag / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    for k, v in changes.items():
        if "." in k:
            a, b = k.split(".")
            cfg[a][b] = v
        else:
            cfg[k] = v
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))


@pytest.mark.parametrize("layout", ["full", "fallback"])
def test_every_guard_passes_on_a_consistent_tree(check, tmp_path, layout):
    root = tmp_path / "results"
    build_tree(root, layout)
    for mode in ("guard1", "guard3", "guard8-smoke", "guard7", "guard8"):
        ok, lines, rec = check.run_mode(mode, root, "c0de", TREES, SUBSETS)
        assert ok, "\n".join(lines)
        assert rec["inputs"] and all(rel.startswith("results/") for rel in rec["inputs"])


def test_g3_guards_run_on_the_smoke_alone_and_catch_a_silent_null(check, tmp_path):
    root = tmp_path / "results"
    build_tree(root, g4=False)                                           # G3: no G4 output exists yet
    for mode in ("guard1", "guard3", "guard8-smoke"):
        ok, lines, _ = check.run_mode(mode, root, "c0de", TREES, SUBSETS)
        assert ok, "\n".join(lines)
    assert not check.run_mode("guard8", root, "c0de", TREES, SUBSETS)[0]    # the full guard 8 needs G4
    exp = smoke_specs()["base_s999"].exp
    p = root / exp / "retinamnist" / "predictions.jsonl"
    rows = [json.loads(line) for line in p.open()]
    rows[4]["feat_maxabs_vs_clean_cache"] = 0.0                         # noised features == clean ones: a null
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not check.run_mode("guard8-smoke", root, "c0de", TREES, SUBSETS)[0]
    build_tree(root, g4=False)
    _set_cfg(root, exp, "dermamnist", noised_cache_state={"start": {}, "end": {"n500": {"step": 500, "n_files": 1}}})
    ok, lines, _ = check.run_mode("guard8-smoke", root, "c0de", TREES, SUBSETS)
    assert not ok and any("noised cache was present or written" in line for line in lines)


def test_guard8_needs_the_first_noised_passes_to_start_empty(check, tmp_path):
    root = tmp_path / "results"
    build_tree(root)
    _set_cfg(root, f"{V1_EXP}/base_s999", "retinamnist",
             noised_cache_state={"start": {"n999": {"step": 999, "n_files": 30}}, "end": {}})
    ok, lines, _ = check.run_mode("guard8", root, "c0de", TREES, SUBSETS)
    assert not ok and any("base_s999: its noised cache held 30 files" in line for line in lines)


def test_compare_rows(check):
    saved = {i: _saved_row("retinamnist", i, "base") for i in range(3)}
    new = {i: {**r, **_new_fields(r["p_option"])} for i, r in saved.items()}
    assert check.compare_rows(new, saved)["identical"] == 3            # VRAM differs: ignored
    new[1] = {**new[1], "e2e_ms": 5.0}
    assert check.compare_rows(new, saved)["identical"] == 3            # timings are not compared
    bumped = dict(new)
    p = list(saved[2]["p_option"])
    p[0] = float(np.nextafter(p[0], 1.0))
    bumped[2] = {**new[2], "p_option": p}
    r = check.compare_rows(bumped, saved)
    assert r["differing"] == [2] and r["max_abs_dp"] > 0
    gone = {**new, 0: {k: v for k, v in new[0].items() if k != "text"}}
    assert check.compare_rows(gone, saved)["missing_fields"] == ["text"]


def test_guard1_and_guard3_detect_breaks(check, tmp_path):
    root = tmp_path / "results"
    build_tree(root)
    specs = smoke_specs()
    p = root / specs["base_s999_rerun"].exp / "dermamnist" / "predictions.jsonl"
    rows = [json.loads(line) for line in p.open()]
    rows[3]["logp_ids"][0][1] += 1e-9
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok, lines, _ = check.run_mode("guard3", root, "c0de", TREES, SUBSETS)
    assert not ok and any("DIFFER ['logp_ids']" in line for line in lines)
    import shutil
    shutil.rmtree(root / specs["base_s999_rerun"].exp)                 # only the step-500 pair left
    assert not check.run_mode("guard3", root, "c0de", TREES, SUBSETS)[0]
    build_tree(root)
    for flag in FLAGS:
        _set_cfg(root, specs["base_s500_rerun"].exp, flag, process_record="proc_smoke")   # same process
    assert not check.run_mode("guard3", root, "c0de", TREES, SUBSETS)[0]
    p = root / specs["R_identity"].exp / "pneumoniamnist" / "predictions.jsonl"
    rows = [json.loads(line) for line in p.open()]
    rows[0]["feat_maxabs_vs_clean_cache"] = 1e-3
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not check.run_mode("guard1", root, "c0de", TREES, SUBSETS)[0]


def test_guard7_refuses_a_layout_that_disagrees_with_the_record(check, tmp_path):
    root = tmp_path / "results"
    build_tree(root, "full")
    for flag in FLAGS:
        _set_cfg(root, f"{V1_EXP}/clean_base", flag, clean_rerun="fallback")
    ok, lines, _ = check.run_mode("guard7", root, "c0de", TREES, SUBSETS)
    assert not ok and any("recorded --clean-rerun fallback, detected from the rows full" in line for line in lines)


def test_guard8_refuses_mixed_max_new_tokens_and_wrong_counters(check, tmp_path):
    root = tmp_path / "results"
    build_tree(root)
    _set_cfg(root, f"{V1_EXP}/base_s999", "retinamnist", **{"decoding.max_new_tokens": 1})
    ok, lines, _ = check.run_mode("guard8", root, "c0de", TREES, SUBSETS)
    assert not ok and any("DIFFER" in line for line in lines)
    build_tree(root)
    _set_cfg(root, f"{V1_EXP}/R_s500", "dermamnist", vision_cache_stats={"hits": 0, "misses": 0, "key_digest": "n500"})
    assert not check.run_mode("guard8", root, "c0de", TREES, SUBSETS)[0]
    build_tree(root)
    _set_cfg(root, f"{V1_EXP}/clean_base", "retinamnist",
             clean_cache_state={"start": CSTATE, "end": {**CSTATE, "n_files": 6}})     # the clean cache was written
    assert not check.run_mode("guard8", root, "c0de", TREES, SUBSETS)[0]
    build_tree(root)
    assert not check.run_mode("guard8", root, "c0de", {**TREES, "scripts": "other"}, SUBSETS)[0]   # guard 6


# --------------------------------------------------------------------------- diag_vcd gate (guard 6, E1)
def _write_guards(check, root):
    for mode in ("guard1", "guard3", "guard7", "guard8"):
        ok, lines, rec = check.run_mode(mode, root, "c0de", TREES, SUBSETS)
        assert ok, "\n".join(lines)
        out = root / "vcd/V1/regression"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{mode}.json").write_text(json.dumps({**rec, "head": "h"}))


def test_diag_gate_passes_only_on_a_complete_checked_g4(check, diag, tmp_path):
    root = tmp_path / "results"
    with pytest.raises(SystemExit, match="refuses"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)                 # empty root
    build_tree(root)
    with pytest.raises(SystemExit, match="guard7: no result"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)                 # G4 complete, not checked
    _write_guards(check, root)
    assert diag.check_ready(root, "c0de", TREES, SUBSETS) == {"layout": "full", "max_new_tokens": 8}
    with pytest.raises(SystemExit, match="code differs"):
        diag.check_ready(root, "c0de", {**TREES, "effvar": "x"}, SUBSETS)
    with pytest.raises(SystemExit, match="checked code commit"):
        diag.check_ready(root, "other", TREES, SUBSETS)


def test_diag_gate_refuses_a_partial_g4(check, diag, tmp_path):
    import shutil
    root = tmp_path / "results"
    build_tree(root)
    _write_guards(check, root)
    shutil.rmtree(root / V1_EXP / "base_s999" / "pneumoniamnist")        # a missing run
    with pytest.raises(SystemExit, match="base_s999 pneumoniamnist: missing"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)
    build_tree(root)
    p = root / V1_EXP / "R_s500" / "dermamnist" / "predictions.jsonl"
    p.write_text("".join(p.read_text().splitlines(keepends=True)[:-1]))   # one row short
    with pytest.raises(SystemExit, match="R_s500 dermamnist: .* rows, registered"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)


def test_diag_gate_refuses_failed_or_stale_guard_records(check, diag, tmp_path):
    root = tmp_path / "results"
    build_tree(root)
    _write_guards(check, root)
    g8 = root / "vcd/V1/regression/guard8.json"
    rec = json.loads(g8.read_text())
    g8.write_text(json.dumps({**rec, "ok": False}))
    with pytest.raises(SystemExit, match="guard8: did not pass"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)
    g8.write_text(json.dumps(rec))
    p = root / V1_EXP / "base_s500" / "retinamnist" / "config.yaml"
    p.write_text(p.read_text() + "\n# edited after the check\n")
    with pytest.raises(SystemExit, match="stale"):
        diag.check_ready(root, "c0de", TREES, SUBSETS)


# --------------------------------------------------------------------------- diag_vcd analysis
def _synthetic(flag, k, n=70, seed=0, break_identity=False):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    n = len(y)
    logits = r.normal(0, 1.5, (n, k)) + 1.2 * np.eye(k)[y]
    P = softmax(logits) * 0.98
    PR = softmax(logits + r.normal(0, 0.3, (n, k))) * 0.985
    D = {"flag": flag, "k": k, "ordinal": flag == "retinamnist", "idx": np.arange(n), "y": y,
         "splits": a1_folds(y)[0], "b2_auc_opt": 0.9}
    for name, PP in (("b", P), ("R", PR)):
        D[f"p_{name}"] = renorm_rows(PP)
        D[f"z_{name}"] = log_clean(D[f"p_{name}"])
        D[f"P_{name}"], D[f"leak_{name}"] = PP, 1 - PP.sum(1)
        D[f"text_{name}"] = PP.argmax(1)
    if break_identity:
        D["z_b"] = D["z_b"][::-1].copy()
    for name, scale in (("b500", 0.6), ("R500", 0.6), ("b999", 0.2)):
        Pn = softmax(scale * logits + r.normal(0, 1, (n, k))) * 0.97
        D[f"lp_{name}"] = np.log(Pn)
        D[f"zp_{name}"] = log_noised(D[f"lp_{name}"])
        D[f"Pn_{name}"], D[f"leakn_{name}"] = Pn, 1 - Pn.sum(1)
        D[f"ftio_{name}"], D[f"nnew_{name}"], D[f"textn_{name}"] = np.ones(n, bool), np.full(n, 3), Pn.argmax(1)
    Pn = np.exp(D["lp_b500"])
    D["lpid_b500"] = np.log(np.stack([Pn * 0.85, Pn * 0.15], -1))
    D["rep_pos"] = np.arange(0, n, 2)
    D["zp_rep2"] = log_noised(np.log(softmax(0.6 * logits[D["rep_pos"]] + r.normal(0, 1, (len(D["rep_pos"]), k)))))
    D["c_pos"] = np.arange(n)
    D["pid_clean"] = np.stack([P * 0.9, P * 0.1], -1)
    D["pcf_b"], D["pcf_R"] = np.full(k, 1 / k), np.full(k, 1 / k)
    D["mismatch"] = []
    return D


def _data(**kw):
    return {"retinamnist": _synthetic("retinamnist", 5, **kw), "pneumoniamnist": _synthetic("pneumoniamnist", 2, **kw),
            "dermamnist": _synthetic("dermamnist", 7, **kw)}


def test_diag_aborts_before_writing_when_an_alpha0_identity_fails(diag, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="step 1 failed"):
        diag.analyse(_data(break_identity=True), out, {"test": True}, B=10, jobs=1)
    assert not out.exists() or not any(out.iterdir())


def test_diag_runs_end_to_end_on_synthetic_inputs(diag, tmp_path):
    res = diag.analyse(_data(), tmp_path, {"test": True}, B=20, jobs=1)
    assert (tmp_path / "vcd.json").exists() and (tmp_path / "vcd_report.md").exists()
    for flag, r in res["datasets"].items():
        ids = r["ovo_identities"]
        assert all(abs(v) < 1e-12 for key, v in ids.items() if not key.startswith("base_cal")), (flag, ids)
        arms = r["arms"]
        assert arms["VCD_0.5"]["origin"] == arms["VCD_0.5_cal"]["origin"] == arms["VCD_mean_1.0"]["origin"] == "card"
        for added in ("VCD_mean_0.25", "VCD_mean_2.0", "VCD+R_clean_0.5_cal", "VCD+R_both_1.0_cal", "VCD999_2.0_cal"):
            assert arms[added]["origin"] == "added at G2, report-only", added
        assert arms["VCD_0.5"]["role"] == "deciding" and arms["VCD_2.0"]["role"] == "report-only"
        c = r["contrasts"]
        assert c["primary: VCD+R_clean_0.5_cal - VCD_0.5_cal"]["origin"] == "added at G2, report-only"
        assert c["primary: VCD_2.0 - VCD_mean_2.0"]["origin"] == "added at G2, report-only"
        assert c["ovo: VCD_1.0 - base"]["origin"] == "card"
        assert set(r["rules"]["screen"]) == {0.5, 1.0}
    assert set(res["holm"]) == set(FLAGS) and all(len(v) == 2 for v in res["holm"].values())
    assert "added at G2, report-only" in (tmp_path / "vcd_report.md").read_text(encoding="utf-8")
