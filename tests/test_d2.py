"""D2 yes/no path (Amendments 3b-d, 4c-f): token ids, the score s_c, OvO invariance on log s_c, class alignment,
the six rule tests on a known-answer fixture, the compliance guard, the smoke-reproduction check. No model."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig

from effvar import FLAGS, MODEL_ID
from effvar.diag_stats import apply_bias, boot_idx, fit_bias, ovo_logratio_auc
from effvar.model import TEMPLATE_KWARGS, load_processor
from effvar.prompt_variants import VARIANTS, YesNoSpec, variant_spec, yesno_specs
from effvar.prompts import ANSWER_PREFIX, get_spec
from effvar.yesno import (COMPLIANCE_MIN, NO_IDS, NO_TOKENS, YES_IDS, YES_TOKENS, image_row, yesno_first_token,
                          yesno_summary)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
V = 60_000   # > every yes/no id, so synthetic logits can address them


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.fixture(scope="module")
def d2():
    return _script("diag_d2")


# --------------------------------------------------------------------------- token ids (Amendment 3b)
def test_yes_no_ids_are_the_amendment_list():
    assert YES_TOKENS == {4443: "yes", 11262: " yes", 10784: "Yes", 8438: " Yes", 26915: "YES", 51327: " YES"}
    assert NO_TOKENS == {1904: "no", 951: " no", 3771: "No", 2301: " No", 7018: "NO", 9424: " NO"}
    assert not set(YES_IDS) & set(NO_IDS)


def test_token_ids_decode_and_encode_as_listed(processor):
    tok = processor.tokenizer
    for tid, form in {**YES_TOKENS, **NO_TOKENS}.items():
        assert tok.decode([tid]) == form, (tid, form)
        assert tok(form, add_special_tokens=False)["input_ids"] == [tid], form   # each form is one token


# --------------------------------------------------------------------------- the score s_c
def test_s_in_unit_interval_and_equal_to_the_ratio():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        logits = torch.randn(V, generator=g) * 4
        logits[list(YES_IDS + NO_IDS)] += torch.randn(12, generator=g) * 6 + 8
        m = yesno_first_token(logits)
        assert 0.0 <= m["s"] <= 1.0
        ratio = m["p_yes"] / (m["p_yes"] + m["p_no"])
        assert abs(m["s"] - ratio) <= 1e-5 * max(ratio, 1e-3)
        assert abs(m["leakage"] - (1 - m["p_yes"] - m["p_no"])) < 1e-12
        assert abs(np.log(m["s"]) - m["log_s"]) < 1e-9


def test_s_on_extreme_logits():
    z = torch.zeros(V)
    z[list(YES_IDS)] = 1e4
    m = yesno_first_token(z)
    assert m["s"] == 1.0 and m["compliant"] and m["top_id"] in YES_TOKENS
    z = torch.zeros(V)
    z[list(NO_IDS)] = 1e4
    m = yesno_first_token(z)
    assert m["s"] == 0.0 and m["compliant"] and m["top_id"] in NO_TOKENS
    z = torch.full((V,), -1e4)
    z[7] = 1e4                                 # all mass on a non-yes/no token
    z[4443], z[1904] = -5e3, -5e3 + 2.0        # yes and no both ~0 in probability
    m = yesno_first_token(z)
    assert 0.0 <= m["s"] <= 1.0 and np.isfinite(m["s"]) and not m["compliant"]
    assert m["leakage"] == pytest.approx(1.0) and m["s"] == pytest.approx(1 / (1 + np.e ** 2))
    z = torch.zeros(V)
    assert yesno_first_token(z)["s"] == pytest.approx(0.5)   # six yes ids vs six no ids, all equal


# --------------------------------------------------------------------------- OvO on log s_c (Amendment 3c)
@pytest.mark.parametrize("k", [2, 5, 7])
def test_ovo_on_log_s_is_invariant_to_per_class_offsets(k):
    r = np.random.default_rng(k)
    n = 400
    y = r.integers(0, k, n)
    s = np.clip(r.uniform(0.01, 0.6, (n, k)) + 0.35 * np.eye(k)[y], 1e-6, 1.0)   # unnormalised, like s_c
    base = ovo_logratio_auc(y, s, k)
    b = r.normal(0, 2.0, k)
    assert abs(ovo_logratio_auc(y, s * np.exp(b), k) - base) < 1e-12              # per-class offset on log s
    a = r.uniform(0.1, 3.0, (n, 1))
    assert abs(ovo_logratio_auc(y, s * a, k) - base) < 1e-12                      # per-image constant
    assert abs(ovo_logratio_auc(y, s / s.sum(1, keepdims=True), k) - base) < 1e-12
    assert abs(ovo_logratio_auc(y, apply_bias(s, fit_bias(s, y, k)), k) - base) < 1e-12   # the calibration


# --------------------------------------------------------------------------- class alignment
def test_frozen_variants_unchanged():
    assert sorted(VARIANTS) == ["frozen", "reversed"]          # run_eval --prompt cannot select yes/no
    with pytest.raises(ValueError):
        variant_spec("yesno", "dermamnist")


@pytest.mark.parametrize("flag", FLAGS)
def test_yesno_prompt_c_names_class_c_verbatim(flag):
    frozen, specs = get_spec(flag), yesno_specs(flag)
    assert len(specs) == len(frozen.labels) == len(frozen.option_texts)
    for c, spec in enumerate(specs):
        assert isinstance(spec, YesNoSpec) and spec.cls == c
        assert spec.option_texts == frozen.option_texts and spec.modality == frozen.modality
        assert spec.text() == f"Does this {frozen.modality} show {frozen.option_texts[c]}? Answer yes or no."


def test_absence_type_classes_are_asked_verbatim():   # Amendment 4(b): the prompt is not changed
    assert yesno_specs("pneumoniamnist")[0].text() == "Does this pediatric chest X-ray show normal? Answer yes or no."
    assert yesno_specs("retinamnist")[0].text() == ("Does this retinal fundus photograph show grade 0 - no diabetic "
                                                    "retinopathy? Answer yes or no.")


@pytest.mark.parametrize("flag", FLAGS)
def test_yesno_turn_is_the_frozen_turn_with_only_the_text_swapped(processor, flag):
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    kwargs = {**TEMPLATE_KWARGS, "tokenize": False, "return_dict": False, "return_tensors": None}
    frozen = get_spec(flag)
    ref = processor.apply_chat_template(frozen.messages(image), **kwargs)
    config = AutoConfig.from_pretrained(MODEL_ID)
    for spec in yesno_specs(flag):
        text = processor.apply_chat_template(spec.messages(image), **kwargs)
        assert text == ref.replace(frozen.text(), spec.text())
        assert text.endswith(ANSWER_PREFIX)
        ids = processor.apply_chat_template(spec.messages(image), **TEMPLATE_KWARGS)["input_ids"][0]
        pos = (ids == config.image_token_index).nonzero().flatten()
        assert pos.numel() == config.mm_tokens_per_image
        assert (pos == pos[0] + torch.arange(pos.numel())).all()


def _prompt(c, strength=0.0):
    z = torch.zeros(V)
    z[4443], z[1904] = 1.0 + strength, 1.0     # "yes", "no": the top token is a yes/no id (compliant)
    return {**yesno_first_token(z), "text": f"p{c}", "new_ids": [c], "logits_sha1": f"sha{c}"}


def test_image_row_puts_prompt_c_at_index_c():
    per_class = [_prompt(c, strength=float(c == 2) * 5.0) for c in range(4)]
    row = image_row(17, 3, per_class)
    assert row["index"] == 17 and row["label"] == 3
    assert row["text"] == ["p0", "p1", "p2", "p3"] and row["logits_sha1"] == ["sha0", "sha1", "sha2", "sha3"]
    assert row["s"] == [d["s"] for d in per_class] and row["pred"] == 2


def test_analysis_aligns_by_index_not_by_order(d2):
    k = 3
    letter = {i: {"label": i % k, "pred": 0, "p": np.eye(k)[i % k] * 0.5 + 0.5 / k} for i in range(9)}
    yn = {i: {"label": i % k, "pred": 0, "p": np.full(k, 0.1 * i + 0.05), "p_yes": np.full(k, float(i)),
              "leakage": np.zeros(k), "compliant": np.ones(k, bool)} for i in reversed(range(9))}   # reversed order
    idx, y, arr = d2.align_arms({"L": letter, "Y": yn}, exclude={4})
    assert idx.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert y.tolist() == [i % k for i in idx]
    assert arr["Y"]["p_yes"][:, 0].tolist() == [float(i) for i in idx]
    assert np.allclose(arr["L"]["p"], np.stack([letter[i]["p"] for i in idx]))
    bad = {**yn, 3: {**yn[3], "label": 2}}
    with pytest.raises(AssertionError):
        d2.align_arms({"L": letter, "Y": bad})
    with pytest.raises(ValueError):
        d2.align_arms({"L": letter, "Y": {i: yn[i] for i in range(8)}})


# --------------------------------------------------------------------------- Amendment 3(d) rules, known answers
def _fixture(case, n=240, k=3, seed=0):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    noisy = np.exp(r.normal(0, 1.0, (n, k)) + 0.4 * np.eye(k)[y])
    noisy /= noisy.sum(1, keepdims=True)                                  # weak letter-like scores
    sharp = np.clip(0.05 + 0.9 * np.eye(k)[y] + r.uniform(0, 0.02, (n, k)), 1e-6, 1)   # near-separable yes/no s
    if case == "readout":      # yes/no reads better than letters; R does nothing under either
        scores = {"base_letter": noisy, "R_letter": noisy, "base_yn": sharp, "R_yn": sharp}
    elif case == "c":          # yes/no = letters for base; R helps only under yes/no
        scores = {"base_letter": noisy, "R_letter": noisy, "base_yn": noisy, "R_yn": sharp}
    else:                      # nothing anywhere
        scores = {a: noisy for a in ("base_letter", "R_letter", "base_yn", "R_yn")}
    return y, k, scores


def _rules(d2, case, broken=None):
    y, k, scores = _fixture(case)
    arms = {a: d2.make_arm(s, s.argmax(1), y, k) for a, s in scores.items()}
    idx = boot_idx(len(y), B=200)
    point = {a: np.array(d2.metric_row(y, arms[a], k, False, np.arange(len(y)))) for a in arms}
    boot = {a: d2.boot_arm(y, arms[a], k, False, idx) for a in arms}
    return d2.rule_tests(point, boot, broken)


def test_six_rule_tests_on_a_known_answer_fixture(d2):
    got = {case: _rules(d2, case) for case in ("readout", "c", "null")}   # 3 "datasets" x 2 rules = 6 tests
    assert got["readout"]["readout"]["supported"] is True and got["readout"]["readout"]["lo"] > 0
    assert got["readout"]["c"]["supported"] is False
    assert got["readout"]["c"]["point"] == 0.0 and got["readout"]["c"]["lo"] == got["readout"]["c"]["hi"] == 0.0
    assert got["c"]["readout"]["supported"] is False and got["c"]["readout"]["point"] == 0.0
    assert got["c"]["c"]["supported"] is True and got["c"]["c"]["lo"] > 0
    for rule in ("readout", "c"):
        r = got["null"][rule]
        assert r["supported"] is False and r["point"] == r["lo"] == r["hi"] == 0.0


def test_a_rule_reading_a_broken_arm_is_not_interpreted(d2):
    got = _rules(d2, "c", broken={"R_yn": True})
    assert got["c"]["supported"] is None and not got["c"]["interpreted"]
    assert got["readout"]["supported"] is False and got["readout"]["interpreted"]   # reads base_yn only


# --------------------------------------------------------------------------- compliance guard (Amendment 3c)
def test_compliance_guard(d2):
    assert COMPLIANCE_MIN == 0.99
    comp = np.zeros(1000, bool)
    comp[:985] = True
    assert d2.guard(comp, np.zeros(1000))["broken"] is True        # 0.985 < 0.99
    comp[:990] = True
    g = d2.guard(comp, np.full(1000, 0.25))
    assert g["broken"] is False and g["compliance"] == 0.99 and g["leakage_mean"] == 0.25
    rows = [image_row(i, 0, [_prompt(c) for c in range(2)]) for i in range(5)]
    rows[0]["compliant"] = [False, True]                            # 9/10 compliant
    summ = yesno_summary(rows)
    assert summ["n_prompts"] == 10 and summ["compliance_rate"] == 0.9 and summ["broken"] is True
    assert summ["s_finite"] and 0.0 <= summ["s_min"] <= summ["s_max"] <= 1.0


# --------------------------------------------------------------------------- smoke reproduction (Amendment 4f)
def test_smoke_repro():
    sr = _script("check_d2_regression").smoke_repro
    smoke = {i: {"logits_sha1": [f"{i}-{c}" for c in range(3)]} for i in range(10)}
    full = {i: {"logits_sha1": [f"{i}-{c}" for c in range(3)]} for i in range(300)}
    r = sr(full, smoke)
    assert r["ok"] and r["images_identical"] == 10 and r["prompts_identical"] == r["prompts"] == 30
    changed = {**full, 4: {"logits_sha1": ["4-0", "x", "4-2"]}}
    r = sr(changed, smoke)
    assert not r["ok"] and r["differing_indices"] == [4] and r["prompts_identical"] == 29
    r = sr({i: v for i, v in full.items() if i != 7}, smoke)
    assert not r["ok"] and r["differing_indices"] == [7]
