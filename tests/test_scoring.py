import math

import numpy as np
import pytest
import torch

from effvar.prompts import get_spec
from effvar.scoring import dataset_metrics, first_token_metrics, parse_answer

UNIFORM = [0.0] * 7


@pytest.mark.parametrize("text,expected", [
    ("B", (1, "label")),
    ("**B**", (1, "label")),
    ("B. basal cell carcinoma", (1, "label")),
    ("basal cell carcinoma", (1, "text")),
    ("melanoma", (4, "text")),
    ("melanocytic nevi", (5, "text")),
    ("Answer: E", (4, "label")),
    ("a basal cell carcinoma", (1, "text")),  # lowercase article is not label A
])
def test_parse_derma(text, expected):
    assert parse_answer(text, get_spec("dermamnist"), UNIFORM) == expected


def test_parse_retina_letters_and_fallback():
    spec = get_spec("retinamnist")
    assert parse_answer("C", spec, [0.0] * 5) == (2, "label")
    assert parse_answer("D. grade 3 - severe non-proliferative diabetic retinopathy", spec, [0.0] * 5) \
        == (3, "label")
    assert parse_answer("grade 4 - proliferative diabetic retinopathy", spec, [0.0] * 5) == (4, "text")
    assert parse_answer("I cannot tell", spec, [0.1, 0.2, 0.6, 0.05, 0.05]) == (2, "fallback")


def test_parse_pneumonia():
    spec = get_spec("pneumoniamnist")
    assert parse_answer("A", spec, [0, 0]) == (0, "label")
    assert parse_answer("The image shows pneumonia.", spec, [0.9, 0.1]) == (1, "text")


def test_first_token_metrics_known_values():
    logits = torch.log(torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.05]))
    sets = [[0, 3], [1]]  # option 0 -> tokens 0,3; option 1 -> token 1
    m = first_token_metrics(logits, sets, label=1)
    assert m["p_option"] == pytest.approx([0.6, 0.2], abs=1e-6)
    assert m["target_p"] == pytest.approx(0.2, abs=1e-6)
    assert m["leakage"] == pytest.approx(0.2, abs=1e-6)
    p = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
    assert m["entropy"] == pytest.approx(-sum(x * math.log(x) for x in p), abs=1e-5)
    assert m["first_token_in_options"] is True


def _row(label, pred):
    return {"label": label, "pred": pred, "target_p": 0.5, "entropy": 1.0, "leakage": 0.1,
            "first_token_in_options": True, "match_source": "label", "new_ids": [1, 2],
            "vision_ms": 380.0, "prefill_ms": 550.0, "lm_prefill_ms": 170.0,
            "decode_ms_per_token": None, "lm_ms": None, "e2e_ms": 600.0, "peak_vram_gib": 8.0}


def test_dataset_metrics_toy():
    rows = [_row(0, 0), _row(1, 2), _row(2, 2), _row(2, 1)]
    m = dataset_metrics(rows, n_classes=3, ordinal=True)
    assert m["accuracy"] == pytest.approx(0.5)
    # per-class F1: c0 = 1.0; c1: tp 0, fp 1, fn 1 -> 0; c2: tp 1, fp 1, fn 1 -> 0.5
    assert m["macro_f1"] == pytest.approx(0.5)
    # QWK by hand: O = [[1,0,0],[0,0,1],[0,1,1]], weights (i-j)^2/4, E from marginals
    O = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]], float)
    W = np.array([[(i - j) ** 2 / 4 for j in range(3)] for i in range(3)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    assert m["qwk"] == pytest.approx(1 - (W * O).sum() / (W * E).sum())
    assert m["decode_ms_per_token_mean"] is None and m["lm_ms_mean"] is None
    assert m["lm_prefill_ms_median"] == pytest.approx(170.0)
    assert m["new_tokens_median"] == 2 and m["new_tokens_max"] == 2
