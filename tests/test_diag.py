"""Model-free checks of the Phase 6 `diag` family (effvar/diag.py; notes/diagnosis_plan.md §2).

The load-bearing ones: at defaults the diag path is bitwise `R`'s (edit arithmetic, gate, and the whole
`edit_attention` including stats), and every arm keeps rows summing to 1 and is the identity at p = 0.
"""
from dataclasses import replace

import pytest
import torch

from effvar.diag import (DIAG_DEFAULT, SOURCES, TARGETS, DiagConfig, DiagState, centre_prior, diag_gate,
                         parse_overrides, redistribute_diag)
from effvar.var import VarConfig, VarState, head_gate, redistribute

IM, P, T = 4, 16, 40          # P = 16 is a 4 x 4 patch grid, so target="centre" works
VIS, TEXT = torch.tensor([6, 11]), torch.tensor([0, 2])


def attn(B=1, H=8, Q=6, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(B, H, Q, T, generator=g) * 2, dim=-1).to(dtype)


def cfg(**kw):
    kw.setdefault("centre_sigma", 1.5 if kw.get("target") == "centre" else None)
    return replace(DIAG_DEFAULT, **kw)


@pytest.mark.parametrize("p", [0.2, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_defaults_are_bitwise_redistribute(p, dtype):
    a = attn(dtype=dtype)
    edit = torch.rand(a.shape[:3], generator=torch.Generator().manual_seed(1)) > 0.3
    assert torch.equal(redistribute_diag(a, IM, P, VIS, TEXT, edit, cfg(p=p)),
                       redistribute(a, IM, P, VIS, TEXT, edit, p, check_any=False))


def test_default_gate_is_head_gate():
    a = attn()
    assert torch.equal(diag_gate(a, IM, P, VIS, DIAG_DEFAULT), head_gate(a, IM, P, VIS, DIAG_DEFAULT))


def _states(c: DiagConfig, n_layers=4):
    base_cfg = VarConfig(**{k: getattr(c, k) for k in ("p", "rho", "tau", "summ")})
    r = VarState(cfg=base_cfg, im=IM, P=P, n_layers=n_layers)
    d = DiagState(cfg=c, im=IM, P=P, n_layers=n_layers, rng=torch.Generator().manual_seed(0))
    for s in (r, d):
        s.sinks[2] = torch.cat([TEXT, VIS])
    return r, d


@pytest.mark.parametrize("Q", [T, 1])            # a full prefill (rows filtered) and one decode step
def test_state_at_defaults_is_varstate(Q):
    r, d = _states(DIAG_DEFAULT)
    a = torch.softmax(torch.randn(1, 8, Q, T) * 3, dim=-1).to(torch.bfloat16)
    assert torch.equal(d.edit_attention(2, a), r.edit_attention(2, a))
    r.finalise_stats(), d.finalise_stats()
    assert r.stats == d.stats


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("target", TARGETS)
def test_every_arm_keeps_row_sums_and_is_identity_at_p0(source, target):
    a = attn()
    edit = torch.ones(a.shape[:3], dtype=torch.bool)
    out = redistribute_diag(a, IM, P, VIS, TEXT, edit, cfg(source=source, target=target, p=0.5))
    torch.testing.assert_close(out.sum(-1), a.sum(-1), rtol=0, atol=2e-6)
    assert not torch.equal(out, a)
    for dtype in (torch.float32, torch.bfloat16):
        b = a.to(dtype)
        assert torch.equal(redistribute_diag(b, IM, P, VIS, TEXT, edit, cfg(source=source, target=target, p=0.0)), b)


def test_source_visual_and_text_leave_the_other_sinks_alone():
    a = attn()
    edit = torch.ones(a.shape[:3], dtype=torch.bool)
    vis_only = redistribute_diag(a, IM, P, VIS, TEXT, edit, cfg(source="visual", p=0.5))
    assert torch.equal(vis_only[..., TEXT], a[..., TEXT])
    torch.testing.assert_close(vis_only[..., VIS], a[..., VIS] * 0.5)
    text_only = redistribute_diag(a, IM, P, VIS, TEXT, edit, cfg(source="text", p=0.5))
    assert torch.equal(text_only[..., VIS], a[..., VIS])
    torch.testing.assert_close(text_only[..., TEXT], a[..., TEXT] * 0.5)


def test_targets_place_the_budget_as_specified():
    a = attn()
    edit = torch.ones(a.shape[:3], dtype=torch.bool)
    budget = 0.5 * (a[..., VIS].sum(-1) + a[..., TEXT].sum(-1))
    free = torch.ones(P, dtype=torch.bool)
    free[VIS - IM] = False
    gained = lambda t: (redistribute_diag(a, IM, P, VIS, TEXT, edit, cfg(target=t, p=0.5)) - a)  # noqa: E731
    uni = gained("uniform")[..., IM:IM + P][..., free]
    torch.testing.assert_close(uni, (budget / free.sum()).unsqueeze(-1).expand_as(uni))
    sq = a[..., IM:IM + P][..., free] ** 2
    torch.testing.assert_close(gained("sharpen")[..., IM:IM + P][..., free],
                               budget.unsqueeze(-1) * sq / sq.sum(-1, keepdim=True))
    prior = centre_prior(P, 1.5, "cpu")[free]
    torch.testing.assert_close(gained("centre")[..., IM:IM + P][..., free],
                               budget.unsqueeze(-1) * prior / prior.sum())
    txt = gained("text")
    assert torch.equal(txt[..., IM:IM + P][..., free], torch.zeros_like(txt[..., IM:IM + P][..., free]))
    cols = torch.ones(T, dtype=torch.bool)
    cols[IM:IM + P] = False
    cols[TEXT] = False
    torch.testing.assert_close(txt[..., cols].sum(-1), budget)


def test_centre_prior_peaks_in_the_middle():
    g = centre_prior(16, 1.5, "cpu").view(4, 4)
    assert g[1:3, 1:3].min() > g[0, 0] and torch.allclose(g, g.T)


def test_answer_summ_changes_only_the_last_prefill_row():
    # Every row: image mass 0.05 spread evenly (sink share 2/16, so rho passes), the rest on text.
    # summ = 0.1 then rejects every row; only the floor-off answer row can open.
    a = torch.zeros(1, 8, 8, T)
    a[..., IM:IM + P] = 0.05 / P
    text_cols = [c for c in range(T) if not IM <= c < IM + P]
    a[..., text_cols] = 0.95 / len(text_cols)
    base = diag_gate(a, IM, P, VIS, DIAG_DEFAULT)
    opened = diag_gate(a, IM, P, VIS, cfg(answer_summ=0.0))
    assert not base.any()
    assert torch.equal(opened[..., :-1], base[..., :-1]) and opened[..., -1].any()


@pytest.mark.parametrize("rows,expected", [("post_image", [20, 21, 22, 23]), ("answer", [23]),
                                           ("prompt", [20, 21, 22])])
def test_row_filters(rows, expected):
    st = DiagState(cfg=cfg(rows=rows), im=IM, P=P, n_layers=4)
    assert torch.nonzero(st._row_filter(24, "cpu")).flatten().tolist() == expected


def test_answer_and_prompt_modes_skip_decode_steps():
    for rows in ("answer", "prompt"):
        _, d = _states(cfg(rows=rows))
        a = attn(Q=1)
        assert d.edit_attention(2, a) is a


def test_random_sinks_same_size_disjoint_from_detector_and_reproducible():
    picks = []
    for _ in range(2):
        st = DiagState(cfg=cfg(sinks="random"), im=IM, P=P, n_layers=4, rng=torch.Generator().manual_seed(0))
        picks.append(st._visual_override(2, VIS, "cpu"))
    assert torch.equal(picks[0], picks[1]) and picks[0].numel() == VIS.numel()
    assert not set(picks[0].tolist()) & set(VIS.tolist())
    assert all(IM <= i < IM + P for i in picks[0].tolist())
    empty = torch.empty(0, dtype=torch.long)
    assert st._visual_override(2, empty, "cpu").numel() == 0   # no visual sink stays a no-op layer


def test_const_sinks():
    st = DiagState(cfg=cfg(sinks="const", const_patch=3, const_from=2), im=IM, P=P, n_layers=4)
    assert st._visual_override(2, VIS, "cpu").tolist() == [IM + 3]
    assert st._visual_override(1, VIS, "cpu").numel() == 0


def test_const_sinks_stay_in_R_band_9_to_32():
    """Amendment 2d: R never edits layer 33, so the constant arm must not either."""
    st = DiagState(cfg=cfg(sinks="const"), im=IM, P=P, n_layers=34)
    assert [st._visual_override(ly, VIS, "cpu").numel() for ly in (8, 9, 32, 33)] == [0, 1, 1, 0]


def test_budget_logging_is_read_only_and_matches_the_edit():
    r, d = _states(cfg(record_budget=True))
    a = torch.softmax(torch.randn(1, 8, T, T, generator=torch.Generator().manual_seed(3)) * 3, dim=-1)
    out = d.edit_attention(2, a)
    assert torch.equal(out, r.edit_attention(2, a))          # logging changes nothing
    log = d.budget[2]
    row = a[0, :, -1]
    expect = 0.4 * (row[:, VIS].sum(-1) + row[:, TEXT].sum(-1))
    ed = log["moved"] > 0
    torch.testing.assert_close(log["moved"][ed], expect[ed], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(log["S_v"], row[:, VIS].sum(-1))
    s = d.budget_summary()
    assert s["by_layer"]["2"]["heads_edited"] == int(ed.sum())
    assert abs(s["moved_total"] - float(log["moved"].mean())) < 1e-7


def test_budget_logs_only_prefill_and_only_when_asked():
    _, d = _states(DIAG_DEFAULT)
    d.edit_attention(2, attn(Q=T))
    assert d.budget == {}
    _, d = _states(cfg(record_budget=True))
    d.edit_attention(2, attn(Q=1))                             # decode step
    assert d.budget == {}


def test_observe_state_records_without_editing_and_edits_like_diag_when_asked():
    from effvar.diag import ObserveState
    a = torch.softmax(torch.randn(1, 8, T, T, generator=torch.Generator().manual_seed(4)) * 3, dim=-1)
    obs = ObserveState(cfg=DIAG_DEFAULT, im=IM, P=P, n_layers=4, rng=torch.Generator().manual_seed(0))
    obs.sinks[2] = torch.cat([TEXT, VIS])
    assert obs.edit_attention(2, a) is a
    rec = obs.answer[2]
    row = a[0, :, -1]
    torch.testing.assert_close(rec["heads"][:, 0], row[:, IM:IM + P].sum(-1))
    torch.testing.assert_close(rec["heads"][:, 1], row[:, VIS].sum(-1))
    torch.testing.assert_close(rec["heads"][:, 3], row[:, 0])
    assert rec["vis"].tolist() == (VIS - IM).tolist() and rec["img_attn"].shape == (P,)
    assert obs.edit_attention(0, a) is a and obs.answer[0]["vis"].numel() == 0   # no sinks below layer 2
    _, d = _states(DIAG_DEFAULT)
    edit = ObserveState(cfg=DIAG_DEFAULT, im=IM, P=P, n_layers=4, edit=True, rng=torch.Generator().manual_seed(0))
    edit.sinks[2] = torch.cat([TEXT, VIS])
    assert torch.equal(edit.edit_attention(2, a), d.edit_attention(2, a))


def test_config_validation():
    for bad in ({"rows": "all"}, {"source": "bos"}, {"sinks": "fixed"}, {"target": "edges"},
                {"target": "centre", "centre_sigma": None}, {"sinks": "random", "sink_source": "attention"}):
        with pytest.raises(ValueError):
            replace(DIAG_DEFAULT, **bad)


def test_params_record_the_knobs():
    prm = cfg(target="centre", sinks="const").params()
    assert prm["family"] == "diag" and prm["target"] == "centre" and prm["centre_sigma"] == 1.5
    assert prm["const_patch"] == 192 and prm["convention"] == "paper"


def test_parse_overrides():
    got = parse_overrides(["p=0.6", "target=uniform", "answer_summ=none", "layers=9-11", "seed=3",
                           "except_last_layer=true", "record_budget=true", "const_to=31"])
    assert got == {"p": 0.6, "target": "uniform", "answer_summ": None, "layers": (9, 10, 11), "seed": 3,
                   "except_last_layer": True, "record_budget": True, "const_to": 31}
    for bad in (["record_delta=1"], ["p"], ["ablate=x"]):
        with pytest.raises(ValueError):
            parse_overrides(bad)
