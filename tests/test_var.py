"""Model-free checks of the VAR port (validation_plan.md G0 pytest half, G1).

`literal_attn_redist` is a transcription of VisAttnSink/src/logic/logic.py:163-212 in its own (code)
convention, with p_code = 1 - p. The vectorised edit must match it exactly in fp32.
"""
import copy

import pytest
import torch

from effvar.var import VarConfig, head_gate, redistribute, sink_mask, split_sinks

torch.manual_seed(0)


def literal_attn_redist(attention_map, im, pa, indices, coord, p_code):
    """logic.py:163-212, loop and all. `coord` holds [bsz, head, query] rows kept by HeadFork."""
    attention_map = attention_map.clone()
    n_heads = attention_map.shape[1]
    if len(coord) == 0:
        return attention_map
    for h in range(n_heads):
        query_coord = coord[coord[:, 1] == h][:, 2]
        bsz_coord = coord[coord[:, 1] == h][:, 0][:len(query_coord)]
        head_coord = coord[coord[:, 1] == h][:, 1][:len(query_coord)]
        if not query_coord.shape[0] or not head_coord.shape[0]:
            continue

        selected_attn_map = attention_map[bsz_coord, head_coord, query_coord, :].clone()
        vis_indices = indices[(im <= indices) & (indices < im + pa)]
        text_indices = indices[~torch.isin(indices, vis_indices)]
        copied_attention_map = copy.deepcopy(selected_attn_map.detach())

        selected_attn_map[:, text_indices] *= p_code
        selected_attn_map[:, vis_indices] *= p_code

        weight_budget_vis = copied_attention_map[:, vis_indices].sum(dim=1) * (1 - p_code)
        weight_budget_text = copied_attention_map[:, text_indices].sum(dim=1) * (1 - p_code)

        copied_attention_map[:, vis_indices] *= 0
        ratios_vis = copied_attention_map[:, im:im + pa] / copied_attention_map[:, im:im + pa].sum(
            dim=1, keepdim=True).to(selected_attn_map.dtype)
        selected_attn_map[:, im:im + pa] += (weight_budget_vis + weight_budget_text).view(-1, 1) * ratios_vis
        attention_map[bsz_coord, head_coord, query_coord, :] = selected_attn_map
    return attention_map


def make_attn(B=1, H=8, Q=6, T=24):
    return torch.softmax(torch.randn(B, H, Q, T, dtype=torch.float32), dim=-1)


@pytest.mark.parametrize("n_vis", [1, 3, 7])  # n_vis = P is the degenerate case below
@pytest.mark.parametrize("n_text", [0, 2])
@pytest.mark.parametrize("p", [0.2, 0.4, 0.6])
def test_vectorised_edit_matches_literal_loop(n_vis, n_text, p):
    """Not bitwise: the loop reduces gathered [rows, P] slices, the vectorised edit reduces [B,H,Q,P],
    so the sums associate differently and a few elements land within ~1e-7 relative (fp32 eps is
    1.2e-7). p = 0 is exact, which is what G0 rests on (see test_p_zero_is_the_identity)."""
    im, P, T = 4, 8, 24
    attn = make_attn(T=T)
    vis = torch.arange(im, im + n_vis)
    text = torch.arange(1, 1 + n_text)
    sinks = torch.cat([text, vis])
    edit = torch.rand(attn.shape[:3]) > 0.4

    coord = torch.nonzero(edit)
    mine = redistribute(attn, im, P, vis, text, edit, p)
    theirs = literal_attn_redist(attn, im, P, sinks, coord, p_code=1.0 - p)
    assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-7)
    rel = ((mine - theirs).abs() / theirs.abs().clamp_min(1e-12)).max()
    assert rel < 1e-5, rel


def test_all_image_tokens_sink_is_skipped_not_nan():
    """|I_v| = P leaves no non-sink image column; the reference divides by zero, we no-op the row."""
    im, P = 4, 8
    attn = make_attn()
    vis = torch.arange(im, im + P)
    edit = torch.ones(attn.shape[:3], dtype=torch.bool)
    out = redistribute(attn, im, P, vis, torch.empty(0, dtype=torch.long), edit, 0.4)
    assert torch.isfinite(out).all()
    assert torch.equal(out, attn)


def test_p_zero_is_the_identity():
    im, P = 4, 8
    for dtype in (torch.float32, torch.bfloat16):
        attn = make_attn().to(dtype)
        edit = torch.ones(attn.shape[:3], dtype=torch.bool)
        out = redistribute(attn, im, P, torch.tensor([5, 7]), torch.tensor([1]), edit, 0.0)
        assert torch.equal(out, attn), dtype


def test_redistribution_preserves_row_sums_and_moves_mass_to_image():
    im, P = 4, 8
    attn = make_attn()
    vis, text = torch.tensor([5, 6]), torch.tensor([1])
    edit = torch.ones(attn.shape[:3], dtype=torch.bool)
    out = redistribute(attn, im, P, vis, text, edit, 0.5)
    assert torch.allclose(out.sum(-1), attn.sum(-1), atol=1e-6)
    assert torch.equal(out[..., vis], attn[..., vis] * 0.5)
    assert (out[..., im:im + P].sum(-1) > attn[..., im:im + P].sum(-1)).all()  # image gains, text does not


def test_no_visual_sinks_is_a_no_op_even_with_text_sinks():
    im, P = 4, 8
    attn = make_attn()
    edit = torch.ones(attn.shape[:3], dtype=torch.bool)
    out = redistribute(attn, im, P, torch.empty(0, dtype=torch.long), torch.tensor([1, 2]), edit, 0.4)
    assert torch.equal(out, attn)


def test_head_gate_matches_the_code_formula():
    im, P = 4, 8
    attn = make_attn()
    vis = torch.tensor([5, 6])
    cfg = VarConfig(rho=0.5, summ=0.1)
    img = attn[..., im:im + P]
    portion = attn[..., vis].sum(-1) / torch.sum(img + 1e-6, dim=-1)
    expected = (portion <= 1.0 - cfg.rho) & (img.sum(-1) >= cfg.summ)
    assert torch.equal(head_gate(attn, im, P, vis, cfg), expected)


def test_sink_mask_finds_spiked_dims_and_respects_zero_dim():
    hidden = torch.randn(1, 6, 2560) * 0.1
    hidden[0, 2, 1313] = 400.0   # a real sink on a detector dim
    hidden[0, 4, 443] = 4000.0   # the dominant dim that must not create a sink on its own
    cfg = VarConfig(tau=10.0, dims=(1313, 2511), zero_dim=443)
    mask = sink_mask(hidden, cfg)
    assert mask[0, 2] and not mask[0, 4]
    # Without the exclusion, dim 443 swamps the RMS and nothing clears tau (medgemma_probe.md §B).
    assert not sink_mask(hidden, VarConfig(tau=10.0, dims=(1313, 2511), zero_dim=None))[0, 4]


def test_attention_sinks_picks_the_spiked_columns():
    from effvar.var import attention_sinks
    # P must exceed `mult`: a column's share of the mean caps the achievable ratio at P (real P = 256).
    im, P, T = 4, 32, 48
    attn = torch.full((1, 2, T, T), 0.01)
    attn[..., im + P:, im + 3] = 0.5   # one image column dominates the post-image rows
    picked = attention_sinks(attn, im, P, mult=10.0)
    assert picked.tolist() == [im + 3]
    # a flat image block has no sink, however many rows are averaged
    flat = torch.full((1, 2, T, T), 0.01)
    assert attention_sinks(flat, im, P, mult=10.0).numel() == 0


def test_attention_sinks_empty_without_post_image_rows():
    from effvar.var import attention_sinks
    attn = torch.rand(1, 2, 3, 20)  # Q = 3 rows, all inside/before the image block
    assert attention_sinks(attn, im=4, P=16, mult=10.0).numel() == 0


def test_split_sinks():
    vis, text = split_sinks(torch.tensor([0, 3, 5, 9, 20]), im=4, P=8)
    assert vis.tolist() == [5, 9] and text.tolist() == [0, 3, 20]


@pytest.mark.parametrize("first_layer,except_last,expected", [(2, False, [False, False, True, True]),
                                                              (2, True, [False, False, True, False])])
def test_active_layers(first_layer, except_last, expected):
    from effvar.var import VarState
    cfg = VarConfig(first_layer=first_layer, except_last_layer=except_last)
    state = VarState(cfg=cfg, im=4, P=8, n_layers=4)
    assert [state.active_layer(i) for i in range(4)] == expected


def test_layer_subset_restricts_the_whole_mechanism():
    """effvar_rank1_topm: a layer outside `layers` is inactive, so the DimProspector and the gate are
    skipped there too -- that is where the saving comes from, not the edit alone."""
    from effvar.var import VarState
    state = VarState(cfg=VarConfig(first_layer=2, layers=(3, 5)), im=4, P=8, n_layers=8)
    assert [state.active_layer(i) for i in range(8)] == [False] * 3 + [True, False, True, False, False]


def test_layer_subset_none_is_the_unrestricted_mechanism():
    from effvar.var import VarState
    a = VarState(cfg=VarConfig(first_layer=2, layers=None), im=4, P=8, n_layers=6)
    b = VarState(cfg=VarConfig(first_layer=2, layers=tuple(range(2, 6))), im=4, P=8, n_layers=6)
    assert [a.active_layer(i) for i in range(6)] == [b.active_layer(i) for i in range(6)]


def test_layer_subset_still_respects_first_layer_and_except_last():
    """`layers` narrows, never widens: a listed layer below first_layer or the excluded last stays off."""
    from effvar.var import VarState
    state = VarState(cfg=VarConfig(first_layer=2, except_last_layer=True, layers=(0, 2, 5)),
                     im=4, P=8, n_layers=6)
    assert [state.active_layer(i) for i in range(6)] == [False, False, True, False, False, False]


# ---- measurement 3: the per-row (batched) path must be R per row -----------------------------------
from effvar.ablate import Ablation  # noqa: E402
from effvar.var import VarState, head_gate_rows, redistribute_rows  # noqa: E402


def _row_case(B=3, H=4, Q=6, im=2, P=8, T0=14, extra=2):
    """Distinct per-row sink sets, including a row with no visual sink (text only), keys past T0."""
    attn = torch.rand(B, H, Q, T0 + extra, dtype=torch.float32).softmax(-1)
    sets = [torch.tensor([0, im + 1, im + 4]), torch.tensor([im + 6]), torch.tensor([1])][:B]
    mask = torch.zeros(B, T0, dtype=torch.bool)
    for b, s in enumerate(sets):
        mask[b, s] = True
    return attn, sets, mask, im, P


@pytest.mark.parametrize("p", [0.2, 0.4, 0.6])
def test_per_row_path_is_the_index_path_row_by_row(p):
    """G1-style, fp32: each row of the masked batch edit equals the index edit on that row alone."""
    attn, sets, mask, im, P = _row_case()
    cfg = VarConfig(p=p)
    in_img = torch.zeros(mask.shape[-1], dtype=torch.bool)
    in_img[im:im + P] = True
    vis = mask & in_img
    keep = head_gate_rows(attn, im, P, vis[:, im:im + P], cfg) & vis.any(-1)[:, None, None]
    out = redistribute_rows(attn, im, P, vis, mask, keep, p)
    for b, s in enumerate(sets):
        v, t = split_sinks(s, im, P)
        a = attn[b:b + 1]
        if v.numel() == 0:                     # the index path returns early: no visual sinks, no edit
            assert not keep[b].any()
            assert torch.equal(out[b:b + 1], a)
            continue
        k = head_gate(a, im, P, v, cfg)
        assert torch.equal(keep[b:b + 1], k)
        ref = redistribute(a, im, P, v, t, k, p)
        torch.testing.assert_close(out[b:b + 1], ref, rtol=1e-5, atol=1e-7)


def test_index_path_refuses_a_batch():
    """`nonzero(mask)[:, 1]` would union distinct images' sinks, so B > 1 needs per_row."""
    st = VarState(cfg=VarConfig(), im=2, P=8, n_layers=4)
    with pytest.raises(ValueError, match="per_row"):
        st.observe(2, torch.randn(2, 14, 16))


def test_per_row_state_keeps_one_mask_row_per_image():
    cfg = VarConfig(dims=(0, 1), zero_dim=None, ablate=Ablation(per_row=True))
    st = VarState(cfg=cfg, im=2, P=8, n_layers=4)
    h = torch.randn(3, 14, 16)
    st.observe(2, h)
    assert st.sinks[2].shape == (3, 14) and st.sinks[2].dtype == torch.bool


def test_per_row_rejects_other_ablations_and_var_attn():
    with pytest.raises(ValueError):
        Ablation(per_row=True, legacy=True)
    with pytest.raises(ValueError):
        VarState(cfg=VarConfig(sink_source="attention", ablate=Ablation(per_row=True)), im=0, P=4, n_layers=4)


def test_per_row_workload_counts_are_per_image():
    """m3 must report how much R edited per image, so a cheaper batch that edits less is visible."""
    st = VarState(cfg=VarConfig(ablate=Ablation(per_row=True)), im=2, P=8, n_layers=4)
    vis = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    sinks = vis.clone()
    sinks[:, 0] = True                                     # one text sink in both rows
    st._prep_cache[2] = (vis, sinks, vis.any(-1))
    keep = torch.zeros(2, 4, 6, dtype=torch.bool)
    keep[0, 1, 3:5] = True                                 # row 0: 2 rows edited; row 1: none
    st._keep_log.append((2, keep))
    st.finalise_stats()
    pr = st.stats["per_row"]
    assert pr["rows_edited"] == [2, 0] and pr["edit_events"] == [1, 0]
    assert pr["n_vis"] == [2.0, None] and pr["n_text"] == [1.0, None]
    from effvar.var import stats_summary
    s = stats_summary(st.stats)
    assert s["rows_edited"] == 1.0 and s["edit_events"] == 0.5 and s["n_vis"] == 2.0 and s["batch"] == 2
