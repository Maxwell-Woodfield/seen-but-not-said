"""Model-free tests for the timing-only ablation machinery (effvar/ablate.py, effvar/profile.py).

The load-bearing claim is that an un-ablated run is unchanged and a `legacy` run (the first port's host
work, restored for timing) is output-identical to R's lean implementation; the on-GPU proof is
`scripts/check_ablate.py`, but these pin the parts that need no model.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from effvar.ablate import ARM_SETS, M24_LAYERS, Ablation, arms, detector_lever, load_const_sinks, method_kwargs
from effvar.profile import REGIONS, SYNC_REGIONS, HostTimer, RegionTimer
from effvar.var import VAR_CONFIGS, VarConfig, VarState, redistribute


def test_default_config_is_unablated_and_params_stay_clean():
    cfg = VarConfig()
    assert cfg.ablate is None and cfg.profile_host is False
    # An accuracy run's config.yaml must not gain keys just because the machinery exists.
    assert "ablate" not in cfg.params() and "profile_host" not in cfg.params()
    for name, preset in VAR_CONFIGS.items():
        assert preset.ablate is None, f"{name} must ship un-ablated"


def test_ablated_config_labels_itself():
    cfg = VarConfig(ablate=Ablation(stop_after="gate", legacy=True))
    p = cfg.params()
    assert p["ablate"] == {"stop_after": "gate", "legacy": True}


def test_lean_is_the_default_implementation():
    """Since 2026-09-20 R runs the lean host path; only the timing-only `legacy` ablation turns it off."""
    assert VarState(cfg=VarConfig(), im=0, P=4, n_layers=4)._lean is True
    assert VarState(cfg=VarConfig(ablate=Ablation(legacy=True)), im=0, P=4, n_layers=4)._lean is False
    assert VarState(cfg=VarConfig(ablate=Ablation(stop_after="gate")), im=0, P=4, n_layers=4)._lean is True


def test_bad_stage_is_rejected():
    with pytest.raises(ValueError):
        Ablation(stop_after="everything")


def test_detect_layers_requires_a_stand_in():
    with pytest.raises(ValueError):
        Ablation(detect_layers=(12, 13))


def test_detector_lever_removes_twenty_of_thirty_two_sites():
    lever = detector_lever()
    # 32 detector call sites for R (layers 2..33); the lever keeps 12 and stands a constant in for 20.
    assert len(lever.detect_layers) == 12
    assert len(lever.const_sinks) == 20
    assert set(lever.detect_layers).isdisjoint(lever.const_sinks)
    # The four image-specific layers must keep a real detector: layer 15 alone carries the whole
    # Pneumonia effect and matches the positional set on 0.0% of images (notes/sink_geometry.md).
    assert {12, 13, 14, 15} <= set(lever.detect_layers)


def test_const_sink_sets_are_cost_faithful():
    """The stand-in must have R's typical set sizes, or the arm measures a cheaper mechanism."""
    for layer, entry in load_const_sinks()["sets"].items():
        sinks = entry["sinks"]
        assert 1 <= len(sinks) <= 4, f"layer {layer}: {sinks}"
        assert 197 in sinks, f"layer {layer} lost the visual sink"


def test_arm_table_declares_its_bitwise_groups():
    table = arms()
    assert table["R_dup"]["ablate"] is None and table["R_dup"]["bitwise"] == "R"
    for name in ("Hk", "Dt", "DtS", "DtG"):
        assert table[name]["bitwise"] == "Ev", f"{name} disables the edit, so it must match stock eager"
    assert table["R_legacy"]["bitwise"] == "R" and table["R_legacy"]["ablate"].legacy
    # m = 24 is bitwise R on every val/tune image, in both implementations.
    assert table["M24"]["bitwise"] == table["M24_legacy"]["bitwise"] == "R"
    assert table["VA_legacy"]["bitwise"] == "VA"
    # The dropped lever changes the sink set on purpose, so it is not in a bitwise group.
    assert table["R_det1215"]["bitwise"] is None


def test_every_arm_set_carries_its_own_denominator_and_noise_floor():
    table = arms()
    for name, members in ARM_SETS.items():
        assert {"S", "R", "R_dup"} <= set(members), name
        assert set(members) <= set(table), name
        for m in members:  # a bitwise group needs its anchor, or at least two members to compare
            ref = table[m]["bitwise"]
            if ref is not None and ref not in members:
                assert sum(table[x]["bitwise"] == ref for x in members) >= 2, (name, m)


def test_method_kwargs_carry_layers_and_ablation():
    table = arms()
    assert method_kwargs(table["S"]) == {}
    assert method_kwargs(table["R"]) == {}
    assert method_kwargs(table["M24"]) == {"layers": M24_LAYERS}
    kw = method_kwargs(table["M24_legacy"])
    assert kw["layers"] == M24_LAYERS and kw["ablate"].legacy
    assert M24_LAYERS == tuple(range(9, 33))


def _ab_report():
    path = Path(__file__).resolve().parent.parent / "scripts" / "ab_report.py"
    spec = importlib.util.spec_from_file_location("ab_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_as_run_metric_is_prefill_plus_measured_decode():
    """Per-site costs must divide an AS-RUN marginal; `lm_ms` extrapolates decode to D = 7."""
    rows = [{"lm_prefill_ms": 10.0, "decode_total_ms": 4.0}, {"lm_prefill_ms": 10.0, "decode_total_ms": None}]
    _ab_report().add_run_metric(rows)
    assert rows[0]["lm_run_ms"] == 14.0 and rows[1]["lm_run_ms"] is None


def test_check_any_false_is_identical_when_a_row_is_kept():
    """`lean` drops the `edit.any()` sync. Its only caller proves `keep.any()` first, so the guarded
    and unguarded paths must agree exactly whenever the edit actually runs."""
    torch.manual_seed(0)
    im, P = 2, 8
    attn = torch.rand(1, 4, 6, im + P + 3, dtype=torch.float32).softmax(-1).to(torch.bfloat16)
    vis = torch.tensor([im + 1, im + 4])
    txt = torch.tensor([0])
    keep = torch.zeros(1, 4, 6, dtype=torch.bool)
    keep[0, 1, 3] = True
    a = redistribute(attn, im, P, vis, txt, keep, 0.4, check_any=True)
    b = redistribute(attn, im, P, vis, txt, keep, 0.4, check_any=False)
    assert torch.equal(a, b)


def test_host_timer_duck_types_the_region_timer():
    t = HostTimer()
    for _ in range(3):
        with t.region("gate", 7, "prefill"):
            pass
    with t.region("edit", 7, "decode"):
        pass
    t.flush()                       # must be a no-op, not an error
    s = t.summary()
    assert s["kind"] == "host"
    assert s["calls"]["gate"] == 3 and s["calls"]["edit"] == 1
    assert s["ms"]["gate"]["prefill"] >= 0.0
    assert set(s["ms"]) == set(REGIONS)
    assert RegionTimer().summary()["kind"] == "gpu"


def test_prep_is_a_region_and_the_sync_regions_are_named():
    assert "prep" in REGIONS, "split_sinks/visual_sinks must be attributable separately from the gate"
    assert set(SYNC_REGIONS) == {"detect", "gate", "edit"}
    assert set(SYNC_REGIONS) <= set(REGIONS)


def test_profile_and_profile_host_are_mutually_exclusive():
    from effvar.methods import make_method
    with pytest.raises(ValueError):
        make_method("var", attn="var_eager", profile=True, profile_host=True)


def test_ladder_rungs_are_labelled_with_everything_they_add():
    """lean_ladder skips Ev and Hk, so Dt - S must not be reported as the detector alone."""
    rep = _ab_report()
    assert rep.adds("DtS", "Dt") == "prep"
    assert rep.adds("Dt", "S") == "eager vs SDPA + hook dispatch + detect"
    assert ARM_SETS["lean_ladder"] == ["S", "Dt", "DtS", "DtG", "R", "R_dup"]
    assert ARM_SETS["m3"] == ["S", "R", "R_dup"]
