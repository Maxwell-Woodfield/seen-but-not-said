"""TIMING-ONLY ablation knobs for the VAR path. ACCURACY IS NEVER EVALUATED WITH THESE.

These switch parts of the mechanism off, or stand a constant in for the detector, so the *cost* of
each part can be measured on its own (notes/status.md next-session item 1: the current
"~2.3 ms per call site" figure came from removing detector + gate + eager attention together in 8
layers, so it over-attributes).

Deliberately NOT reachable from `scripts/run_eval.py`: an `Ablation` can only be built by the timing
scripts, so it is impossible to produce an accuracy result from an ablated run through the normal CLI.
`VarConfig.params()` emits `ablate` only when one is set, so an un-ablated run is provably clean and an
ablated run is self-labelling in its own `config.yaml`.

`legacy` is the exception: it changes only *how* the work is done, never *what* is computed, so a
`legacy` run must be bitwise identical to `R`. Since 2026-09-20 the lean host path (four redundant
syncs/recomputations removed; +66.3 ms of D = 7 LM time on the m1 ladder) IS `R`'s implementation, and
`legacy=True` restores the first port's host work so the old `R` stays measurable. That 66.3 ms was
waste in our first port, not a method saving (notes/framework.md §0).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

CONST_SINKS_FILE = Path(__file__).resolve().parent.parent / "data" / "const_sinks.json"

# Where `edit_attention` returns early. Each value names the last stage that runs.
#   None     -- nothing skipped (this is `R`)
#   "hook"   -- the pre-hook fires but the detector does not run, so nothing downstream can
#   "detect" -- detector only; return before sink prep
#   "prep"   -- detector + split_sinks/visual_sinks; return before the gate
#   "gate"   -- detector + prep + gate; return before the edit
STOP_STAGES = (None, "hook", "detect", "prep", "gate")

_CONST_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def load_const_sinks() -> dict:
    """The modal per-layer sink set over the 900 TUNE images (notes/sink_geometry.md).

    Derived from the **train** split and never from val or test: val made the `R` pick and test is
    untouched, so fitting these constants on either would leak an evaluation split into the method.
    If this file is ever regenerated, regenerate it from `results/tune/sinkcheck` only.
    """
    return json.loads(CONST_SINKS_FILE.read_text(encoding="utf-8"))


def const_sink_tensor(layer_idx: int, sinks: list[int], device) -> torch.Tensor:
    """One shared device tensor per (layer, device), allocated once per process.

    A real 'constant set elsewhere' implementation would keep these resident rather than rebuild them
    per image, so caching here is what makes the arm's cost honest rather than pessimistic.
    """
    key = (layer_idx, str(device))
    t = _CONST_CACHE.get(key)
    if t is None:
        t = torch.tensor(sorted(sinks), dtype=torch.long, device=device)
        _CONST_CACHE[key] = t
    return t


@dataclass(frozen=True)
class Ablation:
    """Timing-only. Every field defaults to 'behave exactly like `R`'."""

    stop_after: str | None = None
    # Layers where the REAL detector runs. None = everywhere `active_layer` allows. Any other active
    # layer takes its set from `const_sinks` instead -- the shape of the layers-12-15 detector lever
    # (notes/sink_geometry.md), which removes 20 of 32 detector call sites.
    detect_layers: tuple[int, ...] | None = None
    const_sinks: dict[int, list[int]] | None = field(default=None, compare=False)
    # Restore the first port's host work, which the default path no longer does: the `int(keep.sum())`
    # bookkeeping sync, the `edit.any()` re-check that `keep.any()` already proved, `split_sinks` on
    # every forward, and the row-filter `arange` per call. Output-identical; asserted, not assumed.
    legacy: bool = False
    # Measurement 3 (batch regime): per-row sink sets as [B, T] masks instead of one index list, so a
    # batch of DISTINCT images gets each image's own sinks (`nonzero(mask)[:, 1]` would union them).
    # Mathematically R per row, but the masked reductions sum in a different order from the index path,
    # so it is NOT bitwise R -- validated by tests/test_ablate.py and scripts/check_batch.py instead.
    per_row: bool = False

    def __post_init__(self):
        if self.stop_after not in STOP_STAGES:
            raise ValueError(f"stop_after must be one of {STOP_STAGES}, got {self.stop_after!r}")
        if self.detect_layers is not None and self.const_sinks is None:
            raise ValueError("detect_layers needs const_sinks to stand in for the skipped layers")
        if self.per_row and (self.stop_after or self.detect_layers is not None or self.legacy):
            raise ValueError("per_row is the batched R and combines with no other ablation")

    def params(self) -> dict:
        out: dict = {}
        if self.stop_after is not None:
            out["stop_after"] = self.stop_after
        if self.detect_layers is not None:
            out["detect_layers"] = list(self.detect_layers)
            out["const_sink_layers"] = sorted(self.const_sinks)
        if self.legacy:
            out["legacy"] = True
        if self.per_row:
            out["per_row"] = True
        return out


def detector_lever() -> Ablation:
    """The layers-12-15 detector lever's SHAPE, for timing only (notes/sink_geometry.md).

    The real detector runs in layers 2-8 (never characterised), 12-15 (the four image-specific layers,
    and the only ones measured to change a prediction) and 33; the 20 positional layers take the modal
    set instead. That removes 20 of 32 detector call sites, which is the accounting in the note.
    """
    data = load_const_sinks()
    const = {int(k): v["sinks"] for k, v in data["sets"].items()}
    return Ablation(detect_layers=tuple(data["_meta"]["detect_layers"]), const_sinks=const)


# Named arm sets for scripts/ab_rounds.py --set. Every set keeps `S` (the dR_LM denominator) and
# `R_dup` (the noise floor), so each run carries its own denominator and its own resolution limit.
ARM_SETS = {
    "ladder": ["S", "Ev", "Hk", "Dt", "DtS", "DtG", "R", "R_dup", "R_legacy"],
    # Reduced ladder on the LEAN R (2026-09-21): prep = DtS - Dt, gate = DtG - DtS, edit = R - DtG.
    # Dt - S is eager + hooks + detect together (none of them changed by lean; m1 splits them).
    "lean_ladder": ["S", "Dt", "DtS", "DtG", "R", "R_dup"],
    # Measurement 3: with --batch B, ab_rounds runs R and R_dup on the per-row path.
    "m3": ["S", "R", "R_dup"],
    # m = 24 against BOTH references: `M24_legacy` vs `R_legacy` repeats the published +18.6 ms
    # like-for-like (that was a fixed-order A/B, drift-inflated); `M24` vs `R` is the current saving.
    "m24": ["S", "R", "R_dup", "R_legacy", "M24", "M24_legacy"],
    # Final test session (approved 2026-09-21): base timed SDPA (`S`) and eager (`Ev`), both references,
    # m = 24. One mirrored run, so every method shares one denominator (validation_plan.md §2).
    "test": ["S", "Ev", "R", "R_dup", "M24", "VA15"],
}
M24_LAYERS = tuple(range(9, 33))  # band 9-32, frozen 2026-09-18 (validation_plan.md §3)


def arms() -> dict[str, dict]:
    """Every timing arm. `bitwise` names the arm whose output this arm must reproduce exactly.

    Rungs Ev..R are nested, so their marginals telescope to `R - Ev` BY CONSTRUCTION -- that sum is an
    identity, not a consistency check, and must never be reported as one. `R_dup` is a byte-for-byte
    copy of `R`: its measured effect is zero, so it calibrates the whole runner, and any component
    effect smaller than the `R` vs `R_dup` spread is reported as unresolved.

    `R` is the lean implementation since 2026-09-20; `R_legacy` is the first port. The detector lever
    (`R_det1215`) was measured and dropped the same day (3.1% of dR_LM on top of the lean path); it is
    kept, on the current R, only so that result can be reproduced, and is in no arm set. `VA*` exist
    for scripts/check_ablate.py's implementation-swap check and are in no arm set either.
    """
    legacy = Ablation(legacy=True)
    return {
        "S":      dict(method="base", attn="sdpa", ablate=None, bitwise=None,
                       note="base SDPA -- the dR_LM denominator"),
        "Ev":     dict(method="base", attn="var_eager", ablate=None, bitwise="Ev",
                       note="eager attention via our kernel, state=None: no hooks, no VAR"),
        "Hk":     dict(method="var", attn="var_eager", ablate=Ablation(stop_after="hook"), bitwise="Ev",
                       note="34 pre-hooks registered and firing, detector never runs"),
        "Dt":     dict(method="var", attn="var_eager", ablate=Ablation(stop_after="detect"), bitwise="Ev",
                       note="detector only"),
        "DtS":    dict(method="var", attn="var_eager", ablate=Ablation(stop_after="prep"), bitwise="Ev",
                       note="detector + split_sinks/visual_sinks"),
        "DtG":    dict(method="var", attn="var_eager", ablate=Ablation(stop_after="gate"), bitwise="Ev",
                       note="detector + prep + gate, no edit"),
        "R":      dict(method="var", attn="var_eager", ablate=None, bitwise="R",
                       note="R = var, tau 10 (lean implementation since 2026-09-20)"),
        "R_dup":  dict(method="var", attn="var_eager", ablate=None, bitwise="R",
                       note="byte-for-byte copy of R: the in-run noise floor"),
        "R_legacy": dict(method="var", attn="var_eager", ablate=legacy, bitwise="R",
                         note="R as first ported: the old R, with the host work lean removed"),
        "R_det1215": dict(method="var", attn="var_eager", ablate=detector_lever(), bitwise=None,
                          note="dropped lever A on the current R: detector in 12 layers, modal set in 20"),
        # m = 24 is bitwise identical to R on all val and tune images (status.md), so its group is R.
        "M24":    dict(method="effvar_rank1_topm", attn="var_eager", layers=M24_LAYERS, ablate=None,
                       bitwise="R", note="effvar_rank1_topm m = 24, current implementation"),
        "M24_legacy": dict(method="effvar_rank1_topm", attn="var_eager", layers=M24_LAYERS, ablate=legacy,
                           bitwise="R", note="effvar_rank1_topm m = 24 as first ported"),
        "VA":     dict(method="var_attn", attn="var_eager", ablate=None, bitwise="VA",
                       note="var_attn, current implementation (swap check only)"),
        "VA_legacy": dict(method="var_attn", attn="var_eager", ablate=legacy, bitwise="VA",
                          note="var_attn as first ported (swap check only)"),
        "VA15":   dict(method="var_attn", attn="var_eager", tau=15.0, ablate=None, bitwise="VA15",
                       note="var_attn, tau 15: the H0 reference (results/val/var_attn_tau15)"),
    }


def method_kwargs(arm: dict) -> dict:
    """make_method() overrides for one arm; `base` takes none."""
    if arm["method"] == "base":
        return {}
    out = {}
    if arm["ablate"] is not None:
        out["ablate"] = arm["ablate"]
    if arm.get("layers") is not None:
        out["layers"] = arm["layers"]
    if arm.get("tau") is not None:
        out["tau"] = arm["tau"]
    return out
