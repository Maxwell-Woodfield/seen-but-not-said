"""`diag`: the Phase 6 config family (effvar/diag.py) wired in exactly as `var` is.

Same hook sites and attention implementation as `VarMethod`; only the per-generation state differs
(`DiagState`), so everything outside `edit_attention` -- detection, masks, stats -- is `R`'s own code.
`ObserveMethod` (B2's instrument) is `DiagMethod` with an `ObserveState`: it records every layer's
decision row, and edits only when `edit=True`.
"""
from dataclasses import replace

import torch

from effvar.diag import DIAG_DEFAULT, DiagState, ObserveState
from effvar.methods.var import VarMethod


class DiagMethod(VarMethod):
    def __init__(self, attn: str = "var_eager", **overrides):
        super().__init__(attn=attn)
        self.cfg = replace(DIAG_DEFAULT, **overrides) if overrides else DIAG_DEFAULT
        self.name = self.cfg.name
        # One generator per method: `sinks="random"` draws a fresh set per image, reproducibly for a
        # fixed image order (the subset files fix it).
        self.rng = torch.Generator().manual_seed(self.cfg.seed)

    def new_state(self, **kwargs) -> DiagState:
        return DiagState(rng=self.rng, **kwargs)


class ObserveMethod(DiagMethod):
    def __init__(self, attn: str = "var_eager", edit: bool = False, **overrides):
        super().__init__(attn=attn, **overrides)
        self.edit = edit
        self.name = "observe_R" if edit else "observe_base"

    def new_state(self, **kwargs) -> ObserveState:
        return ObserveState(rng=self.rng, edit=self.edit, **kwargs)
