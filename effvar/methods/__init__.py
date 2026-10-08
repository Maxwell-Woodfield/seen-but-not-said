"""Method interface. A method declares its attention mode and installs hooks around one generation.

Hooks are installed inside the timed region (effvar.generation), so setup cost counts toward latency.
"""
from contextlib import nullcontext
from typing import ContextManager, Protocol


class Method(Protocol):
    name: str
    attn: str  # a key of effvar.model.ATTN_MODES

    def params(self) -> dict: ...

    def activate(self, model, inputs) -> ContextManager: ...


class BaseMethod:
    """Unmodified MedGemma."""

    name = "base"

    def __init__(self, attn: str = "sdpa"):
        self.attn = attn

    def params(self) -> dict:
        return {}

    def activate(self, model, inputs) -> ContextManager:
        return nullcontext()


from functools import partial  # noqa: E402

from effvar.methods.var import VarMethod  # noqa: E402  (registers the var attention implementation)
from effvar.methods.diag import DiagMethod  # noqa: E402

REGISTRY = {
    "diag": DiagMethod,  # Phase 6 (notes/diagnosis_plan.md): R with one part altered, R at defaults
    "base": BaseMethod,
    "var": VarMethod,
    "var_literal": partial(VarMethod, config_name="var_literal"),
    "var_attn": partial(VarMethod, config_name="var_attn"),
    "effvar_rank1_topm": partial(VarMethod, config_name="effvar_rank1_topm"),
}


def make_method(name: str, attn: str, **kwargs) -> Method:
    if name not in REGISTRY:
        raise ValueError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](attn=attn, **kwargs)
