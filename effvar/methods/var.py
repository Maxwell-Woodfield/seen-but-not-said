"""`var` / `var_literal`: the faithful VAR port, wired into MedGemma.

Two hook sites, mirroring `VisAttnSink/src/model/llava.py`:
- decoder-layer forward pre-hook (llava.py:314-316): DimProspector on the layer's *input* residual
  stream, prefill only, layers >= first_layer.
- a registered attention implementation (llava.py:139-148): HeadFork + redistribution on the softmaxed
  weights, before they multiply V. The body is a verbatim copy of transformers'
  `eager_attention_forward` for Gemma 3 with the edit spliced in, so p = 0 is bitwise identical to
  `base` on the eager path (validation_plan.md G0).

State lives on the attention modules (`_effvar_state`), not in a global stash, so an inactive model is
untouched and nested runs cannot leak into each other.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface
from transformers.models.gemma3.modeling_gemma3 import repeat_kv

from effvar.profile import HostTimer, RegionTimer, maybe
from effvar.model import VAR_ATTN_NAME, image_span
from effvar.var import VAR_CONFIGS, VarConfig, VarState

STATE_ATTR = "_effvar_state"


def var_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          attention_mask: torch.Tensor | None, dropout: float = 0.0,
                          scaling: float | None = None, softcap: float | None = None,
                          **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """transformers' Gemma 3 `eager_attention_forward` with the VAR edit spliced in after the softmax."""
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    state: VarState | None = getattr(module, STATE_ATTR, None)
    timer = state.timer if state is not None else None
    phase = "prefill" if query.shape[-2] > 1 else "decode"

    with maybe(timer, "qk", module.layer_idx, phase):
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    before = None
    if state is not None:
        if state.cfg.record_delta:
            before = attn_weights
        attn_weights = state.edit_attention(module.layer_idx, attn_weights)

    with maybe(timer, "av", module.layer_idx, phase):
        attn_output = torch.matmul(attn_weights, value_states)
    if before is not None and attn_weights is not before:  # `is not` => this layer was actually edited
        state.record_delta_o(module.layer_idx, before, attn_weights, value_states, attn_output)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


AttentionInterface.register(VAR_ATTN_NAME, var_attention_forward)
# Without its own mask entry, a custom implementation is handed attention_mask=None (transformers
# assumes the kernel applies causality itself) and the eager body above would attend everywhere.
# `eager_mask` is exactly what attn_implementation="eager" gets, so `base` and `var` share one mask.
AttentionMaskInterface.register(VAR_ATTN_NAME, eager_mask)


def text_layers(model) -> nn.ModuleList:
    return model.model.language_model.layers


class VarMethod:
    """Faithful VAR (paper convention). Needs the `var_eager` text path; vision stays on SDPA."""

    def __init__(self, attn: str = "var_eager", config_name: str = "var", **overrides):
        cfg = VAR_CONFIGS[config_name]
        self.cfg = replace(cfg, **overrides) if overrides else cfg
        if self.cfg.profile and self.cfg.profile_host:
            # CUDA event records would land inside the perf_counter brackets and inflate them.
            raise ValueError("profile and profile_host are mutually exclusive")
        self.name = self.cfg.name
        self.attn = attn
        self.last_state: VarState | None = None

    def params(self) -> dict:
        return self.cfg.params()

    def new_state(self, **kwargs) -> VarState:
        """The per-generation state. `effvar.methods.diag.DiagMethod` overrides this, and nothing else."""
        return VarState(**kwargs)

    @contextmanager
    def activate(self, model, inputs):
        ids = inputs["input_ids"]
        if ids.shape[0] > 1 and not bool((ids == ids[:1]).all()):
            # One image span per state: batches must share the prompt (same dataset), which they do.
            raise ValueError("batched VAR needs identical input_ids across rows (one dataset per batch)")
        im, P = image_span(ids[:1], model.config)
        layers = text_layers(model)
        timer = RegionTimer() if self.cfg.profile else HostTimer() if self.cfg.profile_host else None
        state = self.new_state(cfg=self.cfg, im=im, P=P, n_layers=len(layers), timer=timer)
        self.last_state = state

        def make_hook(idx):
            def hook(module, args, kwargs):
                state.observe(idx, args[0] if args else kwargs["hidden_states"])
            return hook

        handles = []
        for i, layer in enumerate(layers):
            setattr(layer.self_attn, STATE_ATTR, state)
            handles.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=True))
        try:
            yield state
        finally:
            for h in handles:
                h.remove()
            for layer in layers:
                if hasattr(layer.self_attn, STATE_ATTR):
                    delattr(layer.self_attn, STATE_ATTR)
