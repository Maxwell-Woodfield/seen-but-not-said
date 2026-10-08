"""Greedy generation with vision/prefill/decode timing and first-step logits.

Timing wraps generation only (inputs are built beforehand), with torch.cuda.synchronize around
time.perf_counter. The prefill boundary is the first logits-processor call, which happens once the
prefill logits exist. The vision encoder is timed as its own component (SigLIP + projector), so the
language-model share of prefill can be reported separately.

LM time (notes/validation_plan.md §2, H1) = (prefill_ms - vision_ms) + DECODE_STEPS_D * decode ms/token.
DECODE_STEPS_D is a fixed decode budget, identical for every method, dataset and image, so a method that
stops after a different number of tokens cannot move it.
"""
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from effvar.model import TEMPLATE_KWARGS, build_inputs
from effvar.vision_cache import CACHE_ATTR

MAX_NEW_TOKENS = 8
DECODE_STEPS_D = MAX_NEW_TOKENS - 1  # = 7
SEED = 0
WARMUP_RUNS = 2


class FirstCallTimer(LogitsProcessor):
    def __init__(self):
        self.t_first = None

    def __call__(self, input_ids, scores):
        if self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()
        return scores


@contextmanager
def vision_timer(model):
    """Accumulate SigLIP + projector time. Same hook sites as the probe (scripts/probe_medgemma.py §E)."""
    state = {"t0": None, "ms": 0.0}

    def pre(mod, args, kwargs):
        torch.cuda.synchronize()
        state["t0"] = time.perf_counter()

    def post(mod, args, out):
        torch.cuda.synchronize()
        state["ms"] += (time.perf_counter() - state["t0"]) * 1e3

    h1 = model.model.vision_tower.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = model.model.multi_modal_projector.register_forward_hook(post)
    try:
        yield state
    finally:
        h1.remove()
        h2.remove()


@dataclass
class Generation:
    input_len: int
    new_ids: list[int]
    text: str
    first_logits: torch.Tensor  # [vocab], float32, CPU
    vision_ms: float | None     # None on the untimed path (tuning runs)
    prefill_ms: float | None
    decode_ms_per_token: float | None
    e2e_ms: float | None
    peak_vram_gib: float

    @property
    def lm_prefill_ms(self) -> float | None:
        if self.prefill_ms is None or self.vision_ms is None:
            return None
        return self.prefill_ms - self.vision_ms

    @property
    def lm_ms(self) -> float | None:
        if self.decode_ms_per_token is None or self.lm_prefill_ms is None:
            return None
        return self.lm_prefill_ms + DECODE_STEPS_D * self.decode_ms_per_token

    def timings(self) -> dict:
        return {"vision_ms": self.vision_ms, "prefill_ms": self.prefill_ms, "lm_prefill_ms": self.lm_prefill_ms,
                "decode_ms_per_token": self.decode_ms_per_token, "lm_ms": self.lm_ms, "e2e_ms": self.e2e_ms,
                "peak_vram_gib": self.peak_vram_gib}

    def digest(self) -> str:
        """Identity of a generation: the emitted ids plus the exact first-token logits (the same sha1 as
        `digest()` in scripts/ab_rounds.py and scripts/check_ablate.py). Computed after the timed region."""
        h = hashlib.sha1()
        h.update(repr(list(self.new_ids)).encode())
        h.update(self.first_logits.detach().to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()


def _generate(model, processor, method, inputs, max_new_tokens, timer=None):
    logits_processor = LogitsProcessorList([timer]) if timer is not None else None
    with torch.inference_mode(), method.activate(model, inputs):
        return model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                              return_dict_in_generate=True, output_logits=True,
                              logits_processor=logits_processor)


def run_untimed(model, processor, method, spec, image, max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
    """Greedy generation without timing. The only path allowed to use a vision cache."""
    inputs = build_inputs(processor, spec.messages(image), model.device)
    n_in = inputs["input_ids"].shape[-1]
    torch.cuda.reset_peak_memory_stats()
    out = _generate(model, processor, method, inputs, max_new_tokens)
    new_ids = out.sequences[0, n_in:].tolist()
    return Generation(input_len=n_in, new_ids=new_ids,
                      text=processor.decode(new_ids, skip_special_tokens=True),
                      first_logits=out.logits[0][0].float().cpu(),
                      vision_ms=None, prefill_ms=None, decode_ms_per_token=None, e2e_ms=None,
                      peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)


def run_one(model, processor, method, spec, image, max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
    """The timed path. Refuses to run against a cached vision tower: the timings would be fiction."""
    if getattr(model, CACHE_ATTR, None) is not None:
        raise RuntimeError("a vision cache is attached to this model; timing it would be meaningless. "
                           "Use effvar.generation.run_untimed for cached (tuning) runs.")
    inputs = build_inputs(processor, spec.messages(image), model.device)
    n_in = inputs["input_ids"].shape[-1]
    timer = FirstCallTimer()
    torch.cuda.reset_peak_memory_stats()
    with vision_timer(model) as vision:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _generate(model, processor, method, inputs, max_new_tokens, timer)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
    new_ids = out.sequences[0, n_in:].tolist()
    n_new = len(new_ids)
    return Generation(
        input_len=n_in,
        new_ids=new_ids,
        text=processor.decode(new_ids, skip_special_tokens=True),
        first_logits=out.logits[0][0].float().cpu(),
        vision_ms=vision["ms"],
        prefill_ms=(timer.t_first - t0) * 1e3,
        decode_ms_per_token=(t_end - timer.t_first) * 1e3 / (n_new - 1) if n_new > 1 else None,
        e2e_ms=(t_end - t0) * 1e3,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
    )


@dataclass
class BatchGeneration:
    """One timed generate() over B distinct images (measurement 3). Timings are for the WHOLE batch."""
    batch: int
    input_len: int
    new_ids: list[list[int]]      # per row; rows that stopped early carry padding after their EOS
    first_logits: torch.Tensor    # [B, vocab], float32, CPU
    vision_ms: float
    prefill_ms: float
    decode_ms_per_token: float | None
    e2e_ms: float
    peak_vram_gib: float

    @property
    def n_steps(self) -> int:
        return len(self.new_ids[0])


def build_batch_inputs(processor, spec, images, device):
    """B images of ONE dataset in one batch. The prompt is identical per dataset and the image block is a
    fixed 256 placeholder tokens, so every row has the same ids and length: no padding, one image span."""
    inputs = processor.apply_chat_template([spec.messages(im) for im in images], **TEMPLATE_KWARGS)
    ids = inputs["input_ids"]
    if not (bool((ids == ids[:1]).all()) and bool(inputs["attention_mask"].all())):
        raise ValueError("batch rows differ or are padded; batch images from one dataset only")
    return inputs.to(device, dtype=torch.bfloat16)


def run_batch(model, processor, method, spec, images, max_new_tokens: int = MAX_NEW_TOKENS) -> BatchGeneration:
    """`run_one` for a batch: same timing boundaries (vision hooks, first logits-processor call)."""
    if getattr(model, CACHE_ATTR, None) is not None:
        raise RuntimeError("a vision cache is attached to this model; timing it would be meaningless.")
    inputs = build_batch_inputs(processor, spec, images, model.device)
    n_in = inputs["input_ids"].shape[-1]
    timer = FirstCallTimer()
    torch.cuda.reset_peak_memory_stats()
    with vision_timer(model) as vision:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _generate(model, processor, method, inputs, max_new_tokens, timer)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
    new_ids = out.sequences[:, n_in:].tolist()
    n_steps = len(new_ids[0])
    return BatchGeneration(
        batch=len(images), input_len=n_in, new_ids=new_ids,
        first_logits=out.logits[0].float().cpu(),
        vision_ms=vision["ms"],
        prefill_ms=(timer.t_first - t0) * 1e3,
        decode_ms_per_token=(t_end - timer.t_first) * 1e3 / (n_steps - 1) if n_steps > 1 else None,
        e2e_ms=(t_end - t0) * 1e3,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
    )


def warmup(model, processor, method, spec, image, runs: int = WARMUP_RUNS) -> None:
    for _ in range(runs):
        run_one(model, processor, method, spec, image)
