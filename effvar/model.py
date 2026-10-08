"""Load MedGemma-1.5-4b-it and locate image tokens. Counts come from model.config, never hardcoded."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from effvar import MODEL_ID

# Registered by effvar.methods.var (imported via effvar.methods) on the AttentionInterface.
VAR_ATTN_NAME = "effvar_var_eager"

# Vision always stays on SDPA: eager SigLIP is ~7x slower (notes/medgemma_probe.md §E).
ATTN_MODES = {
    "sdpa": "sdpa",
    "text_eager": {"text_config": "eager", "vision_config": "sdpa"},
    "var_eager": {"text_config": VAR_ATTN_NAME, "vision_config": "sdpa"},
}


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def load_model(attn: str):
    if attn not in ATTN_MODES:
        raise ValueError(f"attn must be one of {sorted(ATTN_MODES)}, got {attn!r}")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa").eval()
    if attn != "sdpa":
        model.set_attn_implementation(ATTN_MODES[attn])
    return model


def image_span(input_ids: torch.Tensor, config) -> tuple[int, int]:
    """(start, P) of the single contiguous image-token block in a [1, T] or [T] input_ids tensor."""
    ids = input_ids.reshape(-1)
    pos = torch.nonzero(ids == config.image_token_index).flatten()
    if pos.numel() == 0:
        raise ValueError("no image tokens in input_ids")
    start, P = int(pos[0]), int(pos.numel())
    if not bool((pos == torch.arange(start, start + P, device=pos.device)).all()):
        raise ValueError("image tokens are not contiguous")
    if P != config.mm_tokens_per_image:
        raise ValueError(f"expected {config.mm_tokens_per_image} image tokens, got {P}")
    return start, P


# The assistant turn is prefilled (effvar.prompts.ANSWER_PREFIX), so the rendered text continues that turn
# instead of opening a new one: it ends with "Answer:" and no <end_of_turn>.
TEMPLATE_KWARGS = dict(add_generation_prompt=False, continue_final_message=True, tokenize=True,
                       return_dict=True, return_tensors="pt")


def build_inputs(processor, messages, device):
    return processor.apply_chat_template(messages, **TEMPLATE_KWARGS).to(device, dtype=torch.bfloat16)
