import pytest
import torch
from PIL import Image
from transformers import AutoConfig

from effvar import FLAGS, MODEL_ID
from effvar.model import TEMPLATE_KWARGS, load_processor
from effvar.prompts import ANSWER_PREFIX, get_spec
from effvar.scoring import option_token_sets


@pytest.fixture(scope="module")
def processor():
    return load_processor()


def _render(processor, spec) -> str:
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    kwargs = {**TEMPLATE_KWARGS, "tokenize": False, "return_dict": False, "return_tensors": None}
    return processor.apply_chat_template(spec.messages(image), **kwargs)


@pytest.mark.parametrize("flag", FLAGS)
def test_option_token_sets_disjoint(processor, flag):
    spec = get_spec(flag)
    sets = option_token_sets(processor.tokenizer, spec)
    assert len(sets) == len(spec.labels)
    assert all(sets)
    flat = [i for s in sets for i in s]
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("flag", FLAGS)
def test_prompt_lists_every_option(flag):
    spec = get_spec(flag)
    text = spec.text()
    for lab, opt in zip(spec.labels, spec.option_texts):
        assert f"{lab}. {opt}" in text


@pytest.mark.parametrize("flag", FLAGS)
def test_labels_are_letters_and_instruction_matches(flag):
    spec = get_spec(flag)
    assert spec.labels == tuple(chr(ord("A") + i) for i in range(len(spec.labels)))
    assert f"({spec.labels[0]}-{spec.labels[-1]})" in spec.instruction
    assert spec.instruction in spec.text()


def test_retina_labels_spell_out_grades():
    spec = get_spec("retinamnist")
    assert spec.labels == ("A", "B", "C", "D", "E")
    for i, opt in enumerate(spec.option_texts):
        assert opt.startswith(f"grade {i} ")


@pytest.mark.parametrize("flag", FLAGS)
def test_rendered_turn_is_prefilled(processor, flag):
    """The assistant turn continues, so generation starts straight after "Answer:"."""
    text = _render(processor, get_spec(flag))
    assert text.endswith(ANSWER_PREFIX)
    assert text.count("<start_of_turn>") == 2
    assert text.count("<end_of_turn>") == 1  # the user turn only


@pytest.mark.parametrize("flag", FLAGS)
def test_chat_template_single_image_span(processor, flag):
    config = AutoConfig.from_pretrained(MODEL_ID)
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    ids = processor.apply_chat_template(get_spec(flag).messages(image), **TEMPLATE_KWARGS)["input_ids"][0]
    pos = (ids == config.image_token_index).nonzero().flatten()
    assert pos.numel() == config.mm_tokens_per_image
    assert (pos == pos[0] + torch.arange(pos.numel())).all()
