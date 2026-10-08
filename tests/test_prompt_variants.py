import pytest
from PIL import Image

from effvar import FLAGS
from effvar.model import TEMPLATE_KWARGS, load_processor
from effvar.prompt_variants import reversed_spec, variant_spec
from effvar.prompts import ANSWER_PREFIX, get_spec
from effvar.scoring import option_token_sets, parse_answer


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.mark.parametrize("flag", FLAGS)
def test_frozen_variant_is_the_frozen_spec(flag):
    assert variant_spec("frozen", flag) == get_spec(flag)
    assert variant_spec("frozen", flag).text() == get_spec(flag).text()


@pytest.mark.parametrize("flag", FLAGS)
def test_reversed_letters_run_in_listing_order_with_a_on_the_last_class(flag):
    frozen, rev = get_spec(flag), reversed_spec(flag)
    k = len(frozen.labels)
    assert rev.labels == tuple(frozen.labels[k - 1 - c] for c in range(k))  # letter of class c
    assert rev.option_texts == frozen.option_texts                           # still class-indexed
    lines = rev.text().split("\n")
    opts = lines[lines.index("Options:") + 1: lines.index("Options:") + 1 + k]
    assert opts == [f"{frozen.labels[i]}. {frozen.option_texts[k - 1 - i]}" for i in range(k)]
    assert rev.instruction == frozen.instruction          # "(A-E)", not "(E-A)"
    # nothing but the option block differs from the frozen prompt
    f_lines = frozen.text().split("\n")
    assert [a for a, b in zip(lines, f_lines) if a != b] == [o for o in opts if o not in f_lines]
    assert len(lines) == len(f_lines)


def test_reversed_pneumonia_text():
    assert "Options:\nA. pneumonia\nB. normal\nAnswer with a single letter (A-B)" in reversed_spec("pneumoniamnist").text()


@pytest.mark.parametrize("flag", FLAGS)
def test_reversed_parse_maps_letters_back_to_classes(flag):
    rev = reversed_spec(flag)
    k = len(rev.labels)
    assert parse_answer("A", rev, [0.0] * k) == (k - 1, "label")
    assert parse_answer(rev.labels[0], rev, [0.0] * k) == (0, "label")
    assert parse_answer(rev.option_texts[1], rev, [0.0] * k)[0] == 1
    p = [0.0] * k
    p[2 % k] = 1.0
    assert parse_answer("I cannot tell", rev, p) == (2 % k, "fallback")


@pytest.mark.parametrize("flag", FLAGS)
def test_reversed_token_sets_are_class_aligned(processor, flag):
    frozen = option_token_sets(processor.tokenizer, get_spec(flag))
    rev = option_token_sets(processor.tokenizer, reversed_spec(flag))
    assert rev == frozen[::-1]   # class c scored on the tokens of the letter it was shown with


@pytest.mark.parametrize("flag", FLAGS)
def test_reversed_rendered_turn_is_prefilled(processor, flag):
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    kwargs = {**TEMPLATE_KWARGS, "tokenize": False, "return_dict": False, "return_tensors": None}
    text = processor.apply_chat_template(reversed_spec(flag).messages(image), **kwargs)
    assert text.endswith(ANSWER_PREFIX)
    assert reversed_spec(flag).text() in text


def test_unknown_variant():
    with pytest.raises(ValueError):
        variant_spec("shuffled", "dermamnist")
