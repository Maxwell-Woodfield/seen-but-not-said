"""The vision cache must be exact, and must be impossible to use while timing.

Loads the model, so it is the slow test in the suite (one load, three images).
"""
import pytest
import torch

from effvar.data import iter_samples, load_subset
from effvar.generation import run_one, run_untimed
from effvar.methods import make_method
from effvar.model import build_inputs, load_model, load_processor
from effvar.prompts import get_spec
from effvar.vision_cache import CACHE_ATTR, VisionCache, cache_key

FLAG, N = "retinamnist", 3


@pytest.fixture(scope="module")
def setup():
    processor = load_processor()
    model = load_model("sdpa")
    subset = load_subset(FLAG, "smoke")
    samples = list(iter_samples(subset, limit=N))
    return processor, model, subset, samples


def test_cached_features_are_bitwise_identical(setup, tmp_path):
    processor, model, subset, samples = setup
    spec = get_spec(FLAG)
    cache = VisionCache(cache_key(model, processor), root=tmp_path)

    for idx, _, image in samples:
        pixel_values = build_inputs(processor, spec.messages(image), model.device)["pixel_values"]
        with torch.inference_mode():
            fresh = model.model.get_image_features(pixel_values).pooler_output.clone()
        with cache.attached(model), cache.sample(FLAG, subset.split, idx):
            with torch.inference_mode():
                written = model.model.get_image_features(pixel_values).pooler_output.clone()
            with torch.inference_mode():   # second call: served from disk
                reloaded = model.model.get_image_features(pixel_values).pooler_output.clone()
        assert torch.equal(fresh, written)
        assert torch.equal(fresh, reloaded), "cached features differ from freshly computed ones"
    assert cache.hits == N and cache.misses == N


def test_cached_generation_matches_uncached(setup, tmp_path):
    processor, model, subset, samples = setup
    spec = get_spec(FLAG)
    method = make_method("base", attn="sdpa")
    cache = VisionCache(cache_key(model, processor), root=tmp_path)

    for idx, _, image in samples:
        plain = run_untimed(model, processor, method, spec, image)
        with cache.attached(model):
            with cache.sample(FLAG, subset.split, idx):
                run_untimed(model, processor, method, spec, image)     # populate
            with cache.sample(FLAG, subset.split, idx):
                cached = run_untimed(model, processor, method, spec, image)  # serve
        assert cached.new_ids == plain.new_ids
        assert torch.equal(cached.first_logits, plain.first_logits)


def test_timed_path_refuses_a_cache(setup, tmp_path):
    processor, model, subset, samples = setup
    spec = get_spec(FLAG)
    method = make_method("base", attn="sdpa")
    cache = VisionCache(cache_key(model, processor), root=tmp_path)
    with cache.attached(model):
        assert getattr(model, CACHE_ATTR, None) is cache
        with pytest.raises(RuntimeError, match="vision cache"):
            run_one(model, processor, method, spec, samples[0][2])
    assert getattr(model, CACHE_ATTR, None) is None  # detached again


def test_key_covers_what_changes_the_features(setup, tmp_path):
    processor, model, _, _ = setup
    key = cache_key(model, processor)
    assert set(key) == {"model_id", "revision", "dtype", "vision_attn", "image_processor_sha256",
                        "mm_tokens_per_image"}
    assert all(key.values())
    base = VisionCache(key, root=tmp_path).digest
    for field, value in [("revision", "deadbeef"), ("dtype", "torch.float16"),
                         ("vision_attn", "eager"), ("image_processor_sha256", "0" * 64),
                         ("mm_tokens_per_image", 128), ("model_id", "other/model")]:
        assert VisionCache({**key, field: value}, root=tmp_path).digest != base, field
