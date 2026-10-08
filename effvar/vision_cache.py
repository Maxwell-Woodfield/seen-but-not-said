"""Projected image-feature cache. **Tuning runs only — never a timed run.**

SigLIP + the projector are ~70% of prefill and are identical for every method (they are never edited),
so a hyperparameter sweep can compute them once per image and reuse them. That makes every latency
number meaningless, so the cache is deliberately hard to misuse:

- `effvar.generation.run_one` (the timed path) raises if a cache is attached to the model;
- `scripts/run_eval.py --cache-vision` switches to the untimed path and refuses to write outside
  `results/tune/`;
- every run records `vision_cache: true/false` in both `config.yaml` and `metrics.json`.

The key covers everything that could change the features: model id and revision, dtype, the vision
attention implementation, the image-processor config and the image token count, plus the
dataset/split/index of the image itself. Any difference is a miss, and a miss recomputes.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from effvar import MODEL_ID, ROOT

CACHE_DIR = ROOT / "data" / "cache" / "vision"
CACHE_ATTR = "_effvar_vision_cache"


class _VisionOutput:
    """The one field Gemma3Model.forward reads back from get_image_features."""

    def __init__(self, pooler_output: torch.Tensor):
        self.pooler_output = pooler_output


def cache_key(model, processor) -> dict:
    cfg = model.config
    revision = getattr(cfg, "_commit_hash", None)
    if not revision:
        raise RuntimeError("no model revision on the config; refusing to cache against an unknown checkpoint")
    vision_attn = getattr(cfg.vision_config, "_attn_implementation", None)
    if not vision_attn:
        raise RuntimeError("vision attention implementation is unset; refusing to cache")
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "dtype": str(model.dtype),
        "vision_attn": vision_attn,
        "image_processor_sha256": hashlib.sha256(
            processor.image_processor.to_json_string().encode()).hexdigest(),
        "mm_tokens_per_image": int(cfg.mm_tokens_per_image),
    }


class VisionCache:
    """One directory per key; one file per (dataset, split, index)."""

    def __init__(self, key: dict, root: Path = CACHE_DIR):
        self.key = key
        self.digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        self.dir = root / self.digest
        self.dir.mkdir(parents=True, exist_ok=True)
        key_file = self.dir / "key.json"
        if not key_file.exists():
            key_file.write_text(json.dumps(key, indent=2, sort_keys=True))
        self.hits = self.misses = 0
        self._slot: tuple[str, str, int] | None = None

    @classmethod
    def for_model(cls, model, processor, root: Path = CACHE_DIR) -> "VisionCache":
        return cls(cache_key(model, processor), root=root)

    def path(self, flag: str, split: str, index: int) -> Path:
        return self.dir / f"{flag}_{split}_{index}.pt"

    @contextmanager
    def sample(self, flag: str, split: str, index: int):
        """Scope the cache to one image. Outside a sample the cache computes and stores nothing."""
        self._slot = (flag, split, index)
        try:
            yield
        finally:
            self._slot = None

    @contextmanager
    def attached(self, model):
        inner = model.model
        original = inner.get_image_features

        def cached_get_image_features(pixel_values, **kwargs):
            if self._slot is None:
                return original(pixel_values, **kwargs)
            path = self.path(*self._slot)
            if path.exists():
                self.hits += 1
                stored = torch.load(path, map_location="cpu", weights_only=True)
                return _VisionOutput(stored.to(device=pixel_values.device, dtype=model.dtype))
            out = original(pixel_values, **kwargs)
            self.misses += 1
            torch.save(out.pooler_output.detach().cpu(), path)
            return out

        inner.get_image_features = cached_get_image_features
        setattr(model, CACHE_ATTR, self)
        try:
            yield self
        finally:
            inner.get_image_features = original
            if hasattr(model, CACHE_ATTR):
                delattr(model, CACHE_ATTR)

    def stats(self) -> dict:
        return {"key_digest": self.digest, "hits": self.hits, "misses": self.misses}
