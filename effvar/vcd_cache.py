"""Phase 7 card V1: the noised-vision cache, kept apart from the clean one (guards 4 and 8). Tuning runs only.

- `NoisedVisionCache`: key = `vision_cache.cache_key` + {"noise": {schedule, step}}, so a different directory from the
  clean cache. Files `{flag}_{split}_{index}_s{seed}.pt`, payload {features, step, seed}, checked on every load. It
  refuses a directory whose key.json is not exactly its key, and it refuses to compute or serve outside a sample.
- `open_clean_cache`: `ReadOnlyCleanCache`, a `VisionCache` that serves hits as usual and raises on a miss (the clean
  re-run never writes the clean cache). It is opened only after refusing a noised key.json, seed-named files, or a
  digest other than the one the saved clean runs used.
- `noised_dirs_state`: the noised directories under the cache root and their file counts (guards 3 and 8).
- `CleanFeatureProbe` (the uncached smoke): compares each freshly computed projector output with that image's clean
  cached features. It reads the clean files and never writes them.
- `dir_state`: file count and key.json hash of a cache directory (guard 8: the clean cache is never written).

`effvar/vision_cache.py` itself is unchanged.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from effvar.vcd import SCHEDULE
from effvar.vision_cache import CACHE_ATTR, CACHE_DIR, VisionCache, _VisionOutput, cache_key

SEED_FILE_GLOB = "*_s[0-9]*.pt"


def key_digest(key: dict) -> str:
    """The directory name `VisionCache` derives from a key."""
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def noised_key(clean_key: dict, step: int) -> dict:
    if "noise" in clean_key:
        raise ValueError("refusing to derive a noised key from a key that already has 'noise'")
    return {**clean_key, "noise": {"schedule": SCHEDULE, "step": int(step)}}


def clean_file(cache_dir: Path, flag: str, split: str, index: int) -> Path:
    """The clean cache's filename for one image (`VisionCache.path`)."""
    return Path(cache_dir) / f"{flag}_{split}_{index}.pt"


def _read_key(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / "key.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class NoisedVisionCache:
    """One directory per (clean key, step); one file per (dataset, split, index, seed)."""

    def __init__(self, clean_key: dict, step: int, root: Path = CACHE_DIR):
        self.step = int(step)
        self.key = noised_key(clean_key, self.step)
        self.digest = key_digest(self.key)
        self.dir = Path(root) / self.digest
        stored = _read_key(self.dir)
        if stored is not None and stored != self.key:
            raise RuntimeError(f"noised cache refuses {self.dir}: its key.json is not this noised key "
                               f"({'a clean key' if 'noise' not in stored else stored['noise']})")
        self.dir.mkdir(parents=True, exist_ok=True)
        if stored is None:
            (self.dir / "key.json").write_text(json.dumps(self.key, indent=2, sort_keys=True), encoding="utf-8")
        self.hits = self.misses = 0
        self._slot: tuple[str, str, int, int] | None = None

    def path(self, flag: str, split: str, index: int, seed: int) -> Path:
        return self.dir / f"{flag}_{split}_{index}_s{seed}.pt"

    @contextmanager
    def sample(self, flag: str, split: str, index: int, seed: int):
        self._slot = (flag, split, int(index), int(seed))
        try:
            yield
        finally:
            self._slot = None

    def _check(self, payload, seed: int, path: Path) -> torch.Tensor:
        if not isinstance(payload, dict) or set(payload) != {"features", "step", "seed"}:
            raise RuntimeError(f"noised cache refuses {path}: not a noised payload (a clean entry?)")
        if payload["step"] != self.step or payload["seed"] != seed:
            raise RuntimeError(f"noised cache refuses {path}: payload (step {payload['step']}, seed {payload['seed']})"
                               f" != expected (step {self.step}, seed {seed})")
        return payload["features"]

    @contextmanager
    def attached(self, model):
        inner = model.model
        original = inner.get_image_features

        def noised_get_image_features(pixel_values, **kwargs):
            if self._slot is None:
                raise RuntimeError("noised cache attached outside a sample: refusing to compute or serve features")
            flag, split, index, seed = self._slot
            path = self.path(flag, split, index, seed)
            if path.exists():
                features = self._check(torch.load(path, map_location="cpu", weights_only=True), seed, path)
                self.hits += 1
                return _VisionOutput(features.to(device=pixel_values.device, dtype=model.dtype))
            out = original(pixel_values, **kwargs)
            self.misses += 1
            torch.save({"features": out.pooler_output.detach().cpu(), "step": self.step, "seed": seed}, path)
            return out

        inner.get_image_features = noised_get_image_features
        setattr(model, CACHE_ATTR, self)
        try:
            yield self
        finally:
            inner.get_image_features = original
            if hasattr(model, CACHE_ATTR):
                delattr(model, CACHE_ATTR)

    def stats(self) -> dict:
        return {"key_digest": self.digest, "step": self.step, "hits": self.hits, "misses": self.misses}


def check_clean_dir(cache_dir: Path, key: dict) -> None:
    """A clean run refuses a noised directory: key.json must exist, equal the clean key and carry no 'noise', and no
    seed-named (noised) file may sit in it."""
    cache_dir = Path(cache_dir)
    stored = _read_key(cache_dir)
    if stored is None:
        raise RuntimeError(f"no clean cache at {cache_dir} (a clean re-run reads the cache, it never populates it)")
    if "noise" in stored:
        raise RuntimeError(f"a clean run refuses a noised cache directory: {cache_dir}")
    if stored != key:
        raise RuntimeError(f"clean cache key mismatch at {cache_dir}")
    if next(cache_dir.glob(SEED_FILE_GLOB), None) is not None:
        raise RuntimeError(f"seed-named (noised) files in the clean cache {cache_dir}")


class ReadOnlyCleanCache(VisionCache):
    """The clean cache for the clean re-run: a hit is served exactly as `VisionCache` serves it; a miss raises
    instead of computing and writing (guard 8: hits 905 / 156, misses 0; the clean cache is never written). It must
    be built on an existing directory (check_clean_dir), so `VisionCache.__init__` writes nothing either."""

    @contextmanager
    def attached(self, model):
        inner = model.model
        original = inner.get_image_features

        def read_only_get_image_features(pixel_values, **kwargs):
            if self._slot is None:
                raise RuntimeError("clean re-run cache attached outside a sample")
            path = self.path(*self._slot)
            if not path.exists():
                self.misses += 1
                raise RuntimeError(f"clean re-run: cache miss at {path}; the clean cache is read, never written")
            stored = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(stored, torch.Tensor):
                raise RuntimeError(f"clean re-run refuses {path}: not a clean entry")
            self.hits += 1
            return _VisionOutput(stored.to(device=pixel_values.device, dtype=model.dtype))

        inner.get_image_features = read_only_get_image_features
        setattr(model, CACHE_ATTR, self)
        try:
            yield self
        finally:
            inner.get_image_features = original
            if hasattr(model, CACHE_ATTR):
                delattr(model, CACHE_ATTR)


def clean_cache_for_key(key: dict, expect_digest: str, root: Path = CACHE_DIR) -> ReadOnlyCleanCache:
    if "noise" in key:
        raise RuntimeError("a clean run refuses a noised key")
    digest = key_digest(key)
    if digest != expect_digest:
        raise RuntimeError(f"clean cache digest {digest} != {expect_digest} used by the saved clean runs")
    check_clean_dir(Path(root) / digest, key)
    return ReadOnlyCleanCache(key, root=root)


def open_clean_cache(model, processor, expect_digest: str, root: Path = CACHE_DIR) -> ReadOnlyCleanCache:
    return clean_cache_for_key(cache_key(model, processor), expect_digest, root)


def noised_dirs_state(root: Path = CACHE_DIR) -> dict:
    """{digest: {"step", "n_files"}} for every noised cache directory under the cache root (guards 3 and 8: the smoke
    reads and writes no noised cache; G4's first noised passes start from an empty one)."""
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        key = _read_key(d)
        if key is not None and "noise" in key:
            out[d.name] = {"step": key["noise"]["step"],
                           "n_files": sum(1 for e in os.scandir(d) if e.name.endswith(".pt"))}
    return out


class CleanFeatureProbe:
    """Smoke only (no cache attached). After each fresh projector output, it loads that image's clean cached features
    and records max |delta| in fp32. It is read-only and is not a cache: CACHE_ATTR stays unset."""

    def __init__(self, clean_dir: Path):
        self.dir = Path(clean_dir)
        self._slot: tuple[str, str, int] | None = None
        self.calls = 0
        self.max_abs: float | None = None

    @contextmanager
    def sample(self, flag: str, split: str, index: int):
        self._slot, self.calls, self.max_abs = (flag, split, int(index)), 0, None
        try:
            yield
        finally:
            self._slot = None

    @contextmanager
    def attached(self, model):
        inner = model.model
        original = inner.get_image_features

        def probed_get_image_features(pixel_values, **kwargs):
            out = original(pixel_values, **kwargs)
            if self._slot is not None:
                ref = torch.load(clean_file(self.dir, *self._slot), map_location="cpu", weights_only=True)
                fresh = out.pooler_output.detach().float().cpu()
                if fresh.shape != ref.shape:
                    raise RuntimeError(f"probe: feature shape {tuple(fresh.shape)} != cached {tuple(ref.shape)}")
                self.max_abs = float((fresh - ref.float()).abs().max())
                self.calls += 1
            return out

        inner.get_image_features = probed_get_image_features
        try:
            yield self
        finally:
            inner.get_image_features = original

    def stats(self) -> None:
        return None


def dir_state(cache_dir: Path) -> dict:
    """File count (*.pt) and key.json sha256 of a cache directory."""
    cache_dir = Path(cache_dir)
    key = cache_dir / "key.json"
    n = sum(1 for e in os.scandir(cache_dir) if e.name.endswith(".pt")) if cache_dir.exists() else 0
    return {"dir": cache_dir.name, "n_files": n,
            "key_json_sha256": hashlib.sha256(key.read_bytes()).hexdigest() if key.exists() else None}
