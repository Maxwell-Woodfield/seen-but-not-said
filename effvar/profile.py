"""Opt-in region profiler for the VAR attention path (notes: P16 profile of R's LM overhead).

Where does `R`'s +202 ms of LM time per image go? The regions below split it:

- `qk`     -- Q@K, scaling, softcap, mask and softmax. This is the part SDPA would replace, so it is
              the ceiling for "compute the gate from narrow attention columns instead of full eager".
- `detect` -- DimProspector (`sink_mask`) in the decoder-layer pre-hook. Prefill only.
- `prep`   -- `split_sinks` + `visual_sinks`, every forward. Added 2026-09-20: this ran in no region
              before, so its cost was silently folded into `gate` and inflated the gate's per-call-site
              figure. `split_sinks` recomputes `torch.isin` on every forward from a set fixed at prefill.
- `gate`   -- HeadFork (`head_gate`), every forward.
- `edit`   -- `redistribute`, every forward where the gate keeps a row.
- `av`     -- attn @ V, which SDPA also pays, so it is not part of the overhead.

Timing uses CUDA events, which are recorded asynchronously and read once per image, so the run is not
serialised the way `torch.cuda.synchronize()` around each region would serialise it. Events still cost
something to record: **a profiled run's absolute latency is inflated and must never be quoted as a
latency result.** Only the shares are meaningful, and `run_eval.py --profile` writes
`profile: true` into the config so a profiled run cannot be mistaken for a timed one.

Anything in the LM time that these regions do not account for is Python, hook dispatch and kernel-launch
overhead -- which at batch 1 on a 3060 is not assumed small.

`HostTimer` (2026-09-20) measures the same regions with `perf_counter` and no CUDA events at all, to
turn the "~73% host-side" remainder into a measured number. Two things must be said whenever its
output is quoted:

- **It costs ~0.2 us per region entry and records nothing on the device**, so unlike `RegionTimer` it
  barely perturbs the run. That claim is itself checked: `R_host - R` is carried as an arm of the
  timing ladder and must contain 0.
- **Host time in a region that contains a blocking device->host sync includes the wait for the queue to
  drain.** `detect` (`torch.nonzero`), `gate` and `edit` (`bool(...)`/`int(...)`) all do; `qk`, `av` and
  `prep` do not. So host ms for those three is an UPPER BOUND on host-side work, and every table marks
  them with a dagger.
"""
import time
from contextlib import contextmanager

import torch

REGIONS = ("qk", "av", "detect", "prep", "gate", "edit")

# Regions whose host time absorbs a blocking device->host sync; reported with a dagger.
SYNC_REGIONS = ("detect", "gate", "edit")


class RegionTimer:
    """Accumulates GPU time per (region, layer, phase). `flush()` syncs once and folds the events in."""

    def __init__(self) -> None:
        self.totals: dict[tuple[str, int, str], float] = {}
        self.counts: dict[tuple[str, int, str], int] = {}
        self._pending: list[tuple[tuple[str, int, str], torch.cuda.Event, torch.cuda.Event]] = []
        self._pool: list[torch.cuda.Event] = []

    def _event(self) -> torch.cuda.Event:
        return self._pool.pop() if self._pool else torch.cuda.Event(enable_timing=True)

    @contextmanager
    def region(self, name: str, layer: int, phase: str):
        start, end = self._event(), self._event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append(((name, layer, phase), start, end))

    def flush(self) -> None:
        """Read every pending pair. Called once per image, outside the region being measured."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for key, start, end in self._pending:
            self.totals[key] = self.totals.get(key, 0.0) + start.elapsed_time(end)
            self.counts[key] = self.counts.get(key, 0) + 1
            self._pool.append(start)
            self._pool.append(end)
        self._pending.clear()

    def summary(self) -> dict:
        """{region: {phase: ms}, 'by_layer': {region: {layer: ms}}} -- totals over the images so far."""
        out: dict = {r: {} for r in REGIONS}
        by_layer: dict = {r: {} for r in REGIONS}
        for (name, layer, phase), ms in self.totals.items():
            out[name][phase] = out[name].get(phase, 0.0) + ms
            by_layer[name][str(layer)] = by_layer[name].get(str(layer), 0.0) + ms
        return {"kind": "gpu", "ms": out, "by_layer": by_layer,
                "calls": {r: sum(c for (n, _, _), c in self.counts.items() if n == r) for r in REGIONS}}


class HostTimer:
    """CPU time per (region, layer, phase) via `perf_counter`. No CUDA events, no sync.

    Duck-types `RegionTimer` (`region`/`flush`/`summary`) so `maybe()`, `VarState.timer` and
    `evaluate.py`'s flush/summary calls work unchanged. Like `_Null` it avoids a fresh generator per
    region per layer per forward -- one reusable object and a small explicit stack instead, since the
    whole point is to not disturb what is being measured. Regions never overlap in the VAR path, but
    the stack keeps it correct if that ever changes.
    """

    def __init__(self) -> None:
        self.totals: dict[tuple[str, int, str], float] = {}
        self.counts: dict[tuple[str, int, str], int] = {}
        self._stack: list[tuple[tuple[str, int, str], float]] = []

    def region(self, name: str, layer: int, phase: str):
        self._key = (name, layer, phase)
        return self

    def __enter__(self):
        self._stack.append((self._key, time.perf_counter()))
        return None

    def __exit__(self, *exc):
        key, t0 = self._stack.pop()
        ms = (time.perf_counter() - t0) * 1e3
        self.totals[key] = self.totals.get(key, 0.0) + ms
        self.counts[key] = self.counts.get(key, 0) + 1
        return False

    def flush(self) -> None:
        """Nothing is pending -- host time is folded in as each region exits."""

    def summary(self) -> dict:
        out: dict = {r: {} for r in REGIONS}
        by_layer: dict = {r: {} for r in REGIONS}
        for (name, layer, phase), ms in self.totals.items():
            out[name][phase] = out[name].get(phase, 0.0) + ms
            by_layer[name][str(layer)] = by_layer[name].get(str(layer), 0.0) + ms
        return {"kind": "host", "sync_regions": list(SYNC_REGIONS), "ms": out, "by_layer": by_layer,
                "calls": {r: sum(c for (n, _, _), c in self.counts.items() if n == r) for r in REGIONS}}


class _Null:
    """Reused no-op context. A fresh @contextmanager generator per region per layer per forward is
    ~500 objects an image, which is real time on the hot path; this costs one call and one branch."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def maybe(timer: "RegionTimer | None", name: str, layer: int, phase: str):
    """`timer.region(...)` when profiling, otherwise the shared no-op."""
    return _NULL if timer is None else timer.region(name, layer, phase)
