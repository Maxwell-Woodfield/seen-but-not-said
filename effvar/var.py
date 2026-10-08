"""VAR mechanism (Kang et al., ICLR 2025), ported to MedGemma in the PAPER convention.

Reference behaviour: `VisAttnSink/src/logic/logic.py:104-212` (DimProspector, HeadFork,
VARProcessor.attn_redist). Conventions are inverted between the paper and that code
(`notes/derivation_rank1.md` §0); everything here is the paper's:

    sink weights x (1 - p)        budget = p * (S_v + S_t)        head kept iff non-sink share >= rho

so `p = 0` is the identity and `p = 1 - p_code`, `rho = 1 - rho_code`. The released config's
p_code 0.6 / rho_code 0.5 are p = 0.4 / rho = 0.5 here. `tau` and `summ` do not flip.

These functions are model-free: they take post-softmax attention and hidden states, so the vectorised
edit can be checked against a literal transcription of `attn_redist`'s loop (validation_plan.md G1).
"""
from dataclasses import dataclass, field

import torch

from effvar.ablate import const_sink_tensor
from effvar.profile import maybe

RMS_EPS = 1e-6
GATE_EPS = 1e-6  # logic.py:129 adds this PER IMAGE COLUMN, so the denominator gains P * GATE_EPS


@dataclass(frozen=True)
class VarConfig:
    """Paper-convention knobs. Defaults: validation_plan.md §3."""

    p: float = 0.4
    rho: float = 0.5
    tau: float = 10.0
    summ: float = 0.1
    dims: tuple[int, ...] = (1313, 2511)
    zero_dim: int | None = 443     # excluded from the RMS; None = the literal paper detector
    first_layer: int = 2           # DimProspector fills layers >= this one only
    except_last_layer: bool = False  # False reproduces the released code, True the paper
    sink_source: str = "detector"  # "detector" = DimProspector; "attention" = var_attn's I_v
    attn_sink_mult: float = 10.0   # var_attn: >= this many times the mean image-token attention
    record_delta: bool = False     # H2 instrumentation: per-layer ||delta_o||. Costs a matmul per
    # edited layer and forces a host sync, so it is OFF by default and never runs on a timed path.
    record_sinks: bool = False     # log the per-layer sink SETS (prefill only). No earlier run ever
    # persisted these, which is why across-layer sink stability could not be checked from the logs.
    profile: bool = False          # P16: per-region CUDA-event timing (effvar/profile.py). Inflates
    # absolute latency, so a profiled run is never a latency result; only the shares are meaningful.
    layers: tuple[int, ...] | None = None  # effvar_rank1_topm: restrict the whole mechanism to these
    # layers. None = every layer from first_layer on. A layer outside the set skips the DimProspector
    # *and* the gate, not just the edit, which is where the saving comes from (notes/cost_model.md).
    profile_host: bool = False     # per-region CPU time (effvar.profile.HostTimer). Mutually exclusive
    # with `profile`; unlike it, near-zero perturbation, which is what makes the host share measurable.
    ablate: object | None = None   # effvar.ablate.Ablation. TIMING ONLY -- never set on an accuracy
    # run, and unreachable from run_eval.py's CLI by design (see effvar/ablate.py).
    name: str = "var"

    def params(self) -> dict:
        out = {"p": self.p, "rho": self.rho, "tau": self.tau, "summ": self.summ, "dims": list(self.dims),
               "zero_dim": self.zero_dim, "first_layer": self.first_layer,
               "except_last_layer": self.except_last_layer, "sink_source": self.sink_source,
               "convention": "paper"}
        if self.record_delta:
            out["record_delta"] = True
        if self.profile:
            out["profile"] = True
        if self.record_sinks:
            out["record_sinks"] = True
        if self.layers is not None:
            out["layers"] = list(self.layers)
            out["m"] = len(self.layers)
        if self.sink_source == "attention":
            out["attn_sink_mult"] = self.attn_sink_mult
        if self.profile_host:
            out["profile_host"] = True
        if self.ablate is not None:  # makes an ablated run self-labelling in its own config.yaml
            out["ablate"] = self.ablate.params()
        return out


VAR_CONFIGS = {
    # `var`: the paper detector with dim 443 kept out of the RMS (medgemma_probe.md §B).
    "var": VarConfig(),
    # `var_literal`: the literal port, expected to be a no-op on MedGemma (framework.md §0).
    "var_literal": VarConfig(tau=20.0, dims=(443, 634, 2511, 1313), zero_dim=None, name="var_literal"),
    # `var_attn`: second reference; visual sinks from attention, text sinks from the same detector.
    "var_attn": VarConfig(sink_source="attention", name="var_attn"),
    # `effvar_rank1_topm`: R's mechanism and sink set, restricted to `layers` (framework.md §2).
    # Identical to `var` when `layers` is None, so the m-curve's top end is R itself.
    "effvar_rank1_topm": VarConfig(name="effvar_rank1_topm"),
}


def attention_sinks(attn: torch.Tensor, im: int, P: int, mult: float) -> torch.Tensor:
    """var_attn's I_v (framework.md §0): image tokens taking >= `mult` x the mean image-token
    attention from post-image queries, averaged over heads and queries. Absolute positions.

    Prefill only -- at decode there are no post-image rows yet to average, so the prefill set is
    reused for the rest of the sample, exactly as the detector's set is.

    `mult` must be < P or nothing can ever qualify: a single column's own contribution to the mean caps
    the achievable ratio at P. Here P = 256 and mult = 10 (probe §C measured 1.9-2.5 such sinks).
    """
    rows = attn[..., im + P:, im:im + P].float()       # [B, H, Q_post, P]
    if rows.shape[-2] == 0:
        return torch.empty(0, dtype=torch.long, device=attn.device)
    per_token = rows.mean(dim=(0, 1, 2))               # [P]
    threshold = mult * per_token.mean()
    return (torch.nonzero(per_token >= threshold).flatten() + im) if threshold > 0 else \
        torch.empty(0, dtype=torch.long, device=attn.device)


def sink_mask(hidden_states: torch.Tensor, cfg: VarConfig) -> torch.Tensor:
    """DimProspector (logic.py:99-110): token j is a sink iff max_{d in dims} |rmsnorm(x_j)_d| > tau.

    RMSNorm is weightless and in fp32, as in the reference; `zero_dim` is zeroed beforehand so one
    dominant dimension cannot swamp the norm. Returns a [B, T] bool mask.
    """
    x = hidden_states.float()
    if cfg.zero_dim is not None:
        x = x.clone()
        x[..., cfg.zero_dim] = 0.0
    variance = x.pow(2).mean(-1, keepdim=True)
    rms = (x * torch.rsqrt(variance + RMS_EPS)).abs()
    peak = rms[..., list(cfg.dims)].max(-1).values
    return peak > cfg.tau  # strict, as in the reference


def head_gate(attn: torch.Tensor, im: int, P: int, vis_idx: torch.Tensor, cfg: VarConfig) -> torch.Tensor:
    """HeadFork (logic.py:129-138). Keeps (batch, head, query) rows that are image-centric:

        non-sink share of image attention >= rho   and   image attention mass >= summ

    The first is the paper's form of the code's `portion <= rho_code`. Returns a [B, H, Q] bool mask.
    """
    img = attn[..., im:im + P].float()
    mass = img.sum(-1)
    sink_mass = attn[..., vis_idx].float().sum(-1)
    portion = sink_mass / (mass + P * GATE_EPS)
    return (portion <= 1.0 - cfg.rho) & (mass >= cfg.summ)


def redistribute(attn: torch.Tensor, im: int, P: int, vis_idx: torch.Tensor, text_idx: torch.Tensor,
                 edit: torch.Tensor, p: float, check_any: bool = True) -> torch.Tensor:
    """VARProcessor.attn_redist (logic.py:163-212), vectorised, paper convention.

    For every selected row (`edit`, [B, H, Q]): both sink groups shrink to (1 - p) of their weight and
    the whole freed budget `p * (S_v + S_t)` is spread over the **non-sink image** columns in proportion
    to their existing weight. Nothing returns to text; the row sum is preserved.

    With no visual sinks the layer is a no-op even when text sinks exist (logic.py:140-141). Rows whose
    non-sink image mass is 0 are skipped: the reference divides by zero there (derivation_rank1.md §7).
    Arithmetic is fp32; the result is cast back, so p = 0 reproduces the input bit for bit.

    `check_any=False` drops the `edit.any()` test, which is a blocking device->host sync. The only
    caller, `edit_attention`, has already established `keep.any()` before calling, so the test can
    never fire from there and dropping it is output-identical, so it passes False unless the timing-only
    `legacy` ablation is set. The default keeps it, so the model-free G1 tests still exercise the
    guarded path.
    """
    if vis_idx.numel() == 0 or (check_any and not bool(edit.any())):
        return attn

    a = attn.float()
    img = a[..., im:im + P]
    non_sink = img.clone()
    non_sink[..., vis_idx - im] = 0.0
    N = non_sink.sum(-1)                                   # [B, H, Q]
    budget = p * (a[..., vis_idx].sum(-1) + a[..., text_idx].sum(-1))

    out = a.clone()
    out[..., vis_idx] *= 1.0 - p
    if text_idx.numel():
        out[..., text_idx] *= 1.0 - p
    ratio = non_sink / N.clamp_min(torch.finfo(a.dtype).tiny).unsqueeze(-1)
    out[..., im:im + P] += budget.unsqueeze(-1) * ratio

    keep = (edit & (N > 0)).unsqueeze(-1)
    return torch.where(keep, out, a).to(attn.dtype)


def head_gate_rows(attn: torch.Tensor, im: int, P: int, vis_img: torch.Tensor, cfg: VarConfig) -> torch.Tensor:
    """`head_gate` with a per-row visual-sink MASK over the image block (`vis_img`, [B, P] bool) instead of
    one shared index list, for batches of distinct images (measurement 3). Same rule, per row."""
    img = attn[..., im:im + P].float()
    mass = img.sum(-1)
    sink_mass = (img * vis_img[:, None, None, :]).sum(-1)
    portion = sink_mass / (mass + P * GATE_EPS)
    return (portion <= 1.0 - cfg.rho) & (mass >= cfg.summ)


def redistribute_rows(attn: torch.Tensor, im: int, P: int, vis: torch.Tensor, sinks: torch.Tensor,
                      edit: torch.Tensor, p: float) -> torch.Tensor:
    """`redistribute` with per-row sink MASKS over the prefill positions: `vis` ([B, T0]) the visual sinks,
    `sinks` ([B, T0]) visual | text. Keys at decode extend past T0, but every sink and every image column
    lies below it. The caller gates rows without visual sinks out through `edit`.

    Same arithmetic as the index path, except that the masked sums reduce over all T0 (or P) columns
    rather than the gathered few, so the result can differ from `redistribute` in the last bits.
    """
    a = attn.float()
    T0 = sinks.shape[-1]
    img = a[..., im:im + P]
    non_sink = img.masked_fill(vis[:, None, None, im:im + P], 0.0)
    N = non_sink.sum(-1)                                   # [B, H, Q]
    s = sinks[:, None, None, :]
    head = a[..., :T0]
    budget = p * (head * s).sum(-1)

    out = a.clone()
    out[..., :T0] = torch.where(s, head * (1.0 - p), head)
    ratio = non_sink / N.clamp_min(torch.finfo(a.dtype).tiny).unsqueeze(-1)
    out[..., im:im + P] += budget.unsqueeze(-1) * ratio

    keep = (edit & (N > 0)).unsqueeze(-1)
    return torch.where(keep, out, a).to(attn.dtype)


def split_sinks(sinks: torch.Tensor, im: int, P: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sink positions -> (visual, text), matching logic.py:123 and :189-190."""
    vis = sinks[(sinks >= im) & (sinks < im + P)]
    text = sinks[~torch.isin(sinks, vis)]
    return vis, text


DELTA_PHASES = ("answer", "prefill", "decode")


def delta_summary(delta: dict) -> dict | None:
    """Per-image, per-layer, per-phase RMS norms from `VarState.delta`; None when nothing was recorded.

    Rank layers on `answer` (or `prefill`), never on `decode` -- see `VarState.record_delta_o`.
    """
    if not delta:
        return None
    out = {}
    for layer, buckets in sorted(delta.items()):
        row = {}
        for phase in DELTA_PHASES:
            acc = buckets.get(phase)
            if acc and acc[2]:
                sq, osq, n = acc
                row[f"{phase}_delta_o_rms"] = (sq / n) ** 0.5
                row[f"{phase}_o_rms"] = (osq / n) ** 0.5
        if row:
            out[str(layer)] = row
    return out


def sinks_summary(sink_sets: dict) -> dict | None:
    """Per-layer sink sets for one image: {layer: {vis, text, n_vis, n_text}}. None unless recorded.

    `vis`/`text` are absolute token indices, so they are directly comparable across layers and across
    images -- across layers answers "can one detection serve every layer" (`effvar_detect1`), across
    images answers whether a single fixed set would do, which is prior variant A.
    """
    if not sink_sets:
        return None
    return {str(ly): {"vis": s["vis"], "text": s["text"], "n_vis": len(s["vis"]), "n_text": len(s["text"])}
            for ly, s in sorted(sink_sets.items())}


def stats_summary(stats: dict) -> dict:
    """A per-image view of `VarState.stats`, read after generation so the timed path is untouched.

    `edit_events` counts (layer, forward) pairs that were edited, so prefill and every decode step each
    contribute: with L active layers and D decode steps the ceiling is L * (1 + D), not L. `n_vis` and
    `n_text` are this image's mean |I_v| and |I_t| over its edits, None when nothing was edited -- which
    is how `var_literal`'s no-op shows up (notes/status.md).

    A per-row (batched) state reports the same four numbers PER IMAGE -- means over the batch's rows of
    each row's own counts -- plus the per-row lists under `per_row`, so measurement 3 can check that
    batching does not change how much work R does (the batch curve must be workload-faithful).
    """
    pr = stats.get("per_row")
    if pr is not None:
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return float(sum(xs) / len(xs)) if xs else None
        return {"edit_events": mean(pr["edit_events"]), "rows_edited": mean(pr["rows_edited"]),
                "n_vis": mean(pr["n_vis"]), "n_text": mean(pr["n_text"]), "batch": len(pr["rows_edited"]),
                "per_row": pr}
    n_vis, n_text = stats["n_vis"], stats["n_text"]
    return {
        "edit_events": int(stats["layers_edited"]),
        "rows_edited": int(stats["rows_edited"]),
        "n_vis": float(sum(n_vis) / len(n_vis)) if n_vis else None,
        "n_text": float(sum(n_text) / len(n_text)) if n_text else None,
    }


@dataclass
class VarState:
    """Per-generation state: the image span and this sample's sink set (prefill only, then fixed)."""

    cfg: VarConfig
    im: int
    P: int
    n_layers: int
    sinks: dict[int, torch.Tensor] = field(default_factory=dict)       # DimProspector, per layer
    vis_sinks: dict[int, torch.Tensor] = field(default_factory=dict)   # var_attn only, per layer
    stats: dict = field(default_factory=lambda: {"layers_edited": 0, "rows_edited": 0, "n_vis": [], "n_text": []})
    delta: dict = field(default_factory=dict)  # layer -> {phase: [sum ||delta_o||^2, sum ||o||^2, n]}
    timer: object | None = None   # effvar.profile.RegionTimer / HostTimer, else None
    sink_sets: dict = field(default_factory=dict)  # layer -> {"vis": [...], "text": [...]}, cfg.record_sinks

    def __post_init__(self) -> None:
        # Hoist the timing-only knobs to plain attributes: the hot path then reads one local bool per
        # check instead of walking cfg.ablate, so an un-ablated run pays a branch and nothing more.
        ab = self.cfg.ablate
        self._stop = ab.stop_after if ab is not None else None
        self._detect_layers = ab.detect_layers if ab is not None else None
        self._const_sinks = ab.const_sinks if ab is not None else None
        # Lean is R's implementation since 2026-09-20; the timing-only `legacy` ablation restores the
        # first port's redundant host work. Both are bitwise identical (scripts/check_ablate.py).
        self._lean = not (ab is not None and ab.legacy)
        self._prep_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # split_sinks once per layer
        self._rows_cache: dict[int, torch.Tensor] = {}                       # row filter once per length
        self._rows_edited: torch.Tensor | None = None                        # reduced at finalise_stats
        self._per_row = bool(ab is not None and ab.per_row)
        self._keep_log: list[tuple[int, torch.Tensor]] = []  # per_row: (layer, keep) per edit, reduced later
        if self._per_row and (self.cfg.sink_source != "detector" or self.cfg.record_sinks
                              or self.cfg.record_delta):
            raise ValueError("per_row supports the detector-sink R only, without instrumentation")

    def active_layer(self, layer_idx: int) -> bool:
        if layer_idx < self.cfg.first_layer:
            return False
        if self.cfg.layers is not None and layer_idx not in self.cfg.layers:
            return False
        return not (self.cfg.except_last_layer and layer_idx == self.n_layers - 1)

    def observe(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        """DimProspector hook: prefill only, layers >= first_layer (llava.py:314-316)."""
        if not self.active_layer(layer_idx) or hidden_states.shape[1] <= 1:
            return
        if self._stop == "hook":  # timing only: the pre-hook still fires, the detector does not run
            return
        if hidden_states.shape[0] > 1 and not self._per_row:
            # `nonzero(mask)[:, 1]` would silently union the rows' sink sets.
            raise ValueError("batch > 1 needs Ablation(per_row=True): the index path unions sink sets")
        if self._detect_layers is not None and layer_idx not in self._detect_layers:
            # Timing only: stand the modal set in for the detector in this layer, from a tensor
            # allocated once per process. This is the layers-12-15 lever's shape, not a method.
            const = self._const_sinks.get(layer_idx)
            if const is not None:
                self.sinks[layer_idx] = const_sink_tensor(layer_idx, const, hidden_states.device)
            return
        with maybe(self.timer, "detect", layer_idx, "prefill"):
            mask = sink_mask(hidden_states, self.cfg)
            self.sinks[layer_idx] = mask if self._per_row else torch.nonzero(mask)[:, 1]
        if self.cfg.record_sinks:  # prefill only, after the detector; .tolist() syncs, so untimed runs only
            vis, text = split_sinks(self.sinks[layer_idx], self.im, self.P)
            self.sink_sets[layer_idx] = {"vis": vis.tolist(), "text": text.tolist()}

    def visual_sinks(self, layer_idx: int, attn: torch.Tensor, detector_vis: torch.Tensor) -> torch.Tensor:
        """I_v for this layer. `var` uses the detector's; `var_attn` reads this layer's own unedited
        prefill attention (framework.md §2) and reuses that set at decode."""
        if self.cfg.sink_source == "detector":
            return detector_vis
        if attn.shape[-2] > 1:  # prefill: derive from the weights before any edit
            self.vis_sinks[layer_idx] = attention_sinks(attn, self.im, self.P, self.cfg.attn_sink_mult)
        return self.vis_sinks.get(layer_idx, torch.empty(0, dtype=torch.long, device=attn.device))

    def record_delta_o(self, layer_idx: int, before: torch.Tensor, after: torch.Tensor,
                       value_states: torch.Tensor, out: torch.Tensor) -> None:
        """H2 instrumentation (off unless `cfg.record_delta`): how much this layer's edit moved the
        attention output. `delta_o = (A_edited - A_base) @ V`, the same quantity as `framework.md` §1.

        Bucketed by phase, because they are not equally relevant. The prompt prefills "Answer:", so the
        prediction is the **first generated token** and comes from the **last prefill row**: edits made at
        a decode step cannot change it. So:
        - `answer`  -- the last prefill row only. The strictly decision-relevant statistic.
        - `prefill` -- all prefill rows (only those >= im + P are edited; the rest contribute 0 to the
                       numerator but still count in the denominator, which dilutes every layer equally
                       and so does not affect the ranking).
        - `decode`  -- the decode steps, recorded for completeness and excluded from any ranking.
        `delta_o_rms` / `o_rms` are per-vector RMS norms; their ratio is the scale-free form.
        """
        delta = torch.matmul(after - before, value_states)
        buckets = self.delta.setdefault(layer_idx, {})

        def add(key: str, dv: torch.Tensor, ov: torch.Tensor) -> None:
            acc = buckets.setdefault(key, [0.0, 0.0, 0])
            acc[0] += float(dv.float().pow(2).sum())
            acc[1] += float(ov.float().pow(2).sum())
            acc[2] += dv.shape[0] * dv.shape[1] * dv.shape[-2]

        if delta.shape[-2] > 1:  # prefill
            add("prefill", delta, out)
            add("answer", delta[:, :, -1:, :], out[:, :, -1:, :])
        else:
            add("decode", delta, out)

    def edit_attention(self, layer_idx: int, attn: torch.Tensor) -> torch.Tensor:
        """HeadFork + redistribution on the softmaxed weights (llava.py:139-148)."""
        sinks = self.sinks.get(layer_idx)
        if sinks is None or not self.active_layer(layer_idx):
            return attn
        if self._stop == "detect":  # timing only: detector ran, stop before sink prep
            return attn
        if self._per_row:
            return self._edit_rows(layer_idx, attn, sinks)

        phase = "prefill" if attn.shape[-2] > 1 else "decode"
        cached = self._prep_cache.get(layer_idx) if self._lean else None
        if cached is None:
            with maybe(self.timer, "prep", layer_idx, phase):
                detector_vis, text_idx = split_sinks(sinks.to(attn.device), self.im, self.P)
                vis_idx = self.visual_sinks(layer_idx, attn, detector_vis)
            if self._lean and self.cfg.sink_source == "detector":
                # The detector's set is fixed at prefill, so split_sinks returns the same two tensors
                # on every forward. var_attn is excluded: its I_v is derived from live attention.
                self._prep_cache[layer_idx] = (vis_idx, text_idx)
        else:
            vis_idx, text_idx = cached
        if vis_idx.numel() == 0:
            return attn
        if self._stop == "prep":  # timing only: stop before the gate
            return attn

        with maybe(self.timer, "gate", layer_idx, phase):
            keep = head_gate(attn, self.im, self.P, vis_idx, self.cfg)
        q_len = attn.shape[-2]
        if q_len > 1:  # prefill: only rows after the image block are edited (logic.py:179)
            rows = self._rows_cache.get(q_len) if self._lean else None
            if rows is None:
                rows = torch.arange(q_len, device=attn.device) >= self.im + self.P
                if self._lean:
                    self._rows_cache[q_len] = rows
            keep = keep & rows
        if not bool(keep.any()):
            return attn
        if self._stop == "gate":  # timing only: gate ran, stop before the edit
            return attn

        self.stats["layers_edited"] += 1
        if self._lean:
            # `int(keep.sum())` is a device->host sync per edited layer per forward, for bookkeeping
            # that nothing reads until the run is over. Accumulate on device, reduce once at summary.
            s = keep.sum()
            self._rows_edited = s if self._rows_edited is None else self._rows_edited + s
        else:
            self.stats["rows_edited"] += int(keep.sum())
        self.stats["n_vis"].append(int(vis_idx.numel()))
        self.stats["n_text"].append(int(text_idx.numel()))
        with maybe(self.timer, "edit", layer_idx, phase):
            return redistribute(attn, self.im, self.P, vis_idx, text_idx, keep, self.cfg.p,
                                check_any=not self._lean)

    def _edit_rows(self, layer_idx: int, attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`edit_attention` for a batch of distinct images (measurement 3): the same stages, with the
        sink set held per row as a [B, T0] mask. Lean like R: prep is cached per layer, no extra syncs.
        Rows with no visual sink are left untouched, as the index path returns early for them."""
        phase = "prefill" if attn.shape[-2] > 1 else "decode"
        cached = self._prep_cache.get(layer_idx)
        if cached is None:
            with maybe(self.timer, "prep", layer_idx, phase):
                sinks = mask.to(attn.device)
                in_img = torch.zeros(sinks.shape[-1], dtype=torch.bool, device=attn.device)
                in_img[self.im:self.im + self.P] = True
                vis = sinks & in_img
                has_vis = vis.any(-1)
            cached = self._prep_cache[layer_idx] = (vis, sinks, has_vis)
        vis, sinks, has_vis = cached

        with maybe(self.timer, "gate", layer_idx, phase):
            keep = head_gate_rows(attn, self.im, self.P, vis[:, self.im:self.im + self.P], self.cfg)
            keep = keep & has_vis[:, None, None]
        q_len = attn.shape[-2]
        if q_len > 1:  # prefill: only rows after the image block are edited (logic.py:179)
            rows = self._rows_cache.get(q_len)
            if rows is None:
                rows = self._rows_cache[q_len] = torch.arange(q_len, device=attn.device) >= self.im + self.P
            keep = keep & rows
        if not bool(keep.any()):
            return attn

        self.stats["layers_edited"] += 1
        # Per-row counts would cost kernels here; a reference to `keep` costs a list append. Reduced in
        # finalise_stats, after the timed region.
        self._keep_log.append((layer_idx, keep))
        with maybe(self.timer, "edit", layer_idx, phase):
            return redistribute_rows(attn, self.im, self.P, vis, sinks, keep, self.cfg.p)

    def finalise_stats(self) -> None:
        """Fold the on-device `rows_edited` accumulator back in. Called after generation, so the one
        sync it costs lands outside the timed region; `stats["rows_edited"]` is incomplete until then."""
        if self._rows_edited is not None:
            self.stats["rows_edited"] += int(self._rows_edited)
            self._rows_edited = None
        if self._per_row and self._keep_log:
            B = self._keep_log[0][1].shape[0]
            dev = self._keep_log[0][1].device
            rows = torch.zeros(B, dtype=torch.long, device=dev)
            events = torch.zeros(B, dtype=torch.long, device=dev)
            nvis = torch.zeros(B, dtype=torch.long, device=dev)
            ntext = torch.zeros(B, dtype=torch.long, device=dev)
            for layer, keep in self._keep_log:
                vis, sinks, _ = self._prep_cache[layer]
                edited = keep.flatten(1).any(-1)
                rows += keep.flatten(1).sum(-1)
                events += edited
                nv = vis.sum(-1)
                nvis += nv * edited
                ntext += (sinks.sum(-1) - nv) * edited
            ev = events.tolist()
            self.stats["per_row"] = {
                "rows_edited": rows.tolist(), "edit_events": ev,
                "n_vis": [v / e if e else None for v, e in zip(nvis.tolist(), ev)],
                "n_text": [v / e if e else None for v, e in zip(ntext.tolist(), ev)]}
            self.stats["rows_edited"] += int(rows.sum())
            self._keep_log = []
