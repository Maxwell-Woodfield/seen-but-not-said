"""Phase 6 `diag` config family (notes/diagnosis_plan.md §2): `R`'s mechanism with ONE part altered.

Every knob defaults to `R`'s behaviour, and the defaults run through this module's own code path -- no
delegation to `VarState.edit_attention` -- so "diag at defaults is bitwise `R`" (scripts/check_diag.py)
tests the very path the arms run on. `var` / `R` themselves are untouched.

Knobs (card in brackets), paper convention throughout (sinks x (1 - p), budget = p x shrunk mass):
- `rows` [C1]: "post_image" (`R`: prefill rows after the image block, and every decode row) | "answer"
  (the last prefill row only) | "prompt" (post-image prefill rows except the last). The last two edit no
  decode row: the prediction is the first generated token, fixed by the prefill.
- `answer_summ` [C1]: None (`R`) | the mass floor used for the last prefill row only (floor-off = 0.0).
- `source` [C2]: "both" (`R`: visual and text sinks shrink, both budgets move) | "visual" | "text". The
  paper's rule stands in every mode: a layer with no visual sink is a no-op, so the row set is `R`'s.
- `sinks` [C5]: "detector" (`R`) | "random" (|I_v| random non-sink image tokens per layer, drawn from a
  generator seeded once per method, so reproducible for a fixed image order) | "const" (image patch
  `const_patch` in layers `const_from`..`const_to` = 9..32, R's band (Amendment 2d); no visual sink
  outside it; text sinks stay the detector's).
- `target` [D1]: "proportional" (`R`) | "uniform" | "sharpen" (proportional to alpha^2) | "centre" (a
  Gaussian over the patch grid, sigma = `centre_sigma` patches; label-free, a dataset-specific spatial
  prior) | "text" (non-sink text columns, proportional to their weight -- the paper's negative ablation).
In every mode the gate sees the (possibly replaced) visual sink set, as VAR's would, and rows sum to 1.

`record_budget` (untimed runs only; it syncs per edited layer) logs the realised edit at the decision row
(the last prefill row) per layer: heads edited and mass moved, next to that row's S_v, S_t and BOS mass.
`ObserveState` is B2's instrument: it records every layer's decision row before any edit and, unless
`edit` is set, returns the weights untouched.
"""
from dataclasses import dataclass, field, fields

import torch

from effvar.var import GATE_EPS, VarConfig, VarState, head_gate, split_sinks

ROWS = ("post_image", "answer", "prompt")
SOURCES = ("both", "visual", "text")
SINKS = ("detector", "random", "const")
TARGETS = ("proportional", "uniform", "sharpen", "centre", "text")


@dataclass(frozen=True)
class DiagConfig(VarConfig):
    rows: str = "post_image"
    answer_summ: float | None = None
    source: str = "both"
    sinks: str = "detector"
    const_patch: int = 192       # absolute token 197 with im_start = 5 (notes/sink_geometry.md)
    const_from: int = 9          # first layer where R's detector finds a visual sink
    const_to: int = 32           # last layer R edits (it never edits 33): Amendment 2d
    target: str = "proportional"
    centre_sigma: float | None = None
    seed: int = 0
    record_budget: bool = False  # log the realised decision-row edit (untimed runs only)
    name: str = "diag"

    def __post_init__(self) -> None:
        for knob, allowed in (("rows", ROWS), ("source", SOURCES), ("sinks", SINKS), ("target", TARGETS)):
            if getattr(self, knob) not in allowed:
                raise ValueError(f"{knob} must be one of {allowed}, got {getattr(self, knob)!r}")
        if self.target == "centre" and not self.centre_sigma:
            raise ValueError("target='centre' needs centre_sigma (fixed at D1's registration)")
        if self.sinks != "detector" and self.sink_source != "detector":
            raise ValueError("sinks='random'/'const' replace the detector's I_v; use sink_source='detector'")
        if self.ablate is not None or self.profile or self.profile_host:
            raise ValueError("diag is an accuracy family: no timing ablations or profilers")

    def params(self) -> dict:
        out = super().params()
        out.update({"family": "diag", "rows": self.rows, "answer_summ": self.answer_summ, "source": self.source,
                    "sinks": self.sinks, "target": self.target, "seed": self.seed})
        if self.sinks == "const":
            out.update({"const_patch": self.const_patch, "const_from": self.const_from, "const_to": self.const_to})
        if self.target == "centre":
            out["centre_sigma"] = self.centre_sigma
        if self.record_budget:
            out["record_budget"] = True
        return out


def diag_gate(attn: torch.Tensor, im: int, P: int, vis_idx: torch.Tensor, cfg: DiagConfig) -> torch.Tensor:
    """HeadFork exactly as `R` (`head_gate`); with `answer_summ` set, the last prefill row uses that floor."""
    keep = head_gate(attn, im, P, vis_idx, cfg)
    if cfg.answer_summ is not None and attn.shape[-2] > 1:
        last = attn[..., -1, :]
        img = last[..., im:im + P].float()
        mass = img.sum(-1)
        portion = last[..., vis_idx].float().sum(-1) / (mass + P * GATE_EPS)
        keep[..., -1] = (portion <= 1.0 - cfg.rho) & (mass >= cfg.answer_summ)
    return keep


def centre_prior(P: int, sigma: float, device) -> torch.Tensor:
    side = int(round(P ** 0.5))
    if side * side != P:
        raise ValueError(f"P = {P} is not a square patch grid")
    r = torch.arange(side, device=device, dtype=torch.float32) - (side - 1) / 2
    g = torch.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma ** 2))
    return g.flatten()


def redistribute_diag(attn: torch.Tensor, im: int, P: int, vis_idx: torch.Tensor, text_idx: torch.Tensor,
                      edit: torch.Tensor, cfg: DiagConfig) -> torch.Tensor:
    """`effvar.var.redistribute` with the budget source and the target made explicit.

    At source="both", target="proportional" this performs `redistribute`'s operations in the same order,
    so the result is bitwise identical (tests/test_diag.py). Rows whose target mass is 0 are left alone.
    """
    p = cfg.p
    a = attn.float()
    img = a[..., im:im + P]
    non_sink = img.clone()
    non_sink[..., vis_idx - im] = 0.0
    N = non_sink.sum(-1)                                   # [B, H, Q]
    if cfg.source == "both":
        budget = p * (a[..., vis_idx].sum(-1) + a[..., text_idx].sum(-1))
    elif cfg.source == "visual":
        budget = p * a[..., vis_idx].sum(-1)
    else:
        budget = p * a[..., text_idx].sum(-1)

    out = a.clone()
    if cfg.source != "text":
        out[..., vis_idx] *= 1.0 - p
    if cfg.source != "visual" and text_idx.numel():
        out[..., text_idx] *= 1.0 - p

    tiny = torch.finfo(a.dtype).tiny
    valid = N > 0
    if cfg.target == "proportional":
        ratio = non_sink / N.clamp_min(tiny).unsqueeze(-1)
        out[..., im:im + P] += budget.unsqueeze(-1) * ratio
    elif cfg.target == "text":
        cols = torch.ones(a.shape[-1], dtype=torch.bool, device=a.device)
        cols[im:im + P] = False
        cols[text_idx] = False
        w = a * cols
        Z = w.sum(-1)
        valid = valid & (Z > 0)
        out += budget.unsqueeze(-1) * (w / Z.clamp_min(tiny).unsqueeze(-1))
    else:
        free = torch.ones(P, dtype=a.dtype, device=a.device)
        free[vis_idx - im] = 0.0
        if cfg.target == "uniform":
            w = free.expand_as(img)
        elif cfg.target == "sharpen":
            w = non_sink * non_sink
        else:  # centre
            w = (centre_prior(P, cfg.centre_sigma, a.device) * free).expand_as(img)
        Z = w.sum(-1)
        valid = valid & (Z > 0)
        out[..., im:im + P] += budget.unsqueeze(-1) * (w / Z.clamp_min(tiny).unsqueeze(-1))

    keep = (edit & valid).unsqueeze(-1)
    return torch.where(keep, out, a).to(attn.dtype)


@dataclass
class DiagState(VarState):
    """`VarState` with the diag knobs applied in `edit_attention`. Detection (`observe`) is `R`'s."""

    rng: torch.Generator | None = None
    _rows_diag: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)   # layer -> decision-row log (cfg.record_budget)

    def _visual_override(self, layer_idx: int, vis_idx: torch.Tensor, device) -> torch.Tensor:
        cfg = self.cfg
        if cfg.sinks == "const":
            if not cfg.const_from <= layer_idx <= cfg.const_to:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.tensor([self.im + cfg.const_patch], dtype=torch.long, device=device)
        if cfg.sinks == "random" and vis_idx.numel():
            free = torch.ones(self.P, dtype=torch.bool)
            free[(vis_idx - self.im).cpu()] = False
            cand = torch.nonzero(free).flatten()
            pick = cand[torch.randperm(cand.numel(), generator=self.rng)[:vis_idx.numel()]]
            return (pick.sort().values + self.im).to(device)
        return vis_idx

    def _row_filter(self, q_len: int, device) -> torch.Tensor:
        rows = self._rows_diag.get(q_len)
        if rows is None:
            pos = torch.arange(q_len, device=device)
            if self.cfg.rows == "post_image":
                rows = pos >= self.im + self.P
            elif self.cfg.rows == "answer":
                rows = pos == q_len - 1
            else:
                rows = (pos >= self.im + self.P) & (pos < q_len - 1)
            self._rows_diag[q_len] = rows
        return rows

    def edit_attention(self, layer_idx: int, attn: torch.Tensor) -> torch.Tensor:
        sinks = self.sinks.get(layer_idx)
        if sinks is None or not self.active_layer(layer_idx):
            return attn
        q_len = attn.shape[-2]
        if q_len == 1 and self.cfg.rows != "post_image":
            return attn
        cached = self._prep_cache.get(layer_idx)
        if cached is None:
            detector_vis, text_idx = split_sinks(sinks.to(attn.device), self.im, self.P)
            vis_idx = self.visual_sinks(layer_idx, attn, detector_vis)
            vis_idx = self._visual_override(layer_idx, vis_idx, attn.device)
            if self.cfg.sink_source == "detector":  # fixed per sample, so the random draw is too
                self._prep_cache[layer_idx] = (vis_idx, text_idx)
        else:
            vis_idx, text_idx = cached
        if vis_idx.numel() == 0:
            return attn

        keep = diag_gate(attn, self.im, self.P, vis_idx, self.cfg)
        if q_len > 1:
            keep = keep & self._row_filter(q_len, attn.device)
        if not bool(keep.any()):
            return attn

        self.stats["layers_edited"] += 1
        s = keep.sum()
        self._rows_edited = s if self._rows_edited is None else self._rows_edited + s
        self.stats["n_vis"].append(int(vis_idx.numel()))
        self.stats["n_text"].append(int(text_idx.numel()))
        out = redistribute_diag(attn, self.im, self.P, vis_idx, text_idx, keep, self.cfg)
        if self.cfg.record_budget and q_len > 1:
            self._log_budget(layer_idx, attn, out, vis_idx, text_idx)
        return out

    def _log_budget(self, layer_idx, before, after, vis_idx, text_idx) -> None:
        """The realised edit at the decision row: mass moved per head is half the L1 change of the row
        (what the sinks lose, the targets gain), so it is exact whatever the source/target knobs."""
        b, a = before[0, :, -1, :].float(), after[0, :, -1, :].float()   # [H, T]
        moved = 0.5 * (a - b).abs().sum(-1)
        self.budget[layer_idx] = {"moved": moved.cpu(), "S_v": b[:, vis_idx].sum(-1).cpu(),
                                  "S_t": b[:, text_idx].sum(-1).cpu(), "bos": b[:, 0].cpu(),
                                  "img": b[:, self.im:self.im + self.P].sum(-1).cpu()}

    def budget_summary(self) -> dict:
        """Per image: per layer, heads edited at the decision row and mass moved (mean over all heads, and
        over edited heads), with the edited heads' S_v / S_t / BOS / image mass; plus the layer total."""
        out, total = {}, 0.0
        for layer, d in sorted(self.budget.items()):
            ed = d["moved"] > 0
            n = int(ed.sum())
            mean_ed = (lambda t: float(t[ed].mean()) if n else None)  # noqa: E731
            out[str(layer)] = {"heads_edited": n, "moved": float(d["moved"].mean()), "moved_edited": mean_ed(d["moved"]),
                               "S_v": mean_ed(d["S_v"]), "S_t": mean_ed(d["S_t"]), "bos": mean_ed(d["bos"]),
                               "img": mean_ed(d["img"])}
            total += float(d["moved"].mean())
        return {"by_layer": out, "moved_total": total}


@dataclass
class ObserveState(DiagState):
    """B2's instrument. Records every layer's decision row (the last prefill row) BEFORE any edit: per head
    image mass, S_v and S_t under the detector's sets, BOS mass and image-attention entropy, plus the
    head-averaged attention over the image block (for pooling) and the layer's visual sinks. With
    `edit=False` the weights pass through untouched, so logits are `base`'s; with `edit=True` it is the
    diag edit (R at defaults)."""

    edit: bool = False
    answer: dict = field(default_factory=dict)

    def edit_attention(self, layer_idx: int, attn: torch.Tensor) -> torch.Tensor:
        if attn.shape[-2] > 1:
            row = attn[0, :, -1, :].float()                               # [H, T]
            img = row[:, self.im:self.im + self.P]
            mass = img.sum(-1)
            sinks = self.sinks.get(layer_idx)
            empty = torch.empty(0, dtype=torch.long, device=attn.device)
            vis, text = split_sinks(sinks.to(attn.device), self.im, self.P) if sinks is not None else (empty, empty)
            q = img / mass.clamp_min(torch.finfo(row.dtype).tiny).unsqueeze(-1)
            ent = -(q * q.clamp_min(torch.finfo(row.dtype).tiny).log()).sum(-1)
            heads = torch.stack([mass, row[:, vis].sum(-1), row[:, text].sum(-1), row[:, 0], ent], -1)
            self.answer[layer_idx] = {"heads": heads.cpu(), "img_attn": img.mean(0).cpu(), "vis": (vis - self.im).cpu()}
        return super().edit_attention(layer_idx, attn) if self.edit else attn


DIAG_DEFAULT = DiagConfig()

# Knobs `--diag key=value` may set: the diag knobs plus R's own hyperparameters. Instrumentation, timing
# and naming stay out of reach.
SETTABLE = ("p", "rho", "tau", "summ", "layers", "except_last_layer", "sink_source", "rows", "answer_summ",
            "source", "sinks", "const_patch", "const_from", "const_to", "target", "centre_sigma", "seed",
            "record_budget")
_FLOATS = {"p", "rho", "tau", "summ", "answer_summ", "centre_sigma"}
_INTS = {"const_patch", "const_from", "const_to", "seed"}
_BOOLS = {"except_last_layer", "record_budget"}


def parse_overrides(pairs: list[str]) -> dict:
    """['p=0.6', 'target=uniform', 'answer_summ=none', 'layers=9-16'] -> DiagConfig overrides."""
    known = {f.name for f in fields(DiagConfig)}
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or key not in SETTABLE or key not in known:
            raise ValueError(f"bad diag override {pair!r}; settable: {', '.join(SETTABLE)}")
        if raw.lower() == "none":
            out[key] = None
        elif key in _FLOATS:
            out[key] = float(raw)
        elif key in _INTS:
            out[key] = int(raw)
        elif key in _BOOLS:
            out[key] = raw.lower() in ("1", "true", "yes")
        elif key == "layers":
            lo, _, hi = raw.partition("-")
            out[key] = tuple(range(int(lo), int(hi) + 1)) if hi else tuple(int(x) for x in raw.split(","))
        else:
            out[key] = raw
    return out
