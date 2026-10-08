"""Phase 7, card V1: visual contrastive decoding (VCD) at MedGemma's first-token readout.

Binding text: notes/vcd_plan.md (card V1, pre-registration changes 1-2, Amendment V1-1) and CLAUDE.md "Phase 7".
VCD/ is never imported; only its formulas are ported:
- the noise schedule of VCD/vcd_utils/vcd_add_noise.py (sigmoid beta, 1000 steps), in fp32 as that file computes it;
- the contrast u_c = (1+alpha)*log P_c - alpha*log P'_c, at the OPTION level (change 1): P_c is letter c's
  first-token probability summed over its two token ids (" X", "X"), i.e. the stored `p_option`;
- the adaptive plausibility constraint (APC) V_head(beta), on the arm's own CLEAN branch.

Score conventions:
- clean side: z = log of p_option renormalised over the options, clipped at diag_stats.EPS. This is the exact array
  Phase 6's OvO saw, so at alpha = 0 VCD reproduces base bitwise;
- noised side: z' = the stored fp32 `logp_option` renormalised in log space, never clipped;
- the per-image constants (log sum P) cancel in every pair log-ratio, every argmax and every softmax.

Also here: the noised input path (build_inputs replicated up to its bf16 cast), the fp32 readout stored per row, the
G3/G4 run layout (`SubRun`, `preset`) and the provenance helpers shared by the scripts.
"""
import hashlib
import re
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

from effvar import ROOT
from effvar.diag_stats import EPS, binary_auc
from effvar.generation import MAX_NEW_TOKENS, Generation, _generate
from effvar.model import TEMPLATE_KWARGS

# --------------------------------------------------------------------------- registered constants
REGISTERED_IDS = {"A": (562, 236776), "B": (603, 236799), "C": (565, 236780), "D": (622, 236796),
                  "E": (645, 236788), "F": (633, 236811), "G": (667, 236823)}   # (" X", "X"), change 1
# Amendment V1-1: saved pred != argmax_c P_c, under the arm each belongs to ("var" = R).
MISMATCH_ROWS = {"base": {"retinamnist": (258,), "dermamnist": (209,)},
                 "var": {"pneumoniamnist": (291, 4328), "dermamnist": (2086, 2353, 6009)}}
FOLD_SHA1 = {"retinamnist": "7352fd59ab38", "pneumoniamnist": "a99b0b75d72d", "dermamnist": "f5fe03d3068e"}
B2_C = {"retinamnist": 1e-4, "pneumoniamnist": 1e-2, "dermamnist": 0.1}            # change 2
MDE80_TUNE = {"retinamnist": 0.012, "pneumoniamnist": 0.012, "dermamnist": 0.019}   # Amendment 2b
ALPHAS = (0.25, 0.5, 1.0, 2.0)
DECIDING_ALPHAS = (0.5, 1.0)
CF_ALPHAS = (0.5, 1.0)
BETA, BETA_SENS = 0.1, 0.2
SCHEDULE = "vcd_sigmoid_1000"
N_STEPS = 1000
STEPS = (500, 999)
GPU_CAP_S = 3600.0             # 1.0 GPU-h on tune (CLAUDE.md "Phase 7")
FALLBACK_S_PER_IMAGE = 1.1     # uncached, Phase 4 (card "Cost"); used only before anything is measured
SAVED_RUNS = {"base": "tune/base", "var": "tune/var/rho0.5_summ0.1"}   # under results/
ATTN = "var_eager"
V1_EXP = "tune/vcd/V1"
SMOKE_EXP = f"{V1_EXP}/smoke"
PROCESS_DIR = f"{V1_EXP}/processes"
REPLACED_DIR = f"{V1_EXP}/_replaced"
CODE_DIRS = ("effvar", "scripts", "tests")
PLAN_INDEX = ROOT / "notes" / "vcd_plan.md"


# --------------------------------------------------------------------------- noise (VCD's schedule)
def noise_coefficients(step: int) -> tuple[float, float]:
    """(sqrt(alpha_bar_t), sqrt(1 - alpha_bar_t)), ported from VCD/vcd_utils/vcd_add_noise.py l. 7-16 in fp32:
    beta = sigmoid(linspace(-6, 6, 1000)) * (5e-3 - 1e-5) + 1e-5; alpha_bar_t = prod_{s<=t} (1 - beta_s)."""
    if not 0 <= int(step) < N_STEPS:
        raise ValueError(f"step must be in [0, {N_STEPS}), got {step}")
    betas = torch.sigmoid(torch.linspace(-6, 6, N_STEPS, dtype=torch.float32)) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    return float(torch.sqrt(alphas_prod)[step]), float(torch.sqrt(1 - alphas_prod)[step])


def vcd_seed(flag: str, split: str, index: int, rep: int) -> int:
    """The card's seed: sha1(f"{flag}|{split}|{index}|{rep}")[:8] as a big-endian int, masked to 63 bits. split and
    index are the image's MedMNIST split and index; the step is NOT in the seed."""
    digest = hashlib.sha1(f"{flag}|{split}|{int(index)}|{int(rep)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def draw_eps(seed: int, shape) -> torch.Tensor:
    """eps ~ N(0, I), fp32, from a per-image CPU generator. Takes no step: steps 500 and 999 share eps."""
    return torch.randn(tuple(shape), generator=torch.Generator("cpu").manual_seed(int(seed)), dtype=torch.float32)


def add_noise(x: torch.Tensor, c1: float, c2: float, eps: torch.Tensor) -> torch.Tensor:
    """x' = c1*x + c2*eps on the processor-normalised fp32 pixel_values, unclipped. Always computed (no shortcut at
    (1, 0)), so guard 1 exercises this arithmetic; at (1, 0) the result is x bitwise."""
    if x.dtype != torch.float32 or x.device.type != "cpu":
        raise TypeError(f"noise is added to fp32 CPU pixel_values; got {x.dtype} on {x.device}")
    if eps.dtype != torch.float32 or eps.shape != x.shape:
        raise TypeError(f"eps must be fp32 with shape {tuple(x.shape)}; got {eps.dtype} {tuple(eps.shape)}")
    return c1 * x + c2 * eps


# --------------------------------------------------------------------------- noised input path
def build_noised_inputs(processor, messages, device, c1: float, c2: float, seed: int):
    """`effvar.model.build_inputs`, up to its bf16 cast, with x' = c1*x + c2*eps on the fp32 pixel_values."""
    inputs = processor.apply_chat_template(messages, **TEMPLATE_KWARGS)
    x = inputs["pixel_values"]
    inputs["pixel_values"] = add_noise(x, c1, c2, draw_eps(seed, x.shape))
    return inputs.to(device, dtype=torch.bfloat16)


def run_noised(model, processor, method, spec, image, c1: float, c2: float, seed: int,
               max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
    """`effvar.generation.run_untimed` on the noised input: the same `_generate` call (greedy, early stop)."""
    inputs = build_noised_inputs(processor, spec.messages(image), model.device, c1, c2, seed)
    n_in = inputs["input_ids"].shape[-1]
    torch.cuda.reset_peak_memory_stats()
    out = _generate(model, processor, method, inputs, max_new_tokens)
    new_ids = out.sequences[0, n_in:].tolist()
    return Generation(input_len=n_in, new_ids=new_ids,
                      text=processor.decode(new_ids, skip_special_tokens=True),
                      first_logits=out.logits[0][0].float().cpu(),
                      vision_ms=None, prefill_ms=None, decode_ms_per_token=None, e2e_ms=None,
                      peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)


# --------------------------------------------------------------------------- readout stored per row
def readout(logits: torch.Tensor, token_sets: list[list[int]]) -> dict:
    """Per-id and per-option first-token readout, fp32 over the full vocabulary (the same log_softmax as
    `effvar.scoring.first_token_metrics`). Nested lists follow `option_token_sets` order ([" X", "X"])."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return {"p_ids": [[float(p[i]) for i in s] for s in token_sets],
            "logp_ids": [[float(logp[i]) for i in s] for s in token_sets],
            "logp_option": [float(torch.logsumexp(logp[list(s)], dim=0)) for s in token_sets]}


def registered_sets(spec) -> list[list[int]]:
    return [sorted(REGISTERED_IDS[label]) for label in spec.labels]


def check_registered_ids(token_sets, spec) -> None:
    """Guard 5: each letter's id set equals the registered one (change 1)."""
    expected = registered_sets(spec)
    if [list(s) for s in token_sets] != expected:
        raise SystemExit(f"guard 5: {spec.flag} option ids {token_sets} != registered {expected}")


# --------------------------------------------------------------------------- option-level scores (CPU)
def renorm_rows(P) -> np.ndarray:
    """P renormalised over the options row by row, exactly as `effvar.diag_stats.load_run` does it."""
    return np.stack([np.asarray(r, dtype=float) / np.asarray(r, dtype=float).sum() for r in P])


def log_clean(p_renorm) -> np.ndarray:
    """Clean side (Phase 6): log of the renormalised p_option, clipped at EPS -- the array ovo_logratio_auc sees."""
    return np.log(np.clip(np.asarray(p_renorm, dtype=float), EPS, None))


def log_noised(logp_option) -> np.ndarray:
    """Noised side: the stored fp32 log P'_c renormalised over the options in log space; never clipped."""
    lp = np.asarray(logp_option, dtype=float)
    if not np.isfinite(lp).all():
        raise ValueError("non-finite noised logp_option")
    return lp - logsumexp(lp, axis=-1, keepdims=True)


def vcd_score(z, z_noised, alpha: float) -> np.ndarray:
    """u = (1+alpha)*z - alpha*z'. z' is [n, K], or [K] for a constant distorted view (VCD_cf, VCD_mean)."""
    return (1.0 + alpha) * np.asarray(z, dtype=float) - alpha * np.asarray(z_noised, dtype=float)


def v_head(P, leakage, beta: float):
    """APC on a clean branch: V = {c : P_c >= beta * max(max_c' P_c', m)}, with m the largest non-option probability.
    Every non-option token has p <= leakage, so V is exact when max P >= leakage. Otherwise the row is flagged and
    leakage stands in for m, and a flagged row may come out empty. Returns (mask [n, K], flagged [n], empty [n]).
    P is the RAW p_option (not renormalised)."""
    P = np.asarray(P, dtype=float)
    leakage = np.asarray(leakage, dtype=float)
    top = P.max(-1)
    flagged = top < leakage
    ref = np.where(flagged, leakage, top)
    mask = P >= beta * ref[..., None]
    return mask, flagged, ~mask.any(-1)


def decide(score, mask, empty, clean_pred) -> np.ndarray:
    """argmax of `score` over V_head; an empty V_head keeps the branch's clean argmax. A non-empty V_head always
    holds the clean argmax, so at alpha = 0 this is the clean decision."""
    pred = np.where(mask, score, -np.inf).argmax(-1)
    return np.where(empty, np.asarray(clean_pred), pred)


def lex_score(u, z, mask, M: float | None = None) -> np.ndarray:
    """APC-lexicographic score (report-only): s_c = u_c on V_head, -M + z_c off it, with M > max|u|. Since z <= 0,
    every masked option ranks below every plausible one."""
    u = np.asarray(u, dtype=float)
    z = np.asarray(z, dtype=float)
    big = float(np.abs(u).max())
    M = 1.0 + big if M is None else float(M)
    if M <= big or (z > 0).any():
        raise ValueError("lex_score needs M > max|u| and z <= 0")
    return np.where(mask, u, -M + z)


def ovo_scores_auc(y, s, k: int) -> float:
    """Macro OvO AUC on log-domain scores: pair (c, j) ranks the images of c and j by s_c - s_j. Identical to
    `effvar.diag_stats.ovo_logratio_auc(y, p, k)` when s = log(clip(p, EPS))."""
    s = np.asarray(s, dtype=float)
    aucs = []
    for c in range(k):
        for j in range(c + 1, k):
            m = (y == c) | (y == j)
            aucs.append(binary_auc(y[m] == c, s[m, c] - s[m, j]))
    return float(np.nanmean(aucs))


def softmax(s) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def renyi(logw, logw2, order: float) -> np.ndarray:
    """Renyi divergence D_order(w || w2) along the last axis, from log-weights (each normalised)."""
    return logsumexp(order * np.asarray(logw) + (1.0 - order) * np.asarray(logw2), axis=-1) / (order - 1.0)


def split_logw(logp_ids) -> np.ndarray:
    """Log of letter c's mass split between its ids, w_t = p_t / P_c, from per-id log-probabilities [..., 2]."""
    lp = np.asarray(logp_ids, dtype=float)
    return lp - logsumexp(lp, axis=-1, keepdims=True)


def per_token_score(logp_ids, logp_ids_noised, alpha: float) -> np.ndarray:
    """Per-token VCD summed to the letter: log sum_{t in c} p_t^(1+alpha) * p'_t^(-alpha), from per-id logs [n, K, 2].
    It equals the option-level u_c + alpha * D_{1+alpha}(w_c || w'_c) (change 1)."""
    return logsumexp((1.0 + alpha) * np.asarray(logp_ids, dtype=float)
                     - alpha * np.asarray(logp_ids_noised, dtype=float), axis=-1)


def a1_folds(y):
    """Phase 6 A1's folds: StratifiedKFold(5, shuffle=True, random_state=0) on the aligned tune labels (Amendment 1
    exclusions applied). Returns (splits, fold id per row as int64)."""
    y = np.asarray(y)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    fold_ids = np.empty(len(y), dtype=np.int64)
    for f, (_, te) in enumerate(splits):
        fold_ids[te] = f
    return splits, fold_ids


def fold_sha1(fold_ids) -> str:
    """The G1 pin: sha1 of the int64 fold-id array, first 12 hex digits."""
    return hashlib.sha1(np.asarray(fold_ids, dtype=np.int64).tobytes()).hexdigest()[:12]


def js(p, q) -> np.ndarray:
    """Jensen-Shannon divergence (nats) along the last axis; B1's definition (scripts/diag_b1.py)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * (np.log(np.clip(a, 1e-12, None)) - np.log(np.clip(b, 1e-12, None))), -1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def boot_p(deltas) -> float:
    """Two-sided percentile-bootstrap p-value of a paired contrast: 2 * min(P*(d <= 0), P*(d >= 0)), with the +1
    correction so it is never 0."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    b = d.size
    lo = (np.sum(d <= 0) + 1) / (b + 1)
    hi = (np.sum(d >= 0) + 1) / (b + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p, kind="mergesort")
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


# --------------------------------------------------------------------------- run layout (G3 / G4)
@dataclass(frozen=True)
class SubRun:
    """One registered run, over all its datasets. `cache`: none (smoke, uncached), noised (the noised-vision cache)
    or clean (the clean cache). Selection: `limit` = first N tune images per dataset, `indices` = explicit indices
    per dataset (added to the first N; alone, they also restrict the datasets); neither = all 300.
    `expect` = (hits, misses) summed over datasets (guard 8), None for the smoke."""
    name: str
    exp: str
    arm: str                      # "base" | "var" (= R at its defaults)
    mode: str                     # "noised" | "clean"
    cache: str                    # "none" | "noised" | "clean"
    step: int | None = None
    rep: int | None = None
    identity: bool = False        # guard 1: coefficients forced to (1, 0)
    limit: int | None = None
    indices: tuple = ()           # ((flag, (idx, ...)), ...)
    max_new_tokens: int = MAX_NEW_TOKENS
    expect: tuple | None = None
    clean_rerun: str | None = None   # "full" | "fallback", clean sub-runs of g4 only

    @property
    def rate_key(self) -> str:
        """The timing class used by the budget projection. A noised run of unknown hit pattern counts as misses
        (the slower class)."""
        if self.cache == "none":
            return "uncached"
        if self.cache == "clean":
            return "hit"
        return "hit" if self.expect is not None and self.expect[1] == 0 else "miss"


def selection(sub: SubRun, subset_indices: dict[str, list[int]], flags=None) -> dict[str, list[int]]:
    """{flag: tune indices in subset order} for a sub-run. Explicit indices must be in the tune subset."""
    flags = list(flags) if flags is not None else list(subset_indices)
    explicit = dict(sub.indices)
    for flag, idx in explicit.items():
        missing = sorted(set(idx) - set(subset_indices[flag]))
        if missing:
            raise ValueError(f"{sub.name}: indices {missing} are not in the {flag} tune subset")
    if explicit and sub.limit is None:
        flags = [f for f in flags if f in explicit]
    out = {}
    for flag in flags:
        order = list(subset_indices[flag])
        if sub.limit is None and not explicit:
            keep = set(order)
        else:
            keep = set(order[:sub.limit] if sub.limit is not None else []) | set(explicit.get(flag, ()))
        rows = [i for i in order if i in keep]
        if rows:
            out[flag] = rows
    return out


def _noised(name, exp, arm, step, cache, rep=1, limit=None, identity=False, mnt=MAX_NEW_TOKENS, expect=None):
    return SubRun(name=name, exp=exp, arm=arm, mode="noised", cache=cache, step=step, rep=rep, identity=identity,
                  limit=limit, max_new_tokens=mnt, expect=expect)


def _mismatch_indices(arm: str) -> tuple:
    return tuple((flag, tuple(idx)) for flag, idx in MISMATCH_ROWS[arm].items())


PRESETS = ("smoke", "smoke-rerun", "g4")


def preset(name: str, clean_rerun: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS) -> list[SubRun]:
    """The registered run sequences. smoke runs base before R (guard 1 then also shows that the base -> R switch in
    one process leaves no state); smoke-rerun runs in a second process (guard 3 across processes); g4 runs
    base_s500 before R_s500 (R reads base's noised cache). `max_new_tokens` applies to g4's noised sub-runs only."""
    s = SMOKE_EXP
    if name == "smoke":
        return [_noised("base_identity", f"{s}/base_identity", "base", 500, "none", limit=10, identity=True),
                _noised("R_identity", f"{s}/R_identity", "var", 500, "none", limit=10, identity=True),
                _noised("base_s500", f"{s}/base_s500", "base", 500, "none", limit=10),
                _noised("base_s999", f"{s}/base_s999", "base", 999, "none", limit=10)]
    if name == "smoke-rerun":
        return [_noised("base_s500_rerun", f"{s}/base_s500_rerun", "base", 500, "none", limit=10),
                _noised("base_s999_rerun", f"{s}/base_s999_rerun", "base", 999, "none", limit=10)]
    if name != "g4":
        raise ValueError(f"unknown preset {name!r}; known: {PRESETS}")
    if clean_rerun not in ("full", "fallback"):
        raise ValueError("g4 needs clean_rerun 'full' or 'fallback' (chosen by the dated, timing-only V1- note)")
    if max_new_tokens not in (MAX_NEW_TOKENS, 1):
        raise ValueError("max_new_tokens must be 8 or 1")
    v, m = V1_EXP, max_new_tokens
    if clean_rerun == "full":
        clean_base = SubRun(name="clean_base", exp=f"{v}/clean_base", arm="base", mode="clean", cache="clean",
                            expect=(900, 0), clean_rerun="full")
    else:   # Amendment V1-1(3) fallback: the 150 seed-repeat images plus Retina 258 (Derma 209 is among the 150)
        clean_base = SubRun(name="clean_base", exp=f"{v}/clean_base", arm="base", mode="clean", cache="clean",
                            limit=50, indices=(("retinamnist", (258,)),), expect=(151, 0), clean_rerun="fallback")
    # clean_R (V1-1(2), unconditional) runs before clean_base (V1-1(3), the conditional part), so a budget stop
    # cannot cost the 5 R mismatch rows.
    return [_noised("base_s500", f"{v}/base_s500", "base", 500, "noised", mnt=m, expect=(0, 900)),
            _noised("R_s500", f"{v}/R_s500", "var", 500, "noised", mnt=m, expect=(900, 0)),
            _noised("base_s999", f"{v}/base_s999", "base", 999, "noised", mnt=m, expect=(0, 900)),
            _noised("base_s500_rep2", f"{v}/base_s500_rep2", "base", 500, "noised", rep=2, limit=50, mnt=m,
                    expect=(0, 150)),
            SubRun(name="clean_R", exp=f"{v}/clean_R", arm="var", mode="clean", cache="clean",
                   indices=_mismatch_indices("var"), expect=(5, 0), clean_rerun=clean_rerun),
            clean_base]


def validate_subrun(sub: SubRun) -> None:
    """The run rules every sub-run must satisfy (single runs and presets alike)."""
    if not sub.exp.startswith(V1_EXP + "/"):
        raise ValueError(f"{sub.name}: exp {sub.exp!r} is not under {V1_EXP}/")
    smoke = sub.exp.startswith(SMOKE_EXP + "/")
    if smoke != (sub.cache == "none"):
        raise ValueError(f"{sub.name}: the smoke runs uncached and only the smoke does (guard 3)")
    if sub.arm not in ("base", "var"):
        raise ValueError(f"{sub.name}: arm must be base or var")
    if sub.mode == "clean":
        if sub.cache != "clean" or sub.step is not None or sub.rep is not None or sub.identity:
            raise ValueError(f"{sub.name}: a clean re-run uses the clean cache and takes no step, rep or identity")
        if sub.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError(f"{sub.name}: clean re-runs keep MAX_NEW_TOKENS {MAX_NEW_TOKENS} (guard 7)")
    elif sub.mode == "noised":
        if sub.step not in STEPS or sub.rep not in (1, 2) or sub.cache not in ("none", "noised"):
            raise ValueError(f"{sub.name}: a noised run needs step 500|999, rep 1|2, cache none|noised")
        if sub.identity and (sub.cache != "none" or sub.max_new_tokens != MAX_NEW_TOKENS):
            raise ValueError(f"{sub.name}: identity runs are uncached with MAX_NEW_TOKENS {MAX_NEW_TOKENS} (guard 1)")
        if sub.max_new_tokens not in (MAX_NEW_TOKENS, 1):
            raise ValueError(f"{sub.name}: max_new_tokens must be 8 or 1")
    else:
        raise ValueError(f"{sub.name}: unknown mode {sub.mode!r}")


def g4_specs(clean_rerun: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, SubRun]:
    return {s.name: s for s in preset("g4", clean_rerun=clean_rerun, max_new_tokens=max_new_tokens)}


def smoke_specs() -> dict[str, SubRun]:
    return {s.name: s for s in preset("smoke") + preset("smoke-rerun")}


def subrun_record(sub: SubRun) -> dict:
    """A SubRun as plain YAML-safe data (lists, not tuples)."""
    return {"name": sub.name, "exp": sub.exp, "arm": sub.arm, "mode": sub.mode, "cache": sub.cache, "step": sub.step,
            "rep": sub.rep, "identity": sub.identity, "limit": sub.limit,
            "indices": {f: list(v) for f, v in sub.indices}, "max_new_tokens": sub.max_new_tokens,
            "expect": None if sub.expect is None else list(sub.expect), "clean_rerun": sub.clean_rerun,
            "rate_key": sub.rate_key}


# --------------------------------------------------------------------------- provenance
def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_code_state() -> dict:
    """HEAD, tracked edits, changes/untracked files under the code dirs, and their trees (Phase 6 Amendment 4d)."""
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "code_changes": git("status", "--porcelain", "--untracked-files=all", "--", *CODE_DIRS).splitlines(),
            "code_trees": {d: git("rev-parse", f"HEAD:{d}") for d in CODE_DIRS}}


def is_clean(state: dict) -> bool:
    return not state["dirty"] and not state["code_changes"]


_G2_ROW = re.compile(r"^\|\s*G2\s*\|.*\|\s*done\s+`([0-9a-f]{7,40})`\s*\|\s*$")


def code_commit_from_index(text: str) -> str:
    """The code commit recorded in the plan index's G2 row ("done `<hash>`")."""
    for line in text.splitlines():
        m = _G2_ROW.match(line.strip())
        if m:
            return m.group(1)
    raise SystemExit("no code commit in the plan index: the G2 row must read 'done `<hash>`' (notes/vcd_plan.md)")


def code_trees_at(commit: str) -> dict:
    return {d: git("rev-parse", f"{commit}:{d}") for d in CODE_DIRS}


def expected_code_trees() -> dict:
    """Guard 6: the trees of the code commit recorded in the plan index (refuses while the G2 row has no hash)."""
    return code_trees_at(code_commit_from_index(PLAN_INDEX.read_text(encoding="utf-8")))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
