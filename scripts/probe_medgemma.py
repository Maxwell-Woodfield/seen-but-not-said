"""Phase 2a: read-only probe of MedGemma-1.5-4b-it on <= 20 MedMNIST val images (no method).

Hooks only *record* reduced statistics; no weight, activation or attention value is modified
(checked with a parameter checksum). Writes results/probe_medgemma/{config.yaml,metrics.json,
per_layer.json}; findings are summarised by hand in notes/medgemma_probe.md.

    python scripts/probe_medgemma.py
Sections: A architecture/masks, B sink dims (App. A.1), C visual sinks, D image-centric heads,
E base latency (SDPA vs eager), F fused-attention-with-LSE kernels, G option tokenization.
"""
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import medmnist
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList
from transformers.models.gemma3 import modeling_gemma3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "probe_medgemma"
MODEL_ID = "google/medgemma-1.5-4b-it"
PROBE_PLAN = {"retinamnist": 7, "pneumoniamnist": 6, "dermamnist": 7}  # 20 images
SINK_THRESH = (20.0, 50.0, 100.0)  # |rmsnorm(x)_d| thresholds for sink-dim discovery
TAUS = (5.0, 10.0, 20.0, 50.0)
HEAD_TAUS = (10.0, 20.0)
RHOS = (0.1, 0.2, 0.5)  # paper convention: head kept iff portion <= 1 - rho
SUMMS = (0.1, 0.2)
DECODE_STEPS = 5
LAT_NEW_TOKENS = 16
# Draft prompt (Phase 3 fixes the final one).
MODALITY = {"retinamnist": "retinal fundus image", "pneumoniamnist": "chest X-ray",
            "dermamnist": "dermatoscopic image"}


def option_names(flag):
    labels = medmnist.INFO[flag]["label"]
    if flag == "retinamnist":
        return [f"grade {labels[k]}" for k in sorted(labels, key=int)]
    return [labels[k] for k in sorted(labels, key=int)]


def draft_prompt(flag):
    opts = option_names(flag)
    return (f"This is a {MODALITY[flag]}. Which one of the following options is correct? "
            f"Options: {'; '.join(opts)}. Answer with exactly one option.")


def rmsnorm_abs(x):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).abs()


# ----------------------------------------------------------------------------- data / model
def probe_images():
    items = []
    for flag, k in PROBE_PLAN.items():
        sub = json.loads((ROOT / "data" / "subsets" / f"{flag}_smoke.json").read_text())
        by_class = defaultdict(list)
        for idx, lab in zip(sub["indices"], sub["labels"]):
            by_class[lab].append(idx)
        order, depth = [], 0
        while len(order) < k:  # class round-robin over the smoke subset
            for lab in sorted(by_class):
                if depth < len(by_class[lab]) and len(order) < k:
                    order.append((idx := by_class[lab][depth], lab))
            depth += 1
        ds = getattr(medmnist, medmnist.INFO[flag]["python_class"])(
            split="val", size=224, as_rgb=True, download=False, root=str(ROOT / "data" / "medmnist"))
        for idx, lab in order:
            img, _ = ds[idx]
            items.append({"flag": flag, "index": idx, "label": lab, "image": img})
    return items


def build_inputs(processor, item, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": item["image"]},
                                             {"type": "text", "text": draft_prompt(item["flag"])}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt").to(device, dtype=torch.bfloat16)


def param_checksum(model):
    w = model.model.language_model.layers[0].self_attn.q_proj.weight
    return float(w.float().sum()), float(w.float().abs().sum())


# ----------------------------------------------------------------------------- A: architecture
def section_a(model, processor, items, inputs_list):
    tc = model.config.text_config
    arch = {
        "num_hidden_layers": tc.num_hidden_layers, "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": tc.num_key_value_heads, "head_dim": tc.head_dim,
        "hidden_size": tc.hidden_size, "sliding_window": tc.sliding_window,
        "full_attention_layers": [i for i, t in enumerate(tc.layer_types) if t == "full_attention"],
        "image_token_index": model.config.image_token_index,
        "mm_tokens_per_image": model.config.mm_tokens_per_image,
        "attn_logit_softcapping": tc.attn_logit_softcapping, "query_pre_attn_scalar": tc.query_pre_attn_scalar,
        "bos_token_id": processor.tokenizer.bos_token_id,
    }
    spans = []
    for item, inp in zip(items, inputs_list):
        ids = inp["input_ids"][0]
        pos = torch.nonzero(ids == model.config.image_token_index).flatten()
        im, P, T = int(pos[0]), int(pos.numel()), int(ids.numel())
        contiguous = bool((pos == torch.arange(im, im + P, device=pos.device)).all())
        assert P == model.config.mm_tokens_per_image and contiguous, (item["flag"], item["index"])
        assert T < tc.sliding_window
        spans.append({"flag": item["flag"], "index": item["index"], "T": T, "im_start": im, "P": P,
                      "post_image_rows": T - (im + P), "bos_at_0": int(ids[0]) == arch["bos_token_id"],
                      "has_token_type_ids": "token_type_ids" in inp})
    arch["T_min"] = min(s["T"] for s in spans)
    arch["T_max"] = max(s["T"] for s in spans)
    return arch, spans


def mask_structure(model, inp):
    """Capture the 4D eager masks of one sliding and one full layer at prefill and describe them."""
    tc = model.config.text_config
    layers = model.model.language_model.layers
    sliding = tc.layer_types.index("sliding_attention")
    full = tc.layer_types.index("full_attention")
    got = {}

    def grab(name):
        def hook(mod, args, kwargs):
            if name not in got:
                got[name] = kwargs["attention_mask"].detach()
        return hook

    hs = [layers[sliding].self_attn.register_forward_pre_hook(grab("sliding"), with_kwargs=True),
          layers[full].self_attn.register_forward_pre_hook(grab("full"), with_kwargs=True)]
    with torch.inference_mode():
        model(**inp)
    for h in hs:
        h.remove()
    ids = inp["input_ids"][0]
    pos = torch.nonzero(ids == model.config.image_token_index).flatten()
    im, P, T = int(pos[0]), int(pos.numel()), int(ids.numel())
    res = {"sliding_layer": sliding, "full_layer": full}
    for name, m in got.items():
        allowed = (m[0, 0] == 0) if m.dtype.is_floating_point else m[0, 0].bool()
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=allowed.device))
        img = torch.zeros(T, T, dtype=torch.bool, device=allowed.device)
        img[im:im + P, im:im + P] = True
        res[name] = {
            "dtype": str(m.dtype), "shape": list(m.shape),
            "image_block_fully_visible": bool(allowed[im:im + P, im:im + P].all()),
            "equals_causal_or_image_block": bool((allowed == (causal | img)).all()),
            "text_rows_causal": bool((allowed[im + P:] == causal[im + P:]).all()),
        }
    return res


# ----------------------------------------------------------------------------- B: sink dims
def section_b(model, inputs_list):
    L = model.config.text_config.num_hidden_layers
    D = model.config.text_config.hidden_size
    layers = model.model.language_model.layers
    counts = torch.zeros(len(SINK_THRESH), D, dtype=torch.long)  # tokens above thresh, layers >= 2
    bos_top = defaultdict(int)  # dim -> times in BOS top-4 (layers >= 2, all images)
    bos_vals = defaultdict(list)
    vis_max = torch.zeros(L, D)
    state = {}

    def pre(layer):
        def hook(mod, args, kwargs):
            if layer < 2:
                return
            r = rmsnorm_abs(args[0][0] if args else kwargs["hidden_states"][0])  # [T, D]
            for i, th in enumerate(SINK_THRESH):
                counts[i] += (r > th).sum(0).cpu()
            top = torch.topk(r[0], 4)
            for v, d in zip(top.values.tolist(), top.indices.tolist()):
                bos_top[d] += 1
                bos_vals[d].append(v)
            im, P = state["im"], state["P"]
            vis_max[layer] = torch.maximum(vis_max[layer], r[im:im + P].max(0).values.cpu())
        return hook

    hs = [layers[l].register_forward_pre_hook(pre(l), with_kwargs=True) for l in range(L)]
    for inp in inputs_list:
        pos = torch.nonzero(inp["input_ids"][0] == model.config.image_token_index).flatten()
        state.update(im=int(pos[0]), P=int(pos.numel()))
        with torch.inference_mode():
            model(**inp)
    for h in hs:
        h.remove()
    n_obs = (L - 2) * len(inputs_list)
    ranked = sorted(bos_top.items(), key=lambda kv: -kv[1])
    bos_ranking = [{"dim": d, "freq_in_bos_top4": c / n_obs, "bos_value_mean": float(np.mean(bos_vals[d])),
                    "bos_value_min": float(np.min(bos_vals[d])),
                    "visual_max_over_layers": float(vis_max[2:, d].max())} for d, c in ranked[:10]]
    d_sink = [r["dim"] for r in bos_ranking if r["freq_in_bos_top4"] >= 0.5]
    token_count_top = {f"gt{int(th)}": [{"dim": int(d), "count": int(counts[i, d])}
                                        for d in torch.topk(counts[i], 6).indices.tolist()]
                       for i, th in enumerate(SINK_THRESH)}
    return {"bos_top4_ranking": bos_ranking, "candidate_D_sink": d_sink,
            "dims_by_token_count_layers2plus": token_count_top}


# ----------------------------------------------------------------------------- C + D: sinks and heads
class SinkHeadRecorder:
    """Records, per layer, visual-sink statistics (C) and HeadFork statistics (D). Read-only."""

    def __init__(self, model, d_sink):
        self.model, self.d_sink = model, d_sink
        tc = model.config.text_config
        self.L, self.Hq, self.Hkv, self.hd = (tc.num_hidden_layers, tc.num_attention_heads,
                                              tc.num_key_value_heads, tc.head_dim)
        self.phi = {}
        self.vnorm = {}
        self.step = -1
        self.rows = defaultdict(list)  # (section, layer) -> list of dicts

    def begin_sample(self, im, P):
        self.im, self.P, self.step = im, P, -1
        self.phi.clear()

    def layer_pre(self, layer):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            if x.shape[1] > 1:  # prefill only (DimProspector semantics)
                if layer == 0:
                    self.step = 0
                self.phi[layer] = rmsnorm_abs(x[0])[:, self.d_sink].max(-1).values  # [T]
            elif layer == 0:
                self.step += 1
        return hook

    def vproj(self, layer):
        def hook(mod, args, out):
            if out.shape[1] > 1:
                v = out[0].float().view(out.shape[1], self.Hkv, self.hd)
                self.vnorm[layer] = v.norm(dim=-1).mean(-1)  # [T]
        return hook

    def attn(self, layer):
        def hook(mod, args, out):
            w = out[1]
            if w is None or layer < 2 or self.step > DECODE_STEPS:
                return
            w = w[0].float()  # [H, Q, T]
            im, P = self.im, self.P
            q_start = im + P if w.shape[1] > 1 else 0
            w = w[:, q_start:]
            if w.shape[1] == 0:
                return
            phi_img = self.phi[layer][im:im + P]
            img = w[..., im:im + P]  # [H, Q, P]
            if w.shape[1] > 1:
                self._sink_stats(layer, img, phi_img, w)
            for tau in HEAD_TAUS:
                sink = phi_img > tau
                s_v = (img * sink).sum(-1)
                summation = img.sum(-1)
                portion = s_v / (img + 1e-6).sum(-1)
                rec = {"tau": tau, "step": self.step, "n_vis_sinks": int(sink.sum()),
                       "summation_mean": float(summation.mean()),
                       "portion_median": float(portion.median()), "portion_mean": float(portion.mean())}
                for rho in RHOS:
                    for summ in SUMMS:
                        keep = (portion <= 1 - rho) & (summation >= summ) if sink.any() else torch.zeros_like(portion, dtype=torch.bool)
                        rec[f"gate_rho{rho}_summ{summ}"] = float(keep.float().mean())
                self.rows[("heads", layer)].append(rec)
        return hook

    def _sink_stats(self, layer, img, phi_img, w):
        P = img.shape[-1]
        vn = self.vnorm.get(layer)
        for tau in TAUS:
            sink = phi_img > tau
            k = int(sink.sum())
            rec = {"tau": tau, "n_vis_sinks": k,
                   "n_text_sinks": int((self.phi[layer] > tau).sum()) - k,
                   "img_attn_mean": float(img.sum(-1).mean())}
            if 0 < k < P:
                s_v, n = (img * sink).sum(-1), (img * ~sink).sum(-1)
                rec["sink_attn_share_of_image"] = float((s_v / (s_v + n)).mean())
                rec["sink_token_share"] = k / P
                rec["per_token_ratio_sink_vs_nonsink"] = float(((s_v / k) / (n / (P - k) + 1e-12)).mean())
                top = torch.topk(img, 5, dim=-1).indices  # [H, Q, 5]
                rec["top5_in_sink_frac"] = float(sink[top].float().mean())
                if vn is not None:
                    v_img = vn[self.im:self.im + P]
                    rec["vnorm_ratio_sink_vs_nonsink"] = float(v_img[sink].mean() / v_img[~sink].mean())
            # query independence: mean pairwise Jaccard of top-5 visual tokens across queries, per head
            memb = torch.zeros_like(img, dtype=torch.float).scatter_(-1, torch.topk(img, 5, -1).indices, 1.0)
            inter = memb @ memb.transpose(-1, -2)  # [H, Q, Q]
            Q = inter.shape[-1]
            if Q > 1:
                jac = inter / (10 - inter)
                rec["top5_jaccard_across_queries"] = float((jac.sum((-1, -2)) - Q) .mean() / (Q * (Q - 1)))
            self.rows[("sinks", layer)].append(rec)


def section_cd(model, inputs_list, d_sink):
    rec = SinkHeadRecorder(model, d_sink)
    layers = model.model.language_model.layers
    hs = []
    for l, layer in enumerate(layers):
        hs.append(layer.register_forward_pre_hook(rec.layer_pre(l), with_kwargs=True))
        hs.append(layer.self_attn.v_proj.register_forward_hook(rec.vproj(l)))
        hs.append(layer.self_attn.register_forward_hook(rec.attn(l)))
    for inp in inputs_list:
        pos = torch.nonzero(inp["input_ids"][0] == model.config.image_token_index).flatten()
        rec.begin_sample(int(pos[0]), int(pos.numel()))
        with torch.inference_mode():
            model.generate(**inp, max_new_tokens=DECODE_STEPS + 1, do_sample=False, min_new_tokens=DECODE_STEPS + 1)
    for h in hs:
        h.remove()

    def agg(rows, keys):
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
            out[k] = float(np.mean(vals)) if vals else None
        return out

    per_layer = {"sinks": {}, "heads_prefill": {}, "heads_decode": {}}
    sink_keys = ["n_vis_sinks", "n_text_sinks", "img_attn_mean", "sink_attn_share_of_image", "sink_token_share",
                 "per_token_ratio_sink_vs_nonsink", "top5_in_sink_frac", "vnorm_ratio_sink_vs_nonsink",
                 "top5_jaccard_across_queries"]
    head_keys = ["n_vis_sinks", "summation_mean", "portion_median", "portion_mean"] + \
        [f"gate_rho{r}_summ{s}" for r in RHOS for s in SUMMS]
    for (sec, layer), rows in rec.rows.items():
        if sec == "sinks":
            for tau in TAUS:
                sel = [r for r in rows if r["tau"] == tau]
                per_layer["sinks"].setdefault(f"tau{tau}", {})[layer] = agg(sel, sink_keys) | {
                    "n_vis_sinks_median": float(np.median([r["n_vis_sinks"] for r in sel])),
                    "frac_images_with_vis_sinks": float(np.mean([r["n_vis_sinks"] > 0 for r in sel]))}
        else:
            for tau in HEAD_TAUS:
                pre = [r for r in rows if r["tau"] == tau and r["step"] == 0]
                dec = [r for r in rows if r["tau"] == tau and r["step"] > 0]
                per_layer["heads_prefill"].setdefault(f"tau{tau}", {})[layer] = agg(pre, head_keys)
                per_layer["heads_decode"].setdefault(f"tau{tau}", {})[layer] = agg(dec, head_keys)
    return per_layer


def summarise_by_type(model, table):
    """Average a {layer: {k: v}} table over sliding vs full layers (layers >= 2)."""
    types = model.config.text_config.layer_types
    out = {}
    for kind in ("sliding_attention", "full_attention"):
        rows = [v for l, v in table.items() if types[int(l)] == kind]
        keys = rows[0].keys() if rows else []
        out[kind] = {k: (float(np.mean([r[k] for r in rows if r.get(k) is not None]))
                         if any(r.get(k) is not None for r in rows) else None) for k in keys}
    return out


# ----------------------------------------------------------------------------- E: latency
class FirstCallTimer(LogitsProcessor):
    def __init__(self):
        self.t_first = None

    def __call__(self, input_ids, scores):
        if self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()
        return scores


def section_e(model, processor, items, inputs_list):
    vt = {}

    def vpre(mod, args, kwargs):
        torch.cuda.synchronize()
        vt["t0"] = time.perf_counter()

    def vpost(mod, args, out):
        torch.cuda.synchronize()
        vt["ms"] = vt.get("ms", 0.0) + (time.perf_counter() - vt["t0"]) * 1e3

    results = {}
    impls = {"sdpa": "sdpa", "eager": "eager",
             "text_eager_vision_sdpa": {"text_config": "eager", "vision_config": "sdpa"}}
    for impl, setting in impls.items():
        try:
            model.set_attn_implementation(setting)
        except Exception as e:
            results[impl] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        h1 = model.model.vision_tower.register_forward_pre_hook(vpre, with_kwargs=True)
        h2 = model.model.multi_modal_projector.register_forward_hook(vpost)
        for _ in range(2):  # warm-up
            with torch.inference_mode():
                model.generate(**inputs_list[0], max_new_tokens=LAT_NEW_TOKENS, do_sample=False)
        rows = []
        for item, inp in zip(items, inputs_list):
            n_in = inp["input_ids"].shape[-1]
            timer = FirstCallTimer()
            vt.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=LAT_NEW_TOKENS, do_sample=False,
                                     logits_processor=LogitsProcessorList([timer]))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            n_new = out.shape[-1] - n_in
            rows.append({"flag": item["flag"], "index": item["index"], "T": n_in, "new_tokens": int(n_new),
                         "vision_ms": vt.get("ms"), "prefill_ms": (timer.t_first - t0) * 1e3,
                         "decode_ms": (t1 - timer.t_first) * 1e3,
                         "decode_ms_per_token": (t1 - timer.t_first) * 1e3 / max(n_new - 1, 1),
                         "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                         "text": processor.decode(out[0][n_in:], skip_special_tokens=True)})
        h1.remove()
        h2.remove()
        keys = ["vision_ms", "prefill_ms", "decode_ms", "decode_ms_per_token", "new_tokens", "peak_vram_gib"]
        results[impl] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                         "median": {k: float(np.median([r[k] for r in rows])) for k in keys}, "rows": rows}
    model.set_attn_implementation("eager")
    if "rows" in results.get("sdpa", {}) and "rows" in results.get("eager", {}):
        same = [a["text"] == b["text"] for a, b in zip(results["sdpa"]["rows"], results["eager"]["rows"])]
        results["sdpa_eager_same_text_frac"] = float(np.mean(same))
    return results


# ----------------------------------------------------------------------------- F: kernels
def capture_qkv(model, inp, layers_to_grab):
    """Wrap Gemma's eager attention function to copy post-RoPE q/k/v and the mask; outputs untouched."""
    grabbed = {}
    orig = modeling_gemma3.eager_attention_forward

    def wrapper(module, query, key, value, attention_mask, **kwargs):
        if module.layer_idx in layers_to_grab and module.layer_idx not in grabbed and query.shape[2] > 1:
            grabbed[module.layer_idx] = (query.detach().clone(), key.detach().clone(), value.detach().clone(),
                                         attention_mask.detach().clone(), module.scaling)
        return orig(module, query, key, value, attention_mask, **kwargs)

    modeling_gemma3.eager_attention_forward = wrapper
    try:
        with torch.inference_mode():
            model(**inp)
    finally:
        modeling_gemma3.eager_attention_forward = orig
    return grabbed


def bench(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    times = []
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": float(np.median(times)), "extra_peak_mib": (torch.cuda.max_memory_allocated() - base) / 2**20}


def section_f(model, inp, sink_idx_by_layer):
    tc = model.config.text_config
    sliding = tc.layer_types.index("sliding_attention", 2)
    full = tc.layer_types.index("full_attention")
    grabbed = capture_qkv(model, inp, {sliding, full})
    ids = inp["input_ids"][0]
    pos = torch.nonzero(ids == model.config.image_token_index).flatten()
    im, P = int(pos[0]), int(pos.numel())
    groups = tc.num_attention_heads // tc.num_key_value_heads
    out = {"triton_available": _has_triton()}
    for name, layer in (("sliding", sliding), ("full", full)):
        q, k, v, mask, scale = grabbed[layer]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        T = k.shape[2]
        sinks = sink_idx_by_layer.get(layer) or []
        synthetic = len(sinks) == 0
        if synthetic:
            sinks = list(range(im, im + 4))
        sink_mask = torch.zeros(T, dtype=torch.bool, device=q.device)
        sink_mask[sinks] = True
        vis = torch.zeros(T, dtype=torch.bool, device=q.device)
        vis[im:im + P] = True
        V = vis & ~sink_mask
        res = {"layer": layer, "T": T, "sinks_synthetic": synthetic, "n_sinks": len(sinks)}
        for shape_name, qq, mm in (("prefill", q, mask), ("decode_q1", q[:, :, -1:], mask[:, :, -1:])):
            bias = _aligned_bias(mm.expand(1, q.shape[1], qq.shape[2], T))
            bias_V = _aligned_bias(bias.masked_fill(~V, torch.finfo(bias.dtype).min))

            def eager():
                s = (qq @ k.transpose(-1, -2)) * scale + mm
                return torch.softmax(s, -1, dtype=torch.float32).to(qq.dtype) @ v

            def sdpa():
                return F.scaled_dot_product_attention(qq, k, v, attn_mask=mm, scale=scale)

            def eff_lse(b=bias):
                return torch.ops.aten._scaled_dot_product_efficient_attention(qq, k, v, b, True, 0.0, False, scale=scale)

            def fused_rank1_route():
                o, lse = eff_lse(bias)[:2]
                vbar, lse_v = eff_lse(bias_V)[:2]
                s = (qq @ k[..., sink_mask, :].transpose(-1, -2)) * scale  # |I|-column matmul
                return o, vbar, torch.exp(lse_v - lse), torch.softmax(s.float(), -1)

            def eager_vectorised_var(p=0.4):
                s = (qq @ k.transpose(-1, -2)) * scale + mm
                w = torch.softmax(s, -1, dtype=torch.float32).to(qq.dtype)
                S = (w * sink_mask).sum(-1, keepdim=True)
                N = (w * V).sum(-1, keepdim=True)
                w = torch.where(sink_mask, w * (1 - p), torch.where(V, w + p * S * w / N.clamp_min(1e-12), w))
                return w @ v

            r = {}
            for kname, fn in (("eager", eager), ("sdpa", sdpa), ("efficient_attn_lse", eff_lse),
                              ("fused_rank1_route_2calls", fused_rank1_route),
                              ("eager_vectorised_var_edit", eager_vectorised_var)):
                try:
                    r[kname] = bench(fn)
                except Exception as e:  # report, don't hide
                    r[kname] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            # LSE correctness vs eager logsumexp (fp32)
            try:
                ref = torch.logsumexp(((qq.float() @ k.float().transpose(-1, -2)) * scale + mm.float()), -1)
                lse = eff_lse()[1][..., :ref.shape[-1]].float()
                r["efficient_attn_lse_max_abs_err"] = float((lse - ref).abs().max())
                r["efficient_attn_lse_max_rel_err"] = float(((lse - ref).abs() / ref.abs().clamp_min(1.0)).max())
                o_ref = eager().float()
                r["efficient_attn_out_max_rel_err"] = float(((eff_lse()[0].float() - o_ref).abs().max() / o_ref.abs().max()))
            except Exception as e:
                r["lse_check_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            r["flex_attention"] = _flex_check(qq, k, v, scale)
            res[shape_name] = r
        out[name] = res
    return out


def _aligned_bias(b):
    """Mem-efficient attention wants the bias's last dim memory-aligned to 8 (as SDPA does internally)."""
    T = b.shape[-1]
    buf = torch.empty(*b.shape[:-1], math.ceil(T / 8) * 8, dtype=b.dtype, device=b.device)
    buf[..., :T] = b
    return buf[..., :T]


def _has_triton():
    import importlib.util
    return importlib.util.find_spec("triton") is not None


def _flex_check(q, k, v, scale):
    try:
        from torch.nn.attention.flex_attention import AuxRequest, flex_attention
    except Exception as e:
        return {"error": f"import: {e}"}
    res = {}
    try:
        res["uncompiled"] = bench(lambda: flex_attention(q, k, v, scale=scale, return_aux=AuxRequest(lse=True)), reps=10, warm=2)
    except Exception as e:
        res["uncompiled"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        compiled = torch.compile(flex_attention)
        res["compiled"] = bench(lambda: compiled(q, k, v, scale=scale, return_aux=AuxRequest(lse=True)), reps=10, warm=2)
    except Exception as e:
        res["compiled"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return res


# ----------------------------------------------------------------------------- G: tokenization
def section_g(processor):
    tok = processor.tokenizer
    out = {}
    for flag in PROBE_PLAN:
        res = {}
        for casing, fn in (("as_listed", str), ("capitalized", str.capitalize)):
            rows, firsts = [], defaultdict(list)
            for name in option_names(flag):
                ids = tok(fn(name), add_special_tokens=False)["input_ids"]
                rows.append({"option": fn(name), "first_token": tok.decode(ids[:1]), "n_tokens": len(ids)})
                firsts[tok.decode(ids[:1])].append(fn(name))
            res[casing] = {"rows": rows, "collisions": {t: o for t, o in firsts.items() if len(o) > 1}}
        out[flag] = res
    return out


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    items = probe_images()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="eager").eval()
    checksum0 = param_checksum(model)
    inputs_list = [build_inputs(processor, it, model.device) for it in items]

    metrics = {}
    t = time.perf_counter()
    metrics["A_architecture"], spans = section_a(model, processor, items, inputs_list)
    metrics["A_masks"] = mask_structure(model, inputs_list[0])
    print("A done", time.perf_counter() - t)
    metrics["B_sink_dims"] = section_b(model, inputs_list)
    d_sink = metrics["B_sink_dims"]["candidate_D_sink"]
    print("B done, candidate D_sink =", d_sink)
    if not d_sink:
        d_sink = [r["dim"] for r in metrics["B_sink_dims"]["bos_top4_ranking"][:2]]
        metrics["B_sink_dims"]["fallback_D_sink_used"] = d_sink
    per_layer = section_cd(model, inputs_list, d_sink)
    metrics["C_visual_sinks_by_type"] = {tau: summarise_by_type(model, tbl) for tau, tbl in per_layer["sinks"].items()}
    metrics["D_heads_prefill_by_type"] = {tau: summarise_by_type(model, tbl) for tau, tbl in per_layer["heads_prefill"].items()}
    metrics["D_heads_decode_by_type"] = {tau: summarise_by_type(model, tbl) for tau, tbl in per_layer["heads_decode"].items()}
    print("C/D done")

    # sink indices for the kernel benchmark: first image, tau 20
    sink_idx = {}
    layers = model.model.language_model.layers
    first = inputs_list[0]

    def grab_phi(layer):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            if x.shape[1] > 1:
                sink_idx[layer] = torch.nonzero(rmsnorm_abs(x[0])[:, d_sink].max(-1).values > 20.0).flatten().tolist()
        return hook
    hs = [l.register_forward_pre_hook(grab_phi(i), with_kwargs=True) for i, l in enumerate(layers)]
    with torch.inference_mode():
        model(**first)
    for h in hs:
        h.remove()
    metrics["F_kernels"] = section_f(model, first, sink_idx)
    print("F done")
    metrics["E_latency"] = section_e(model, processor, items, inputs_list)
    print("E done")
    metrics["G_tokenization"] = section_g(processor)

    checksum1 = param_checksum(model)
    assert checksum0 == checksum1, "parameters changed"
    metrics["read_only_checksum_ok"] = True
    metrics["peak_vram_gib_overall"] = torch.cuda.max_memory_allocated() / 2**30

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT / "per_layer.json").write_text(json.dumps({"spans": spans, **per_layer}, indent=2, default=str))
    (OUT / "config.yaml").write_text(yaml.safe_dump({
        "model": MODEL_ID, "dtype": "bfloat16", "attn_implementation": "eager (sections A-D, F), sdpa+eager (E)",
        "probe_images": [{"flag": it["flag"], "index": it["index"], "label": it["label"]} for it in items],
        "prompt_template": {f: draft_prompt(f) for f in PROBE_PLAN}, "sink_thresholds": list(SINK_THRESH),
        "taus": list(TAUS), "head_taus": list(HEAD_TAUS), "rhos_paper": list(RHOS), "summs": list(SUMMS),
        "decode_steps": DECODE_STEPS, "latency_max_new_tokens": LAT_NEW_TOKENS,
        "torch": str(torch.__version__), "transformers": str(__import__("transformers").__version__),
        "gpu": torch.cuda.get_device_name(0)}, sort_keys=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
