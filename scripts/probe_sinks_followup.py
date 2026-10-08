"""Phase 2a follow-up (read-only): characterise visual sinks on MedGemma after the paper's detector degenerated.

probe_medgemma.py found that one hidden dimension carries almost all of every token's norm from layer 2 on,
so `max_{d in D} |rmsnorm(x)_d| > tau` marks every token a sink. This script, on the same 20 probe images:
  1. measures that dominant dimension's norm share per layer and token type, before and after input_layernorm;
  2. repeats sink-dim discovery with the dominant dimension excluded from the RMS;
  3. defines visual sinks directly from attention (image tokens receiving >= SHARE_X times the mean image-token
     attention from post-image queries), and reports which hidden dims separate them, where they sit on the
     16x16 grid, their value norms, and HeadFork gate fractions under that definition and under tau=50 on the
     dominant dim.
Writes results/probe_medgemma/followup.json.
`--calibrate` instead scores a dominant-dim-excluded detector against the attention-defined sinks
(followup_calibration.json).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_medgemma as pm  # noqa: E402

SHARE_X = 10.0  # attention sink: >= 10x the mean per-image-token attention
DOM_TAU = 50.0
RHOS, SUMMS = pm.RHOS, pm.SUMMS


def main():
    items = pm.probe_images()
    processor = pm.AutoProcessor.from_pretrained(pm.MODEL_ID)
    model = pm.AutoModelForImageTextToText.from_pretrained(pm.MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                           attn_implementation="eager").eval()
    checksum0 = pm.param_checksum(model)
    metrics = json.loads((pm.OUT / "metrics.json").read_text())
    dom = metrics["B_sink_dims"]["dims_by_token_count_layers2plus"]["gt50"][0]["dim"]
    tc = model.config.text_config
    L, D = tc.num_hidden_layers, tc.hidden_size
    layers = model.model.language_model.layers

    per_layer = {l: {"dom_share_bos": [], "dom_share_text_med": [], "dom_share_vis_min": [], "dom_share_vis_med": [],
                     "dom_share_post_norm_vis_med": [], "n_attn_sinks": [], "attn_sink_share_of_image": [],
                     "img_attn_mean": [], "vnorm_ratio_attn_sinks": [], "jaccard_attn_vs_dom50": [],
                     "n_dom50_vis_sinks": [], **{f"gate_attn_rho{r}_summ{s}": [] for r in RHOS for s in SUMMS},
                     **{f"gate_dom50_rho{r}_summ{s}": [] for r in RHOS for s in SUMMS}} for l in range(2, L)}
    sep_sum = torch.zeros(L, D)  # (mean_sink - mean_nonsink) / pooled std, per dim, excl. dominant dim
    sep_n = torch.zeros(L)
    bos_top_excl = Counter()
    bos_vals_excl = {}
    vis_massive_excl = Counter()  # dims with |r| > 20 on visual tokens (excl. dominant dim)
    sink_positions = Counter()
    st = {}

    def pre(l):
        def hook(mod, args, kwargs):
            if l < 2:
                return
            x = (args[0] if args else kwargs["hidden_states"])[0].float()
            im, P = st["im"], st["P"]
            share = x[:, dom].abs() / x.norm(dim=-1)
            rec = per_layer[l]
            rec["dom_share_bos"].append(float(share[0]))
            text = torch.ones_like(share, dtype=torch.bool)
            text[im:im + P] = False
            rec["dom_share_text_med"].append(float(share[text].median()))
            rec["dom_share_vis_min"].append(float(share[im:im + P].min()))
            rec["dom_share_vis_med"].append(float(share[im:im + P].median()))
            x_ex = x.clone()
            x_ex[:, dom] = 0
            r = (x_ex * torch.rsqrt(x_ex.pow(2).mean(-1, keepdim=True) + 1e-6)).abs()
            top = torch.topk(r[0], 4)
            for v, d in zip(top.values.tolist(), top.indices.tolist()):
                bos_top_excl[d] += 1
                bos_vals_excl.setdefault(d, []).append(v)
            for d in torch.nonzero((r[im:im + P] > 20).any(0)).flatten().tolist():
                vis_massive_excl[d] += 1
            st["r_vis"] = r[im:im + P]
            st["phi_dom"] = pm.rmsnorm_abs(x)[:, dom]
        return hook

    def post_norm(l):
        def hook(mod, args, out):
            if l < 2:
                return
            y = out[0].float()
            im, P = st["im"], st["P"]
            per_layer[l]["dom_share_post_norm_vis_med"].append(
                float((y[im:im + P, dom].abs() / y[im:im + P].norm(dim=-1)).median()))
        return hook

    def vproj(l):
        def hook(mod, args, out):
            v = out[0].float().view(out.shape[1], tc.num_key_value_heads, tc.head_dim)
            st["vnorm"] = v.norm(dim=-1).mean(-1)
        return hook

    def attn(l):
        def hook(mod, args, out):
            if l < 2 or out[1] is None:
                return
            im, P = st["im"], st["P"]
            w = out[1][0].float()[:, im + P:]  # post-image queries [H, Q, T]
            img = w[..., im:im + P]
            a = img.mean((0, 1))
            rel = a * P / a.sum()
            rec = per_layer[l]
            rec["img_attn_mean"].append(float(img.sum(-1).mean()))
            attn_sink = rel >= SHARE_X
            dom_sink = st["phi_dom"][im:im + P] > DOM_TAU
            k = int(attn_sink.sum())
            rec["n_attn_sinks"].append(k)
            rec["n_dom50_vis_sinks"].append(int(dom_sink.sum()))
            rec["attn_sink_share_of_image"].append(float(a[attn_sink].sum() / a.sum()))
            union = (attn_sink | dom_sink).sum()
            rec["jaccard_attn_vs_dom50"].append(float((attn_sink & dom_sink).sum() / union) if union else float("nan"))
            for idx in torch.nonzero(attn_sink).flatten().tolist():
                sink_positions[idx] += 1
            if 0 < k < P:
                vn = st["vnorm"][im:im + P]
                rec["vnorm_ratio_attn_sinks"].append(float(vn[attn_sink].mean() / vn[~attn_sink].mean()))
                r = st["r_vis"]
                ms, mn = r[attn_sink].mean(0), r[~attn_sink].mean(0)
                sd = torch.sqrt((r[attn_sink].var(0, unbiased=False) + r[~attn_sink].var(0, unbiased=False)) / 2 + 1e-6)
                sep_sum[l] += ((ms - mn) / sd).cpu()
                sep_n[l] += 1
            for name, sink in (("attn", attn_sink), ("dom50", dom_sink)):
                summation = img.sum(-1)
                portion = (img * sink).sum(-1) / (img + 1e-6).sum(-1)
                for rho in RHOS:
                    for summ in SUMMS:
                        keep = ((portion <= 1 - rho) & (summation >= summ)) if sink.any() else torch.zeros_like(portion, dtype=torch.bool)
                        rec[f"gate_{name}_rho{rho}_summ{summ}"].append(float(keep.float().mean()))
        return hook

    hs = []
    for l, layer in enumerate(layers):
        hs.append(layer.register_forward_pre_hook(pre(l), with_kwargs=True))
        hs.append(layer.input_layernorm.register_forward_hook(post_norm(l)))
        hs.append(layer.self_attn.v_proj.register_forward_hook(vproj(l)))
        hs.append(layer.self_attn.register_forward_hook(attn(l)))
    for it in items:
        inp = pm.build_inputs(processor, it, model.device)
        pos = torch.nonzero(inp["input_ids"][0] == model.config.image_token_index).flatten()
        st.update(im=int(pos[0]), P=int(pos.numel()))
        with torch.inference_mode():
            model(**inp)
    for h in hs:
        h.remove()
    assert pm.param_checksum(model) == checksum0, "parameters changed"

    def mean(v):
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else None

    table = {l: {k: mean(v) for k, v in rec.items()} for l, rec in per_layer.items()}
    types = tc.layer_types
    by_type = {kind: {k: mean([table[l][k] for l in table if types[l] == kind and table[l][k] is not None])
                      for k in next(iter(table.values()))} for kind in ("sliding_attention", "full_attention")}
    sep_mean = sep_sum.sum(0) / sep_n.sum().clamp_min(1)
    top_sep = torch.topk(sep_mean, 8)
    n_obs = (L - 2) * len(items)
    out = {
        "dominant_dim": dom, "share_x": SHARE_X, "dom_tau": DOM_TAU, "read_only_checksum_ok": True,
        "by_layer_type": by_type, "per_layer": table,
        "bos_top4_excl_dominant": [{"dim": d, "freq": c / n_obs, "bos_value_mean": float(np.mean(bos_vals_excl[d]))}
                                   for d, c in bos_top_excl.most_common(8)],
        "dims_massive_on_visual_excl_dominant": [{"dim": d, "layer_image_obs": c} for d, c in vis_massive_excl.most_common(8)],
        "dims_separating_attn_sinks": [{"dim": int(d), "mean_effect_size": float(v)}
                                       for v, d in zip(top_sep.values.tolist(), top_sep.indices.tolist())],
        "attn_sink_positions_top10": [{"image_index": i, "row": i // 16, "col": i % 16, "count": c}
                                      for i, c in sink_positions.most_common(10)],
        "attn_sink_observations_total": int(sum(sink_positions.values())),
        "layer_image_observations": n_obs,
    }
    (pm.OUT / "followup.json").write_text(json.dumps(out, indent=2))
    print("wrote followup.json")


def calibrate(d_sets=((1313,), (1313, 2511), (634, 1313, 2511)), taus=(5, 10, 15, 20, 30, 40)):
    """Precision/recall of phi'(x) = max_{d in D} |x_d| / RMS(x with the dominant dim zeroed) > tau
    against attention-defined visual sinks, layers >= 2, same 20 images. Label-free (no class labels)."""
    items = pm.probe_images()
    processor = pm.AutoProcessor.from_pretrained(pm.MODEL_ID)
    model = pm.AutoModelForImageTextToText.from_pretrained(pm.MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                           attn_implementation="eager").eval()
    checksum0 = pm.param_checksum(model)
    dom = json.loads((pm.OUT / "followup.json").read_text())["dominant_dim"]
    L = model.config.text_config.num_hidden_layers
    layers = model.model.language_model.layers
    stats = {(ds, t): {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "text_pred": 0} for ds in d_sets for t in taus}
    st = {}

    def pre(l):
        def hook(mod, args, kwargs):
            if l >= 2:
                x = (args[0] if args else kwargs["hidden_states"])[0].float()
                x[:, dom] = 0
                st["r"] = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).abs()
        return hook

    def attn(l):
        def hook(mod, args, out):
            if l < 2:
                return
            im, P = st["im"], st["P"]
            img = out[1][0].float()[:, im + P:, im:im + P]
            a = img.mean((0, 1))
            label = a * P / a.sum() >= SHARE_X
            text = torch.ones(st["r"].shape[0], dtype=torch.bool, device=label.device)
            text[im:im + P] = False
            for ds in d_sets:
                phi = st["r"][:, list(ds)].max(-1).values
                for t in taus:
                    pred = phi[im:im + P] > t
                    s = stats[(ds, t)]
                    s["tp"] += int((pred & label).sum())
                    s["fp"] += int((pred & ~label).sum())
                    s["fn"] += int((~pred & label).sum())
                    s["n_pred"] += int(pred.sum())
                    s["text_pred"] += int((phi[text] > t).sum())
        return hook

    hs = [h for l, layer in enumerate(layers) for h in (
        layer.register_forward_pre_hook(pre(l), with_kwargs=True), layer.self_attn.register_forward_hook(attn(l)))]
    for it in items:
        inp = pm.build_inputs(processor, it, model.device)
        pos = torch.nonzero(inp["input_ids"][0] == model.config.image_token_index).flatten()
        st.update(im=int(pos[0]), P=int(pos.numel()))
        with torch.inference_mode():
            model(**inp)
    for h in hs:
        h.remove()
    assert pm.param_checksum(model) == checksum0, "parameters changed"
    n_obs = (L - 2) * len(items)
    rows = []
    for (ds, t), s in stats.items():
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        rows.append({"dims": list(ds), "tau": t, "precision": prec, "recall": rec,
                     "vis_sinks_per_layer": s["n_pred"] / n_obs, "text_sinks_per_layer": s["text_pred"] / n_obs})
    (pm.OUT / "followup_calibration.json").write_text(json.dumps(
        {"dominant_dim_excluded": dom, "share_x": SHARE_X, "rows": rows}, indent=2))
    print("wrote followup_calibration.json")


if __name__ == "__main__":
    calibrate() if "--calibrate" in sys.argv else main()
