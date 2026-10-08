"""One method on one dataset subset. Shared by scripts/run_eval.py and scripts/sweep_gate.py."""
from effvar.data import iter_samples, load_subset
from effvar.generation import MAX_NEW_TOKENS, run_one, run_untimed, warmup
from effvar.model import build_inputs, image_span
from effvar.prompts import get_spec
from effvar.scoring import dataset_metrics, first_token_metrics, option_token_sets, parse_answer
from effvar.var import DELTA_PHASES, delta_summary, sinks_summary, stats_summary

# Keys of stats_summary(), aggregated over images into metrics.json as var_<key>_mean.
_STAT_KEYS = ("edit_events", "rows_edited", "n_vis", "n_text")


def evaluate_dataset(model, processor, method, flag: str, subset_name: str, *, limit: int | None = None,
                     cache=None, warmed: bool = True, max_new_tokens: int = MAX_NEW_TOKENS, spec=None):
    """Returns (rows, metrics, extra). `cache` (a VisionCache) forces the untimed path.

    Timed runs warm up once per process; cached runs do not, since nothing is being measured.
    `spec` defaults to the frozen prompt; Phase 6 prompt variants (effvar/prompt_variants.py) pass their own.
    """
    spec, profile = (get_spec(flag) if spec is None else spec), None
    assert spec.flag == flag, (spec.flag, flag)
    subset = load_subset(flag, subset_name)
    token_sets = option_token_sets(processor.tokenizer, spec)
    rows, im_start, n_img = [], None, None

    for i, (idx, label, image) in enumerate(iter_samples(subset, limit=limit)):
        if not warmed:
            warmup(model, processor, method, spec, image)
            warmed = True
        if i == 0:  # image-token layout check, once per dataset (untimed)
            ids = build_inputs(processor, spec.messages(image), model.device)["input_ids"]
            im_start, n_img = image_span(ids, model.config)
        if cache is None:
            gen = run_one(model, processor, method, spec, image, max_new_tokens)
        else:
            with cache.sample(flag, subset.split, idx):
                gen = run_untimed(model, processor, method, spec, image, max_new_tokens)
        ft = first_token_metrics(gen.first_logits, token_sets, label)
        pred, source = parse_answer(gen.text, spec, ft["p_option"])
        state = getattr(method, "last_state", None)  # VAR methods only; read after the timed region
        if state is not None:
            state.finalise_stats()       # fold in the on-device rows_edited accumulator
        if state is not None and getattr(state, "timer", None) is not None:
            state.timer.flush()          # one sync per image, outside the measured region
            profile = _add_profile(profile, state.timer.summary())
        rows.append({"index": idx, "label": label, "pred": pred, "match_source": source, "text": gen.text,
                     "new_ids": gen.new_ids, "input_len": gen.input_len, "logits_sha1": gen.digest(),
                     **ft, **gen.timings(),
                     "var_stats": None if state is None else stats_summary(state.stats),
                     "var_delta": None if state is None else delta_summary(state.delta),
                     "var_sinks": None if state is None else sinks_summary(state.sink_sets)})
        if state is not None and getattr(state.cfg, "record_budget", False):  # Phase 6 diag runs only
            rows[-1]["var_budget"] = state.budget_summary()

    metrics = dataset_metrics(rows, n_classes=len(spec.labels), ordinal=flag == "retinamnist")
    metrics.update(var_stats_metrics(rows))
    metrics["var_delta_o_by_layer"] = var_delta_metrics(rows)
    metrics["var_profile"] = profile   # totals over the whole run; shares only, never a latency result
    metrics["vision_cache"] = cache is not None
    extra = {"image_span": {"start": im_start, "P": n_img}, "vision_cache": cache is not None}
    if cache is not None:
        extra["vision_cache_stats"] = cache.stats()
    return rows, metrics, extra, spec, subset, token_sets


def _add_profile(total: dict | None, one: dict) -> dict:
    """Sum per-image `RegionTimer.summary()` dicts. The timer is per generation, so without this only
    the last image would be recorded."""
    if total is None:
        return {"ms": {r: dict(p) for r, p in one["ms"].items()},
                "by_layer": {r: dict(p) for r, p in one["by_layer"].items()},
                "calls": dict(one["calls"]), "images": 1}
    for r, phases in one["ms"].items():
        for phase, ms in phases.items():
            total["ms"][r][phase] = total["ms"][r].get(phase, 0.0) + ms
    for r, layers in one["by_layer"].items():
        for layer, ms in layers.items():
            total["by_layer"][r][layer] = total["by_layer"][r].get(layer, 0.0) + ms
    for r, c in one["calls"].items():
        total["calls"][r] = total["calls"].get(r, 0) + c
    total["images"] += 1
    return total


def var_delta_metrics(rows: list[dict]) -> dict | None:
    """Mean over images of each layer's `delta_o_rms` and `o_rms` (H2: which layers the edit moves most).

    None unless the run was instrumented with `record_delta`. `ratio` is delta_o_rms / o_rms, the
    scale-free form -- layer norms grow with depth, so the raw norm alone would rank layers by depth.
    """
    per_layer: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        for layer, d in (r.get("var_delta") or {}).items():
            fields = per_layer.setdefault(layer, {})
            for key, val in d.items():
                fields.setdefault(key, []).append(val)
    if not per_layer:
        return None
    out = {}
    for layer, fields in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
        row = {k: sum(v) / len(v) for k, v in fields.items()}
        for phase in DELTA_PHASES:
            dz, oz = row.get(f"{phase}_delta_o_rms"), row.get(f"{phase}_o_rms")
            if dz is not None and oz:
                row[f"{phase}_ratio"] = dz / oz
        row["n_images"] = max(len(v) for v in fields.values())
        out[layer] = row
    return out


def var_stats_metrics(rows: list[dict]) -> dict:
    """Mean over images of each `stats_summary` field -> `var_<key>_mean`; all None for `base`, and
    n_vis/n_text None for a method that edits nothing."""
    out = {}
    for key in _STAT_KEYS:
        vals = [r["var_stats"][key] for r in rows
                if r.get("var_stats") is not None and r["var_stats"][key] is not None]
        out[f"var_{key}_mean"] = float(sum(vals) / len(vals)) if vals else None
    return out
