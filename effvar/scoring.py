"""Answer parsing, first-token option metrics and dataset metrics.

leakage = 1 - sum over options of p(option first token), softmax in float32 over the full vocabulary.
Each option's token set holds the first tokens of `label` and " " + `label` (whitespace-only tokens are
dropped); sets must be non-empty and disjoint.
"""
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

MATCH_SOURCES = ("label", "text", "fallback")


def option_token_sets(tokenizer, spec) -> list[list[int]]:
    sets = []
    for label in spec.labels:
        ids = set()
        for variant in (label, " " + label):
            first = tokenizer(variant, add_special_tokens=False)["input_ids"][0]
            if tokenizer.decode([first]).strip():
                ids.add(first)
        if not ids:
            raise ValueError(f"{spec.flag}: no usable first token for label {label!r}")
        sets.append(sorted(ids))
    flat = [i for s in sets for i in s]
    if len(flat) != len(set(flat)):
        raise ValueError(f"{spec.flag}: option token sets overlap: {sets}")
    return sets


def first_token_metrics(logits: torch.Tensor, token_sets: list[list[int]], label: int) -> dict:
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    p_option = [float(p[s].sum()) for s in token_sets]
    top = int(torch.argmax(logits))
    return {
        "p_option": p_option,
        "target_p": p_option[label],
        "entropy": float(-(p * logp).nan_to_num().sum()),
        "leakage": 1.0 - sum(p_option),
        "first_token_in_options": any(top in s for s in token_sets),
    }


_STRIP = re.compile(r"[*_`]")


def parse_answer(text: str, spec, p_option: list[float]) -> tuple[int, str]:
    """(predicted class index, match source). Label first, then unique option text, then probabilities."""
    clean = _STRIP.sub("", text).strip()
    labels = "|".join(re.escape(lab) for lab in spec.labels)
    m = re.match(rf"^(?:(?:grade|option|answer)\s*:?\s*)?({labels})(?![A-Za-z0-9])", clean, flags=re.IGNORECASE)
    if m and m.group(1) in spec.labels:  # labels themselves are case-sensitive
        return spec.labels.index(m.group(1)), "label"
    lower = clean.lower()
    hits = [i for i, t in enumerate(spec.option_texts) if t.lower() in lower]
    # drop hits that are substrings of another hit's text (none today, but keep matching unambiguous)
    hits = [i for i in hits if not any(i != j and spec.option_texts[i].lower() in spec.option_texts[j].lower()
                                       for j in hits)]
    if len(hits) == 1:
        return hits[0], "text"
    return int(np.argmax(p_option)), "fallback"


def dataset_metrics(rows: list[dict], n_classes: int, ordinal: bool) -> dict:
    y = [r["label"] for r in rows]
    yhat = [r["pred"] for r in rows]
    out = {
        "n": len(rows),
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "target_p_mean": float(np.mean([r["target_p"] for r in rows])),
        "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
        "leakage_mean": float(np.mean([r["leakage"] for r in rows])),
        "compliance_rate": float(np.mean([r["first_token_in_options"] for r in rows])),
        "match_source_share": {s: float(np.mean([r["match_source"] == s for r in rows])) for s in MATCH_SOURCES},
    }
    if ordinal:
        out["qwk"] = float(cohen_kappa_score(y, yhat, labels=list(range(n_classes)), weights="quadratic"))
    for key in ("vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "e2e_ms"):
        vals = [r[key] for r in rows if r[key] is not None]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_median"] = float(np.median(vals)) if vals else None
    out["peak_vram_gib_max"] = float(max(r["peak_vram_gib"] for r in rows))
    # New-token counts: H1 needs them to agree across methods (notes/validation_plan.md §2).
    n_new = [len(r["new_ids"]) for r in rows]
    out["new_tokens_median"] = float(np.median(n_new))
    out["new_tokens_max"] = int(max(n_new))
    return out
