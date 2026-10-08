"""D2 yes/no scoring (notes/diagnosis_amendments.md, Amendment 3b-c). A separate path: `scoring.py` stays the
letter path and is not used here.

One prompt per image and class c ("Does this {modality} show {class}? Answer yes or no.",
`effvar.prompt_variants.yesno_specs`). On its first generated token:
- p(yes), p(no): float32 softmax over the full vocabulary, summed over the Gemma token ids below (each form is
  a single token; tests/test_d2.py checks that every id decodes to its form and every form encodes to its id);
- s_c = p(yes) / (p(yes) + p(no)). It is computed as sigmoid(logsumexp(yes logits) - logsumexp(no logits)) in
  float64: the same number in real arithmetic, and it never divides 0 by 0 or rounds s near 1 to 1 early;
- leakage = 1 - p(yes) - p(no);
- compliant = the top first token is one of the 12 ids. An arm needs >= 0.99 of its K x n prompts compliant,
  else it is broken and not interpreted (Amendment 3c).
"""
import numpy as np
import torch

YES_TOKENS = {4443: "yes", 11262: " yes", 10784: "Yes", 8438: " Yes", 26915: "YES", 51327: " YES"}
NO_TOKENS = {1904: "no", 951: " no", 3771: "No", 2301: " No", 7018: "NO", 9424: " NO"}
YES_IDS, NO_IDS = tuple(YES_TOKENS), tuple(NO_TOKENS)
COMPLIANCE_MIN = 0.99

# Per-class fields of one prompt, stored as K-vectors (class order) in each image row.
PER_CLASS = ("s", "log_s", "p_yes", "p_no", "leakage", "top_id", "compliant", "text", "new_ids", "logits_sha1")


def yesno_first_token(logits: torch.Tensor) -> dict:
    z = logits.float()
    p = torch.softmax(z, dim=-1)
    p_yes, p_no = float(p[list(YES_IDS)].sum()), float(p[list(NO_IDS)].sum())
    d = (torch.logsumexp(z[list(YES_IDS)], 0) - torch.logsumexp(z[list(NO_IDS)], 0)).double()
    top = int(torch.argmax(z))
    return {"s": float(torch.sigmoid(d)), "log_s": float(torch.nn.functional.logsigmoid(d)),
            "p_yes": p_yes, "p_no": p_no, "leakage": 1.0 - p_yes - p_no,
            "top_id": top, "compliant": top in YES_TOKENS or top in NO_TOKENS}


def image_row(index: int, label: int, per_class: list[dict]) -> dict:
    """One image's row. `per_class[c]` belongs to the class-c prompt, so every vector is in class order;
    `pred` = argmax_c s_c (Amendment 3c)."""
    row = {"index": int(index), "label": int(label)}
    for key in PER_CLASS:
        row[key] = [d[key] for d in per_class]
    row["pred"] = int(np.argmax(row["s"]))
    return row


def yesno_summary(rows: list[dict]) -> dict:
    """Code-check quantities only (Amendment 4f): compliance, leakage and the range of s. No label is read."""
    s = np.array([r["s"] for r in rows], dtype=float)
    comp = np.array([r["compliant"] for r in rows], dtype=bool)
    leak = np.array([r["leakage"] for r in rows], dtype=float)
    tops = np.array([r["top_id"] for r in rows])
    bad = {}
    for t in tops[~comp].tolist():
        bad[str(t)] = bad.get(str(t), 0) + 1
    return {"n": len(rows), "k": int(s.shape[1]), "n_prompts": int(s.size),
            "compliance_rate": float(comp.mean()), "broken": bool(comp.mean() < COMPLIANCE_MIN),
            "noncompliant_top_ids": bad,
            "leakage_mean": float(leak.mean()), "leakage_max": float(leak.max()),
            "s_finite": bool(np.isfinite(s).all()), "s_min": float(np.nanmin(s)), "s_max": float(np.nanmax(s))}
