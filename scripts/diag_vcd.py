"""Card V1 analysis (G5): every CPU arm and rule of notes/vcd_plan.md card V1 + Amendment V1-1. CPU only.

    python scripts/diag_vcd.py [--jobs 6] [--token-note "<dated V1- note>"]
    python scripts/diag_vcd.py --check-inputs     # allowed before G4: reads no G4 output and computes no metric

Gate (guard 6, no peeking). It refuses unless:
- every registered G4 run exists with its registered rows, arm, step, rep and per-row seeds;
- every run was clean at start and end, with code trees equal to the code commit in the plan index (G2 row);
- check_vcd_regression's guard7 and guard8 records (and the G3 records guard1 and guard3) say PASS for that code
  commit, and every file they read is unchanged (sha256).

If the G4 noised runs used max_new_tokens != 8, --token-note must cite the dated V1- note.

Step 1, before anything is written: VCD at alpha = 0 must reproduce base's decisions, and VCD + R must reproduce R's
(both branches, beta 0.1 and 0.2), with OvO(VCD_0) = OvO(base) exactly. Otherwise abort.

Scores:
- clean side z = log(clip(p_option renormalised, EPS)), as Phase 6;
- noised side z' = the stored fp32 logp_option renormalised in log space, unclipped.

Arms, contrasts and rules are those of the card. Arms outside the card are labelled "added at G2, report-only". The
paired bootstrap uses 10,000 resamples, seed 0, and one resample set per dataset shared by all arms. Fits are held
fixed, so the CIs cover image sampling only.

Writes results/vcd/V1/{vcd.json, vcd_report.md}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from effvar import FLAGS, ROOT
from effvar.diag_stats import (DATASETS, EPS, apply_bias, boot_idx, ci, fit_bias, load_run, ovo_logratio_auc,
                               primary, ranking_auc)
from effvar.vcd import (ALPHAS, B2_C, BETA, BETA_SENS, CF_ALPHAS, DECIDING_ALPHAS, FOLD_SHA1, MDE80_TUNE,
                        MISMATCH_ROWS, PLAN_INDEX, SAVED_RUNS, V1_EXP, a1_folds, boot_p, code_commit_from_index,
                        code_trees_at, decide, fold_sha1, g4_specs, git, holm, js, lex_score, log_clean, log_noised,
                        ovo_scores_auc, per_token_score, renyi, selection, sha256_file, softmax, split_logw, v_head,
                        vcd_score, vcd_seed)

RESULTS = ROOT / "results"
OUT_REL = "vcd/V1"
CARD, ADDED = "card", "added at G2, report-only"
L = "ABCDEFG"
CF_NAMES = ("gray", "black", "tune_mean")
GUARD_RECORDS = ("guard1", "guard3", "guard7", "guard8")   # E1 needs 7 and 8; 1 and 3 (G3) are required too


def read_rows(results_dir: Path, exp: str, flag: str) -> dict[int, dict]:
    with open(Path(results_dir) / exp / flag / "predictions.jsonl", encoding="utf-8") as f:
        return {int(r["index"]): r for r in map(json.loads, f)}


def read_config(results_dir: Path, exp: str, flag: str) -> dict:
    return yaml.safe_load((Path(results_dir) / exp / flag / "config.yaml").read_text(encoding="utf-8"))


def tune_subsets() -> dict[str, list[int]]:
    from effvar.data import load_subset
    return {f: list(load_subset(f, "tune").indices) for f in FLAGS}


# --------------------------------------------------------------------------- gate (guard 6, E1, E3)
def check_ready(results_dir: Path, code_commit: str, trees: dict, subsets: dict) -> dict:
    """Refuse (SystemExit listing every problem) unless G4 is complete, clean, from the code commit, and guards 7/8
    passed on exactly these files. Returns {"layout", "max_new_tokens"}."""
    results_dir = Path(results_dir)
    problems = []
    layouts = set()
    for flag in FLAGS:
        p = results_dir / V1_EXP / "clean_base" / flag / "config.yaml"
        if p.exists():
            layouts.add(yaml.safe_load(p.read_text(encoding="utf-8")).get("clean_rerun"))
    layout = layouts.pop() if len(layouts) == 1 else None
    if layout not in ("full", "fallback"):
        problems.append(f"clean_base: missing, or no single recorded --clean-rerun (got {sorted(map(str, layouts))})")
    mnt = set()
    for name, sub in g4_specs(layout or "full").items():
        for flag, idx in selection(sub, subsets).items():
            d = results_dir / sub.exp / flag
            if not ((d / "predictions.jsonl").exists() and (d / "config.yaml").exists()):
                problems.append(f"{name} {flag}: missing")
                continue
            rows, cfg = read_rows(results_dir, sub.exp, flag), read_config(results_dir, sub.exp, flag)
            if sorted(rows) != sorted(idx):
                problems.append(f"{name} {flag}: {len(rows)} rows, registered {len(idx)}")
            sr = cfg.get("subrun") or {}
            if (sr.get("arm"), sr.get("mode"), sr.get("step"), sr.get("rep")) != (sub.arm, sub.mode, sub.step, sub.rep):
                problems.append(f"{name} {flag}: recorded arm/mode/step/rep differ from the registered sub-run")
            if sub.mode == "noised":
                mnt.add(cfg["decoding"]["max_new_tokens"])
                split = cfg["subset"]["split"]
                if any(r["seed"] != vcd_seed(flag, split, i, sub.rep) or r["step"] != sub.step for i, r in rows.items()):
                    problems.append(f"{name} {flag}: a row's seed or step differs from the card's formula")
            for when in ("git", "git_end"):
                g = cfg.get(when) or {}
                if g.get("dirty") or g.get("code_changes") or g.get("code_trees") != trees:
                    problems.append(f"{name} {flag} {when}: not clean, or code differs from the code commit")
    if len(mnt) > 1:
        problems.append(f"G4 noised runs mix max_new_tokens {sorted(mnt)}")
    for mode in GUARD_RECORDS:
        path = results_dir / OUT_REL / "regression" / f"{mode}.json"
        if not path.exists():
            problems.append(f"{mode}: no result (run scripts/check_vcd_regression.py {mode})")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("ok") is not True:
            problems.append(f"{mode}: did not pass")
        if rec.get("code_commit") != code_commit:
            problems.append(f"{mode}: checked code commit {rec.get('code_commit')}, not {code_commit}")
        stale = [rel for rel, sha in (rec.get("inputs") or {}).items()
                 if not (results_dir.parent / rel).exists() or sha256_file(results_dir.parent / rel) != sha]
        if not rec.get("inputs") or stale:
            problems.append(f"{mode}: stale or empty record ({len(stale)} file(s) changed since it ran)")
    if problems:
        raise SystemExit("diag_vcd refuses (guard 6: G4 incomplete or unchecked):\n  " + "\n  ".join(problems))
    return {"layout": layout, "max_new_tokens": mnt.pop()}


# --------------------------------------------------------------------------- inputs
def exclusions(flag: str) -> set[int]:
    return set(json.loads((ROOT / f"data/subsets/{flag}_tune_overlap.json").read_text())["exclude_tune_indices"])


def p_cf(flag: str, model: str) -> np.ndarray:
    cf = json.loads((RESULTS / "diag/B1/content_free.json").read_text())
    ps = []
    for name in CF_NAMES:
        p = np.asarray(cf[flag][model][name]["p_option"], dtype=float)
        ps.append(p / p.sum())
    return np.mean(ps, 0)


def saved_mismatch(results_dir: Path) -> dict:
    """Rows where the saved text-parsed pred differs from argmax_c P_c (all tune rows), per arm and dataset."""
    out = {}
    for arm, run in SAVED_RUNS.items():
        for flag in FLAGS:
            rows = read_rows(results_dir, run, flag)
            bad = tuple(i for i in sorted(rows) if rows[i]["pred"] != int(np.argmax(rows[i]["p_option"])))
            if bad:
                out.setdefault(arm, {})[flag] = bad
    return out


def check_inputs(results_dir: Path = RESULTS) -> int:
    """CPU pre-checks that read no G4 output and compute no metric."""
    ok, lines = True, ["diag_vcd --check-inputs (no G4 output read, no metric computed)"]
    b2 = json.loads((results_dir / "diag/B2/b2.json").read_text())
    for flag in FLAGS:
        k, _ = DATASETS[flag]
        base = load_run(results_dir / SAVED_RUNS["base"], flag)
        idx = sorted(set(base) - exclusions(flag))
        _, folds = a1_folds(np.array([base[i]["label"] for i in idx]))
        sha = fold_sha1(folds)
        c = b2[flag]["selected"]["opt"]["C"]
        cf = [len(p_cf(flag, m)) == k for m in ("base", "R")]
        good = sha == FOLD_SHA1[flag] and c == B2_C[flag] and all(cf)
        ok &= good
        lines.append(f"  {flag}: n {len(idx)}, fold sha1 {sha} (pinned {FOLD_SHA1[flag]}); B2 C {c} "
                     f"(registered {B2_C[flag]}), AUC_opt {b2[flag]['selected']['opt']['tune_ovo']:.4f}; "
                     f"B1 p_cf base/R present {cf} -> {'OK' if good else 'FAIL'}")
    found = saved_mismatch(results_dir)
    same = found == MISMATCH_ROWS
    ok &= same
    lines.append(f"  saved pred != argmax P_c: {found} -> {'= registered (V1-1)' if same else 'DIFFERS from V1-1'}")
    lines.append("CHECK-INPUTS OK" if ok else "CHECK-INPUTS FAIL")
    print("\n".join(lines))
    return 0 if ok else 1


def load_dataset(results_dir: Path, flag: str, subsets: dict, layout: str) -> dict:
    """Aligned arrays for one dataset (Amendment 1 exclusions; rows in sorted index order)."""
    k, ordinal = DATASETS[flag]
    specs = g4_specs(layout)
    raw = {"base": read_rows(results_dir, SAVED_RUNS["base"], flag), "R": read_rows(results_dir, SAVED_RUNS["var"], flag),
           "b500": read_rows(results_dir, specs["base_s500"].exp, flag),
           "R500": read_rows(results_dir, specs["R_s500"].exp, flag),
           "b999": read_rows(results_dir, specs["base_s999"].exp, flag),
           "rep2": read_rows(results_dir, specs["base_s500_rep2"].exp, flag),
           "cbase": read_rows(results_dir, specs["clean_base"].exp, flag)}
    if flag in MISMATCH_ROWS["var"]:
        raw["cR"] = read_rows(results_dir, specs["clean_R"].exp, flag)
    idx = sorted(set(subsets[flag]) - exclusions(flag))
    for name in ("base", "R", "b500", "R500", "b999"):
        assert set(idx) <= set(raw[name]), (flag, name)
    y = np.array([raw["base"][i]["label"] for i in idx])
    for name, rows in raw.items():
        assert all(rows[i]["label"] == raw["base"][i]["label"] for i in idx if i in rows), (flag, name)
    splits, folds = a1_folds(y)
    if fold_sha1(folds) != FOLD_SHA1[flag]:
        raise SystemExit(f"{flag}: A1 fold sha1 {fold_sha1(folds)} != pinned {FOLD_SHA1[flag]}")
    b2 = json.loads((RESULTS / "diag/B2/b2.json").read_text())
    if b2[flag]["selected"]["opt"]["C"] != B2_C[flag]:
        raise SystemExit(f"{flag}: B2's C differs from the registered value")

    D = {"flag": flag, "k": k, "ordinal": ordinal, "idx": np.array(idx), "y": y, "splits": splits,
         "b2_auc_opt": float(b2[flag]["selected"]["opt"]["tune_ovo"])}
    for name, run in (("b", SAVED_RUNS["base"]), ("R", SAVED_RUNS["var"])):
        lr = load_run(results_dir / run, flag)
        rows = raw["base" if name == "b" else "R"]
        D[f"p_{name}"] = np.stack([lr[i]["p"] for i in idx])
        D[f"z_{name}"] = log_clean(D[f"p_{name}"])
        D[f"P_{name}"] = np.array([rows[i]["p_option"] for i in idx], dtype=float)
        D[f"leak_{name}"] = np.array([rows[i]["leakage"] for i in idx], dtype=float)
        D[f"text_{name}"] = np.array([rows[i]["pred"] for i in idx])
    for name in ("b500", "R500", "b999"):
        rows = raw[name]
        D[f"lp_{name}"] = np.array([rows[i]["logp_option"] for i in idx], dtype=float)
        D[f"zp_{name}"] = log_noised(D[f"lp_{name}"])
        D[f"Pn_{name}"] = np.array([rows[i]["p_option"] for i in idx], dtype=float)
        D[f"leakn_{name}"] = np.array([rows[i]["leakage"] for i in idx], dtype=float)
        D[f"ftio_{name}"] = np.array([rows[i]["first_token_in_options"] for i in idx])
        D[f"nnew_{name}"] = np.array([len(rows[i]["new_ids"]) for i in idx])
        D[f"textn_{name}"] = np.array([rows[i]["pred"] for i in idx])
    D["lpid_b500"] = np.array([raw["b500"][i]["logp_ids"] for i in idx], dtype=float)
    pos = {i: n for n, i in enumerate(idx)}
    rep = [i for i in subsets[flag][:50] if i in pos]
    D["rep_pos"] = np.array([pos[i] for i in rep], dtype=int)
    D["zp_rep2"] = log_noised([raw["rep2"][i]["logp_option"] for i in rep])
    cpos = [i for i in idx if i in raw["cbase"]]
    for i in cpos:
        assert raw["cbase"][i]["p_option"] == raw["base"][i]["p_option"], (flag, i, "guard 7")
    D["c_pos"] = np.array([pos[i] for i in cpos], dtype=int)
    D["pid_clean"] = np.array([raw["cbase"][i]["p_ids"] for i in cpos], dtype=float)
    D["pcf_b"], D["pcf_R"] = p_cf(flag, "base"), p_cf(flag, "R")
    D["mismatch"] = mismatch_splits(raw, flag)
    return D


def mismatch_splits(raw: dict, flag: str) -> list[dict]:
    """The " X" / "X" split of the V1-1 mismatch rows (clean re-run; the noised view beside)."""
    out = []
    for arm, by_flag in MISMATCH_ROWS.items():
        clean, noised, saved = ((raw["cbase"], raw["b500"], raw["base"]) if arm == "base"
                                else (raw.get("cR", {}), raw["R500"], raw["R"]))
        for i in by_flag.get(flag, ()):
            r = saved[i]
            letters = {"saved_pred": r["pred"], "argmax_P": int(np.argmax(r["p_option"]))}
            rec = {"arm": "R" if arm == "var" else "base", "index": i, "text": r["text"]}
            for role, c in letters.items():
                pc, pn = clean[i]["p_ids"][c], noised[i]["p_ids"][c]
                rec[role] = {"letter": L[c], "P": r["p_option"][c], "clean_p_space": pc[0], "clean_p_bare": pc[1],
                             "clean_w_space": pc[0] / (pc[0] + pc[1]), "noised_w_space": pn[0] / (pn[0] + pn[1])}
            out.append(rec)
    return out


# --------------------------------------------------------------------------- step 1 (E5)
def alpha0_failures(data: dict) -> list[str]:
    """Both alpha = 0 identities on the real arrays, for every dataset, branch and beta."""
    bad = []
    for flag, D in data.items():
        k, y = D["k"], D["y"]
        for beta in (BETA, BETA_SENS):
            for branch, zps in (("b", ("zp_b500", "zp_b999")), ("R", ("zp_b500", "zp_R500"))):
                mask, _, empty = v_head(D[f"P_{branch}"], D[f"leak_{branch}"], beta)
                clean = D[f"P_{branch}"].argmax(1)
                for zp in zps:
                    pred = decide(vcd_score(D[f"z_{branch}"], D[zp], 0.0), mask, empty, clean)
                    if not np.array_equal(pred, clean):
                        bad.append(f"{flag}: {'VCD' if branch == 'b' else 'VCD + R'} at alpha 0 ({zp}, beta {beta}) "
                                   f"differs from {'base' if branch == 'b' else 'R'} on {int((pred != clean).sum())} rows")
        if ovo_scores_auc(y, vcd_score(D["z_b"], D["zp_b500"], 0.0), k) != ovo_logratio_auc(y, D["p_b"], k):
            bad.append(f"{flag}: OvO(VCD_0) != OvO(base)")
    return bad


# --------------------------------------------------------------------------- arms
class Arm:
    def __init__(self, name, origin=CARD, role="report-only", pred=None, scores=None, probs=None, boot=()):
        self.name, self.origin, self.role = name, origin, role
        self.pred, self.scores, self.probs, self.boot = pred, scores, probs, tuple(boot)

    def metric_names(self):
        return (["primary"] if self.pred is not None else []) + \
               (["ovo", "ovr"] if self.scores is not None or self.probs is not None else [])


def metric_value(m, y, pred, scores, probs, k, ordinal, sm=None):
    if m == "primary":
        return primary(y, pred, k, ordinal)
    if m == "ovo":
        return ovo_logratio_auc(y, probs, k) if probs is not None else ovo_scores_auc(y, scores, k)
    return ranking_auc(y, probs if probs is not None else sm, k)


def boot_arm(pred, scores, probs, metrics, y, idx, k, ordinal) -> dict:
    sm = softmax(scores) if probs is None and scores is not None else None
    out = {m: np.empty(len(idx)) for m in metrics}
    for b, ix in enumerate(idx):
        for m in metrics:
            out[m][b] = metric_value(m, y[ix], None if pred is None else pred[ix],
                                     None if scores is None else scores[ix], None if probs is None else probs[ix],
                                     k, ordinal, None if sm is None else sm[ix])
    return out


def cal_decide(u, mask, empty, clean, splits, y, k):
    """VCD_cal: A1's NLL offsets fitted on softmax(u) per fold, applied to the held-out fold; decision inside V_head."""
    pred, S = np.empty(len(y), dtype=int), np.empty_like(u)
    for tr, te in splits:
        b = fit_bias(softmax(u[tr]), y[tr], k)
        S[te] = u[te] + b
        pred[te] = decide(S[te], mask[te], empty[te], clean[te])
    return pred, S


def cal_probs(p, splits, y, k):
    pc = np.empty_like(p)
    for tr, te in splits:
        pc[te] = apply_bias(p[te], fit_bias(p[tr], y[tr], k))
    return pc


def scaler_oof(X, y, k, C, splits):
    """Change 2: StandardScaler -> class-balanced L2 logistic regression at B2's C, A1's folds, out-of-fold."""
    oof = np.zeros((len(y), k))
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))
    return oof


def build_arms(D: dict) -> dict[str, Arm]:
    k, y, splits = D["k"], D["y"], D["splits"]
    arms = {}

    def add(a):
        arms[a.name] = a

    full = ("primary", "ovo", "ovr")
    clean = {br: D[f"P_{br}"].argmax(1) for br in ("b", "R")}
    vh = {(br, beta): v_head(D[f"P_{br}"], D[f"leak_{br}"], beta) for br in ("b", "R") for beta in (BETA, BETA_SENS)}
    add(Arm("base", role="comparator", pred=clean["b"], probs=D["p_b"], boot=full))
    add(Arm("base_text", role="reconciliation (V1-1(1)): saved text pred", pred=D["text_b"], boot=("primary",)))
    pcb = cal_probs(D["p_b"], splits, y, k)
    add(Arm("base_cal", role="comparator", pred=pcb.argmax(1), probs=pcb, boot=("primary",)))
    add(Arm("R", role="comparator", pred=clean["R"], probs=D["p_R"], boot=full))
    add(Arm("R_text", role="reconciliation (V1-1(1)): saved text pred", pred=D["text_R"], boot=("primary",)))
    pcr = cal_probs(D["p_R"], splits, y, k)
    add(Arm("R_cal", role="comparator", pred=pcr.argmax(1), probs=pcr, boot=("primary",)))

    def vcd_family(name, br, zp, alpha, role, cal_origin, with_extras=False):
        mask, _, empty = vh[(br, BETA)]
        u = vcd_score(D[f"z_{br}"], zp, alpha)
        add(Arm(name, role=role, pred=decide(u, mask, empty, clean[br]), scores=u, boot=full))
        pred_c, S = cal_decide(u, mask, empty, clean[br], splits, y, k)
        add(Arm(f"{name}_cal", origin=cal_origin, pred=pred_c, scores=S, boot=("primary",)))
        if with_extras:
            m2, _, e2 = vh[(br, BETA_SENS)]
            add(Arm(f"{name}@beta0.2", role="report-only (beta sensitivity)", pred=decide(u, m2, e2, clean[br]),
                    boot=("primary",)))
            add(Arm(f"{name}_lex", role="report-only (APC-lexicographic)", scores=lex_score(u, D[f"z_{br}"], mask),
                    boot=("ovo",)))

    for a in ALPHAS:
        role = "deciding" if a in DECIDING_ALPHAS else "report-only"
        vcd_family(f"VCD_{a}", "b", D["zp_b500"], a, role, CARD, with_extras=True)
        mask, _, empty = vh[("b", BETA)]
        um = vcd_score(D["z_b"], D["zp_b500"].mean(0), a)
        add(Arm(f"VCD_mean_{a}", origin=CARD if a in DECIDING_ALPHAS else ADDED, role="control (OvO = base exactly)",
                pred=decide(um, mask, empty, clean["b"]), scores=um, boot=("primary",)))
        vcd_family(f"VCD+R_clean_{a}", "R", D["zp_b500"], a, "report-only", ADDED)
        vcd_family(f"VCD+R_both_{a}", "R", D["zp_R500"], a, "report-only", ADDED)
        vcd_family(f"VCD999_{a}", "b", D["zp_b999"], a, "report-only", ADDED)
    for a in CF_ALPHAS:
        mask, _, empty = vh[("b", BETA)]
        ucf = vcd_score(D["z_b"], np.log(np.clip(D["pcf_b"], EPS, None)), a)
        add(Arm(f"VCD_cf_{a}", role="control (OvO = base exactly)", pred=decide(ucf, mask, empty, clean["b"]),
                scores=ucf, boot=("primary",)))
    X1 = np.log(np.clip(D["P_b"], EPS, None))
    X2 = np.hstack([X1, D["lp_b500"]])
    add(Arm("scaler_single", role="report-only (change 2)", probs=scaler_oof(X1, y, k, B2_C[D["flag"]], splits),
            boot=("ovo",)))
    add(Arm("scaler_joint", role="report-only (change 2)", probs=scaler_oof(X2, y, k, B2_C[D["flag"]], splits),
            boot=("ovo",)))
    return arms


def contrast_specs() -> list[tuple]:
    """(metric, arm a, arm b[, arm c, arm d]): a - b, or (a - b) - (c - d)."""
    C = []
    for a in ALPHAS:
        v = f"VCD_{a}"
        C += [("ovo", v, "base"), ("ovr", v, "base"), ("primary", v, "base"), ("primary", f"{v}_cal", "base_cal"),
              ("primary", v, f"VCD_mean_{a}"), ("primary", f"{v}@beta0.2", "base"), ("ovo", f"{v}_lex", "base"),
              ("ovo", f"VCD+R_clean_{a}", v), ("ovo", f"VCD+R_both_{a}", v),
              ("ovo", f"VCD+R_clean_{a}", v, "R", "base"), ("ovo", f"VCD+R_both_{a}", v, "R", "base"),
              ("primary", f"VCD+R_clean_{a}_cal", f"{v}_cal"),
              ("ovo", f"VCD999_{a}", "base"), ("primary", f"VCD999_{a}", "base")]
    C += [("primary", f"VCD_cf_{a}", "base") for a in CF_ALPHAS]
    C += [("ovo", "R", "base"), ("primary", "R", "base"), ("primary", "R_cal", "base_cal"),
          ("primary", "base_cal", "base"), ("primary", "R_text", "base_text"), ("ovo", "scaler_joint", "scaler_single")]
    return C


def flips(pa, pb, y) -> dict:
    moved = pa != pb
    trans = {}
    for u, v in zip(pb[moved], pa[moved]):
        key = f"{L[u]}->{L[v]}"
        trans[key] = trans.get(key, 0) + 1
    return {"n": int(moved.sum()), "corrected": int(((pa == y) & (pb != y)).sum()),
            "broken": int(((pb == y) & (pa != y)).sum()), "transitions": dict(sorted(trans.items(), key=lambda t: -t[1]))}


def pair_stats(z, zp, k) -> dict:
    """Label-free, over all analysis images: d' = a*d + b + r per class pair."""
    out = {}
    for c in range(k):
        for j in range(c + 1, k):
            d, dp = z[:, c] - z[:, j], zp[:, c] - zp[:, j]
            vd = d.var()
            a = float(np.cov(d, dp, bias=True)[0, 1] / vd) if vd > 0 else float("nan")
            b = float(dp.mean() - a * d.mean()) if vd > 0 else float("nan")
            r = dp - a * d - b if vd > 0 else dp - dp.mean()
            out[f"{L[c]}{L[j]}"] = {"slope_a": a, "intercept_b": b,
                                    "corr": float(np.corrcoef(d, dp)[0, 1]) if vd > 0 and dp.var() > 0 else float("nan"),
                                    "resid_sd": float(r.std()), "var_ratio": float(dp.var() / vd) if vd > 0 else float("nan")}
    return out


def noised_view(D, name) -> dict:
    vals, counts = np.unique(D[f"nnew_{name}"], return_counts=True)
    return {"compliance": float(D[f"ftio_{name}"].mean()), "leakage_mean": float(D[f"leakn_{name}"].mean()),
            "leakage_max": float(D[f"leakn_{name}"].max()),
            "new_tokens_hist": {str(int(v)): int(c) for v, c in zip(vals, counts)},
            "text_pred_counts": np.bincount(D[f"textn_{name}"], minlength=D["k"]).tolist(),
            "argmax_counts": np.bincount(D[f"Pn_{name}"].argmax(1), minlength=D["k"]).tolist()}


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
            "p99": float(np.quantile(x, 0.99)), "max": float(x.max())}


# --------------------------------------------------------------------------- analysis
def analyse_dataset(D: dict, B: int, jobs: int) -> dict:
    k, ordinal, y = D["k"], D["ordinal"], D["y"]
    arms = build_arms(D)
    point = {n: {m: metric_value(m, y, a.pred, a.scores, a.probs, k, ordinal,
                                 softmax(a.scores) if a.probs is None and a.scores is not None else None)
                 for m in a.metric_names()} for n, a in arms.items()}
    idx = boot_idx(len(y), B=B, seed=0)    # ONE resample set for every arm of this dataset
    names = [n for n, a in arms.items() if a.boot]
    res = Parallel(n_jobs=jobs)(delayed(boot_arm)(arms[n].pred, arms[n].scores, arms[n].probs, arms[n].boot, y, idx,
                                                  k, ordinal) for n in names)
    boot = dict(zip(names, res))

    contrasts = {}
    for spec in contrast_specs():
        m, a, b = spec[:3]
        d = boot[a][m] - boot[b][m]
        pt = point[a][m] - point[b][m]
        label = f"{m}: {a} - {b}"
        involved = [a, b]
        if len(spec) == 5:
            c, e = spec[3:]
            d = d - (boot[c][m] - boot[e][m])
            pt = pt - (point[c][m] - point[e][m])
            label = f"{m}: ({a} - {b}) - ({c} - {e})"
            involved += [c, e]
        lo, hi = ci(d)
        # the card defines synergy and the interaction on the clean branch only
        added = any(arms[x].origin == ADDED or x.startswith("VCD+R_both") for x in involved)
        contrasts[label] = {"point": float(pt), "lo": lo, "hi": hi, "p": boot_p(d), "origin": ADDED if added else CARD}

    def get(m, a, b):
        return contrasts[f"{m}: {a} - {b}"]

    rules = {"screen": {}, "primary_claim_not_below_0": {}, "synergy": {}, "half_width": {}}
    for a in DECIDING_ALPHAS:
        c = get("ovo", f"VCD_{a}", "base")
        rules["screen"][a] = {"pass": c["lo"] > 0, "p": c["p"]}
        rules["primary_claim_not_below_0"][a] = not get("primary", f"VCD_{a}_cal", "base_cal")["hi"] < 0
        rules["synergy"][a] = get("ovo", f"VCD+R_clean_{a}", f"VCD_{a}")["lo"] > 0
    for a in ALPHAS:
        c = get("ovo", f"VCD_{a}", "base")
        rules["half_width"][a] = (c["hi"] - c["lo"]) / 2
    rules["screen_pass"] = any(v["pass"] for v in rules["screen"].values())
    rules["harm_alpha1"] = get("ovo", "VCD_1.0", "base")["hi"] < 0
    rules["mde80_R_minus_base"] = MDE80_TUNE[D["flag"]]
    rules["joint_exceeds_single_view (change 2)"] = get("ovo", "scaler_joint", "scaler_single")["lo"] > 0

    vstats = {}
    for br, lab in (("b", "base branch"), ("R", "R branch")):
        for beta in (BETA, BETA_SENS):
            mask, flagged, empty = v_head(D[f"P_{br}"], D[f"leak_{br}"], beta)
            vstats[f"{lab}, beta {beta}"] = {"flagged": int(flagged.sum()), "empty": int(empty.sum()),
                                             "share_size1": float((mask.sum(1) == 1).mean()),
                                             "mean_size": float(mask.sum(1).mean())}
    flip = {f"VCD_{a} vs base": flips(arms[f"VCD_{a}"].pred, arms["base"].pred, y) for a in DECIDING_ALPHAS}
    flip.update({f"VCD_{a}_cal vs base_cal": flips(arms[f"VCD_{a}_cal"].pred, arms["base_cal"].pred, y)
                 for a in DECIDING_ALPHAS})

    free = {"base @ 500": {"pairs": pair_stats(D["z_b"], D["zp_b500"], k),
                           "js_clean_vs_noised": float(js(D["p_b"], softmax(D["zp_b500"])).mean()),
                           "js_clean_vs_p_cf (B1 reliance)": float(js(D["p_b"], D["pcf_b"][None, :]).mean()),
                           "noised_view": noised_view(D, "b500")},
            "base @ 999": {"pairs": pair_stats(D["z_b"], D["zp_b999"], k),
                           "js_clean_vs_noised": float(js(D["p_b"], softmax(D["zp_b999"])).mean()),
                           "noised_view": noised_view(D, "b999")},
            "R @ 500": {"pairs": pair_stats(D["z_R"], D["zp_R500"], k),
                        "js_clean_vs_noised": float(js(D["p_R"], softmax(D["zp_R500"])).mean()),
                        "js_clean_vs_p_cf (B1 reliance)": float(js(D["p_R"], D["pcf_R"][None, :]).mean()),
                        "noised_view": noised_view(D, "R500")}}
    rp = D["rep_pos"]
    seed_rep = {}
    for c in range(k):
        for j in range(c + 1, k):
            d1 = D["zp_b500"][rp, c] - D["zp_b500"][rp, j]
            d2 = D["zp_rep2"][:, c] - D["zp_rep2"][:, j]
            seed_rep[f"{L[c]}{L[j]}"] = float(np.corrcoef(d1, d2)[0, 1]) if d1.var() > 0 and d2.var() > 0 else None
    seed_rep = {"n": int(len(rp)), "corr_dprime_rep1_rep2": seed_rep}

    cp = D["c_pos"]
    lc = np.log(np.clip(D["pid_clean"], EPS, None))
    ln = D["lpid_b500"][cp]
    change1 = {"n": int(len(cp)), "rows": "the clean base re-run's rows in the analysis set (base only: per-token "
                                          "VCD + R is not defined; the R mismatch rows are reported as splits)",
               "by_alpha": {}}
    for a in ALPHAS:
        ad = a * renyi(split_logw(lc), split_logw(ln), 1.0 + a)
        change1["by_alpha"][a] = {
            "alpha_D": quantiles(ad),
            "ovo_per_token": ovo_scores_auc(y[cp], per_token_score(lc, ln, a), k),
            "ovo_option_level": ovo_scores_auc(y[cp], vcd_score(D["z_b"][cp], D["zp_b500"][cp], a), k)}

    log_eps = np.log(EPS)
    clips = {"clean p_option (base) < EPS": int((D["p_b"] < EPS).sum()),
             "clean p_option (R) < EPS": int((D["p_R"] < EPS).sum()),
             "clean p_ids (re-run) < EPS": int((D["pid_clean"] < EPS).sum()),
             **{f"noised z' ({n}) < log EPS (unclipped)": int((D[f"zp_{n}"] < log_eps).sum())
                for n in ("b500", "R500", "b999")},
             "noised logp_ids (b500) < log EPS (unclipped)": int((D["lpid_b500"] < log_eps).sum()),
             **{f"_cal fit input softmax(u) < EPS ({n})": int((softmax(a.scores) < EPS).sum())
                for n, a in arms.items() if f"{n}_cal" in arms}}
    table = {n: {"origin": a.origin, "role": a.role, **point[n],
                 **({"pred_counts": np.bincount(a.pred, minlength=k).tolist()} if a.pred is not None else {})}
             for n, a in arms.items()}
    return {"n": int(len(y)), "arms": table, "contrasts": contrasts, "rules": rules, "v_head": vstats,
            "flips": flip, "label_free": free, "seed_repeat": seed_rep, "change1": change1,
            "change2": {"single_tune_cv_ovo": point["scaler_single"]["ovo"],
                        "joint_tune_cv_ovo": point["scaler_joint"]["ovo"], "B2_AUC_opt (sanity only)": D["b2_auc_opt"]},
            "mismatch_splits": D["mismatch"], "clip_counts": clips,
            "ovo_identities": {"base_cal (CV-pooled, fold-wise offsets) - base":
                               point["base_cal"]["ovo"] - point["base"]["ovo"],
                               **{f"VCD_cf_{a} - base": point[f"VCD_cf_{a}"]["ovo"] - point["base"]["ovo"]
                                  for a in CF_ALPHAS},
                               **{f"VCD_mean_{a} - base": point[f"VCD_mean_{a}"]["ovo"] - point["base"]["ovo"]
                                  for a in ALPHAS}}}


def fmt(c: dict, dec: int = 4) -> str:
    mark = "*" if c["lo"] > 0 or c["hi"] < 0 else ""
    return f"{c['point']:+.{dec}f} [{c['lo']:+.{dec}f}, {c['hi']:+.{dec}f}]{mark}"


def report_lines(res: dict, header: list[str]) -> list[str]:
    out = header[:]
    for flag, r in res["datasets"].items():
        out += [f"## {flag} (n = {r['n']})", "", "| arm | origin | role | primary | OvO | OvR | predicted counts |",
                "|---|---|---|---|---|---|---|"]
        for n, a in r["arms"].items():
            cell = lambda m: f"{a[m]:.4f}" if m in a else "—"  # noqa: E731
            out.append(f"| {n} | {a['origin']} | {a['role']} | {cell('primary')} | {cell('ovo')} | {cell('ovr')} | "
                       f"{a.get('pred_counts', '—')} |")
        out += ["", "| contrast | Δ [95% CI] | bootstrap p | origin |", "|---|---|---|---|"]
        out += [f"| {c} | {fmt(v)} | {v['p']:.4f} | {v['origin']} |" for c, v in r["contrasts"].items()]
        ru = r["rules"]
        out += ["", f"- Screen (ΔOvO VCD − base, CI > 0): " + "; ".join(
                    f"α {a}: {'PASS' if v['pass'] else 'no'} (p {v['p']:.4f}, Holm {res['holm'][flag][str(a)]:.4f})"
                    for a, v in ru["screen"].items()) + f" → screen pass: **{ru['screen_pass']}**.",
                f"- Harm at α 1 (CI < 0): {ru['harm_alpha1']}. Primary claim allowed (VCD_cal − base_cal CI not "
                f"entirely below 0): {ru['primary_claim_not_below_0']}. Synergy ((VCD + R)_clean − VCD CI > 0): "
                f"{ru['synergy']}.",
                f"- CI half-width of ΔOvO(VCD − base) by α: {ru['half_width']} (reference MDE80 of ΔOvO(R − base): "
                f"{ru['mde80_R_minus_base']}).",
                f"- OvO identities (must be 0 except CV-pooled base_cal): {r['ovo_identities']}.",
                f"- V_head: {r['v_head']}.", f"- Flips: {r['flips']}.",
                f"- Label-free distortion statistics: {json.dumps(r['label_free'])}.",
                f"- Seed repeat: {r['seed_repeat']}.", f"- Change 1 (per-token VCD, report-only): {r['change1']}.",
                f"- Change 2 (tune-CV scalers, report-only): {r['change2']}.",
                f"- Mismatch rows (V1-1(2)) \" X\" / \"X\" split: {r['mismatch_splits']}.",
                f"- Clip counts: {r['clip_counts']}.", ""]
    return out


def analyse(data: dict, out_dir: Path, header_info: dict, B: int = 10_000, jobs: int = 6) -> dict:
    """Step 1 (abort before writing anything if an alpha = 0 identity fails), then every arm, rule and report."""
    bad = alpha0_failures(data)
    if bad:
        raise SystemExit("step 1 failed: an alpha = 0 identity does not hold; nothing written.\n  " + "\n  ".join(bad))
    res = {"header": header_info, "datasets": {f: analyse_dataset(D, B, jobs) for f, D in data.items()}}
    tests = [(f, a, res["datasets"][f]["rules"]["screen"][a]["p"]) for f in data for a in DECIDING_ALPHAS]
    adj = holm([t[2] for t in tests])
    res["holm"] = {f: {} for f in data}
    for (f, a, _), q in zip(tests, adj):
        res["holm"][f][str(a)] = q
    res["any_screen_pass"] = [f for f in data if res["datasets"][f]["rules"]["screen_pass"]]
    header = ["# Phase 7 · card V1 · VCD at the first-token readout (tune; G5)", ""] + \
             [f"- {k}: {v}" for k, v in header_info.items()] + \
             ["- Paired bootstrap 10,000 (seed 0), one resample set per dataset shared by all arms; `*` = CI excludes "
              "0. Arms and contrasts marked “added at G2, report-only” are not in the card.",
              "- base_cal / R_cal / *_cal are 5-fold CV-pooled, with each fold's own offsets. The card's [der] "
              "identity OvO(base_cal) = OvO(base) holds exactly for one offset vector (so within a fold). Across "
              "pooled folds the offsets differ, so pairs that straddle folds can shift. The pooled difference is "
              "printed under “OvO identities”; the deciding contrast is ΔOvO(VCD − base), as registered.",
              f"- Datasets passing the screen: {res['any_screen_pass'] or 'none'}.", ""]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vcd.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (out_dir / "vcd_report.md").write_text("\n".join(report_lines(res, header)) + "\n", encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--token-note", default=None, help="the dated V1- note, required if max_new_tokens != 8")
    args = ap.parse_args(argv)
    if args.check_inputs:
        return check_inputs()
    code_commit = git("rev-parse", code_commit_from_index(PLAN_INDEX.read_text(encoding="utf-8")) + "^{commit}")
    subsets = tune_subsets()
    ready = check_ready(RESULTS, code_commit, code_trees_at(code_commit), subsets)
    if ready["max_new_tokens"] != 8 and not args.token_note:
        raise SystemExit(f"G4 noised runs used max_new_tokens {ready['max_new_tokens']}: pass --token-note citing "
                         "the dated V1- note")
    data = {f: load_dataset(RESULTS, f, subsets, ready["layout"]) for f in FLAGS}
    diff = git("diff", "--stat", code_commit, "HEAD", "--", "effvar", "scripts", "tests")
    header = {"code commit (plan index)": code_commit, "analysis HEAD": git("rev-parse", "HEAD"),
              "effvar/ scripts/ tests/ changed since the code commit": diff or "none",
              "G4 noised max_new_tokens": ready["max_new_tokens"] if ready["max_new_tokens"] == 8
              else f"{ready['max_new_tokens']} ({args.token_note})",
              "clean re-run layout (V1-1(3))": ready["layout"],
              "clean side": "log(clip(p_option renormalised, 1e-12)), as Phase 6",
              "noised side": "stored fp32 logp_option renormalised in log space, unclipped"}
    analyse(data, RESULTS / OUT_REL, header, jobs=args.jobs)
    print((RESULTS / OUT_REL / "vcd_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
