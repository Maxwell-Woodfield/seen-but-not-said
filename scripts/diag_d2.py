"""D2 — per-class yes/no prompts: analysis (notes/diagnosis_plan.md §5 D2; notes/diagnosis_amendments.md). CPU only.

    python scripts/diag_d2.py --code-commit <the D2 code commit>

Run ONLY after the last D2 tune run has ended (Amendment 4d). A fix made after that goes in a new, dated commit and
is disclosed: the report prints `git diff --stat <code commit> HEAD -- effvar scripts tests`.
Arms (tune, n = 300 per dataset; Amendment 1 exclusions as in A1, so Pneumonia n = 298):
- base_letter, R_letter: the saved frozen-order letter runs (results/tune/base, results/tune/var/rho0.5_summ0.1);
- base_yn, R_yn: results/tune/diag/D2/{base_yn,R_yn} (scripts/run_d2.py), score s_c per class (effvar/yesno.py).
Every output is tagged with the clause and the line numbers of notes/diagnosis_amendments.md that it implements
(the file is append-only, so the lines are stable). Nothing here gates anything beyond Amendment 3(d)'s two rules,
and D2 routes nothing to E (Amendment 3e). Writes results/diag/D2/{d2.json, d2_report.md}.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_d2_regression import SMOKE_RUNS, load_rows, smoke_repro  # noqa: E402
from diag_c4 import cv_calibrate  # noqa: E402  (5-fold stratified CV, A1's folds, seed 0)

from effvar import FLAGS, ROOT  # noqa: E402
from effvar.diag_stats import (DATASETS, EPS, align, binary_auc, boot_idx, ci, fmt, load_run,  # noqa: E402
                               ovo_logratio_auc, primary, ranking_auc)
from effvar.prompt_variants import yesno_specs  # noqa: E402
from effvar.yesno import COMPLIANCE_MIN  # noqa: E402

OUT = ROOT / "results" / "diag" / "D2"
LETTER = {"base_letter": "results/tune/base", "R_letter": "results/tune/var/rho0.5_summ0.1"}
YN = {"base_yn": "results/tune/diag/D2/base_yn", "R_yn": "results/tune/diag/D2/R_yn"}
ARMS = ("base_letter", "R_letter", "base_yn", "R_yn")
COLS = ("ovo", "ovr", "primary", "primary_cal")
CODE_DIRS = ("effvar", "scripts", "tests")
ABSENCE = {"retinamnist": 0, "pneumoniamnist": 0}   # absence-type classes named in Amendment 4(b)
CITE = {  # notes/diagnosis_amendments.md
    "prompt": "A3(b) L70-79", "deciding": "A3(c) L80-81", "beside": "A3(c) L81-83", "exclusions": "A3(c) L84",
    "guard": "A3(c) L84-86", "rules": "A3(d) L87-91", "other": "A3(d) L92", "diagnostic": "A3(e) L93-95",
    "comparator": "A2(e) L55-57", "expectation": "A4(b) L115-117", "report_only": "A4(c) L118-120",
    "clean_tree": "A4(d) L121-125", "kernel": "A4(e) L126-128", "smoke": "A4(f) L129-133",
}


# --------------------------------------------------------------------------- data
def load_yn(run_dir, flag) -> dict[int, dict]:
    """index -> {label, pred, p (= s, NOT renormalised), p_yes, leakage, compliant} from a run_d2.py run."""
    out = {}
    with open(Path(run_dir) / flag / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[int(r["index"])] = {"label": int(r["label"]), "pred": int(r["pred"]), "p": np.asarray(r["s"], float),
                                    "p_yes": np.asarray(r["p_yes"], float),
                                    "leakage": np.asarray(r["leakage"], float),
                                    "compliant": np.asarray(r["compliant"], bool)}
    return out


def align_arms(runs: dict[str, dict[int, dict]], exclude=frozenset()):
    """Common image order across all arms, by image index (never by file order). Every arm must hold the same
    images and the same label for each; `align` asserts the labels."""
    sets = {a: set(r) for a, r in runs.items()}
    first = next(iter(sets.values()))
    if any(s != first for s in sets.values()):
        raise ValueError(f"arms hold different images: { {a: len(s) for a, s in sets.items()} }")
    idx, y, arr = align(runs, exclude=exclude)
    for a, r in runs.items():
        for key in ("p_yes", "leakage", "compliant"):
            if key in r[idx[0]]:
                arr[a][key] = np.stack([r[i][key] for i in idx])
    return idx, y, arr


# --------------------------------------------------------------------------- metrics
def guard(compliant, leakage) -> dict:
    """Amendment 3(c): >= 0.99 of the K x n prompts compliant, else the arm is broken and not interpreted."""
    rate = float(np.mean(compliant))
    return {"compliance": rate, "leakage_mean": float(np.mean(leakage)), "broken": bool(rate < COMPLIANCE_MIN)}


def make_arm(score, pred, y, k) -> dict:
    """score: letter p (renormalised) or yes/no s (unnormalised). OvO reads log score (a per-image constant cancels);
    OvR reads the score renormalised over classes; primary reads `pred` raw and after NLL offsets on log score,
    fitted by 5-fold CV on tune (A1's folds)."""
    return {"score": score, "pred": pred, "pred_cal": cv_calibrate(score, y, k).argmax(1),
            "p_ovr": score / score.sum(1, keepdims=True)}


def metric_row(y, arm, k, ordinal, ix):
    yy = y[ix]
    return (ovo_logratio_auc(yy, arm["score"][ix], k), ranking_auc(yy, arm["p_ovr"][ix], k),
            primary(yy, arm["pred"][ix], k, ordinal), primary(yy, arm["pred_cal"][ix], k, ordinal))


def boot_arm(y, arm, k, ordinal, idx):
    return np.array([metric_row(y, arm, k, ordinal, ix) for ix in idx])


def rule_tests(point: dict, boot: dict, broken: dict | None = None) -> dict:
    """Amendment 3(d), on OvO (column 0), one resample set shared by all four arms:
    readout supported iff dOvO(base_yn - base_letter) CI > 0;
    (c) supported iff [dOvO(R_yn - base_yn) - dOvO(R_letter - base_letter)] CI > 0.
    A rule that reads a broken arm is not interpreted (Amendment 3c): `supported` is None."""
    broken = broken or {}
    j = COLS.index("ovo")
    terms = {"readout": ((point["base_yn"][j] - point["base_letter"][j]),
                         boot["base_yn"][:, j] - boot["base_letter"][:, j], ("base_yn",)),
             "c": ((point["R_yn"][j] - point["base_yn"][j]) - (point["R_letter"][j] - point["base_letter"][j]),
                   (boot["R_yn"][:, j] - boot["base_yn"][:, j]) - (boot["R_letter"][:, j] - boot["base_letter"][:, j]),
                   ("base_yn", "R_yn"))}
    out = {}
    for name, (pt, dist, reads) in terms.items():
        lo, hi = ci(dist)
        interpreted = not any(broken.get(a, False) for a in reads)
        out[name] = {"point": float(pt), "lo": lo, "hi": hi, "interpreted": interpreted,
                     "supported": bool(lo > 0) if interpreted else None}
    return out


def per_class(y, arr, k) -> list[dict]:
    """Amendment 4(c), report-only: per-class OvR AUC of s_c beside the letter p_c, and per-class mean p(yes)."""
    rows = []
    for c in range(k):
        pos = y == c
        rows.append({"class": c,
                     **{f"auc_{a}": binary_auc(pos, arr[a]["p"][:, c]) for a in ARMS},
                     **{f"mean_p_yes_{a}": float(arr[a]["p_yes"][:, c].mean()) for a in ("base_yn", "R_yn")}})
    return rows


# --------------------------------------------------------------------------- provenance
def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def need(cond, msg):
    if not cond:
        raise SystemExit(f"provenance check failed: {msg}")


def check_provenance(flag, code_trees) -> dict:
    cfg = {a: yaml.safe_load((ROOT / d / flag / "config.yaml").read_text(encoding="utf-8"))
           for a, d in {**LETTER, **YN}.items()}
    prompts = [s.text() for s in yesno_specs(flag)]
    for a in YN:
        c = cfg[a]
        need(c.get("prompt_variant") == "yesno", f"{flag} {a}: prompt_variant")
        need(c["prompts"] == prompts, f"{flag} {a}: prompts differ from yesno_specs")
        need(c["subset"]["name"] == "tune" and c["subset"]["n"] == 300 and c["limit"] is None, f"{flag} {a}: subset")
        need(c["attn"] == "var_eager" == cfg["base_letter"]["attn"] == cfg["R_letter"]["attn"],
             f"{flag} {a}: attn ({CITE['kernel']})")
        need(c["vision_cache"] is True, f"{flag} {a}: vision cache ({CITE['prompt']})")
        for when in ("git", "git_end"):
            g = c[when]
            need(not g["dirty"] and not g["code_changes"], f"{flag} {a} {when}: tree not clean ({CITE['clean_tree']})")
            need(g["code_trees"] == code_trees, f"{flag} {a} {when}: code differs from the code commit "
                                                f"({CITE['clean_tree']})")
    need(cfg["base_yn"]["method"] == "base", f"{flag}: base_yn method")
    need(cfg["R_yn"]["method"] == cfg["R_letter"]["method"] and
         cfg["R_yn"]["method_params"] == cfg["R_letter"]["method_params"], f"{flag}: R_yn is not R")
    return {a: cfg[a]["git"]["commit"] for a in YN}


# --------------------------------------------------------------------------- analysis
def analyse(code_commit: str, B: int = 10_000) -> None:
    code_trees = {d: git("rev-parse", f"{code_commit}:{d}") for d in CODE_DIRS}
    diff = git("diff", "--stat", code_commit, "HEAD", "--", *CODE_DIRS)
    a1 = json.loads((ROOT / "results/diag/A1/a1.json").read_text(encoding="utf-8"))
    report = {"code_commit": code_commit, "analysis_head": git("rev-parse", "HEAD"), "code_diff_since": diff,
              "cite": CITE, "datasets": {}}
    L = ["# D2 — per-class yes/no prompts (P3; tune; diagnostic only)", "",
         f"Code commit `{code_commit[:7]}`; analysis at `{report['analysis_head'][:7]}`. "
         f"Changes to effvar/ scripts/ tests/ since the code commit: {'none' if not diff else chr(10) + diff}.",
         f"Tags cite notes/diagnosis_amendments.md. D2 routes nothing to E; Phase 6 ends S-null [{CITE['diagnostic']}].",
         f"Rule tests are unadjusted, tune-only and diagnostic [{CITE['report_only']}]. `*` = CI excludes 0. "
         f"Paired bootstrap {B:,} resamples, seed 0, one set shared by all four arms [{CITE['rules']}].", ""]

    smoke_lines = [f"Tune smoke reproduced by the full runs [{CITE['smoke']}] (logits_sha1 per prompt):"]
    smoke_ok = True
    for arm, (full_run, smoke_run) in SMOKE_RUNS.items():
        for flag in FLAGS:
            r = smoke_repro(load_rows(full_run, flag), load_rows(smoke_run, flag))
            smoke_ok &= r["ok"]
            smoke_lines.append(f"- {arm} {flag}: images {r['images_identical']}/{r['images']}, prompts "
                               f"{r['prompts_identical']}/{r['prompts']}" + ("" if r["ok"] else " **DIFFER**"))
    report["smoke_repro_ok"] = smoke_ok
    L += smoke_lines + [""]

    summary = ["| dataset | readout: dOvO(base_yn − base_letter) | supported | (c): interaction | supported |",
               "|---|---|---|---|---|"]
    body = []
    for flag in FLAGS:
        k, ordinal = DATASETS[flag]
        commits = check_provenance(flag, code_trees)
        excl = set(json.loads((ROOT / f"data/subsets/{flag}_tune_overlap.json").read_text())["exclude_tune_indices"])
        runs = {**{a: load_run(ROOT / d, flag) for a, d in LETTER.items()},
                **{a: load_yn(ROOT / d, flag) for a, d in YN.items()}}
        keep, y, arr = align_arms(runs, exclude=excl)
        n = len(y)
        g = {a: guard(arr[a]["compliant"], arr[a]["leakage"]) for a in YN}
        broken = {a: g[a]["broken"] for a in YN}
        arms = {a: make_arm(arr[a]["p"], arr[a]["pred"], y, k) for a in ARMS}
        a1_bc = a1[flag]["tune 5-fold CV"]["arms"]["base_cal"]["primary"]
        need(abs(primary(y, arms["base_letter"]["pred_cal"], k, ordinal) - a1_bc) < 1e-12,
             f"{flag}: base_cal differs from A1's tune-CV base_cal ({CITE['beside']})")

        idx = boot_idx(n, B=B)
        point = {a: np.array(metric_row(y, arms[a], k, ordinal, np.arange(n))) for a in ARMS}
        boot = dict(zip(ARMS, Parallel(n_jobs=len(ARMS))(delayed(boot_arm)(y, arms[a], k, ordinal, idx)
                                                         for a in ARMS)))
        rules = rule_tests(point, boot, broken)

        def d(a, b, ja, jb=None):
            jb = ja if jb is None else jb
            return [float(point[a][ja] - point[b][jb]), *ci(boot[a][:, ja] - boot[b][:, jb])]

        jo, jr, jp, jc = (COLS.index(c) for c in COLS)
        contrasts = {  # OvO decides [A3(c)]; OvR beside it; primary vs base_cal = base_letter's calibrated column
            "base_yn − base_letter": {"ovo": d("base_yn", "base_letter", jo), "ovr": d("base_yn", "base_letter", jr)},
            "R_yn − base_yn": {"ovo": d("R_yn", "base_yn", jo), "ovr": d("R_yn", "base_yn", jr)},
            "R_letter − base_letter": {"ovo": d("R_letter", "base_letter", jo), "ovr": d("R_letter", "base_letter", jr)},
            "R_yn − R_letter": {"ovo": d("R_yn", "R_letter", jo), "ovr": d("R_yn", "R_letter", jr)},
        }
        vs_cal = {f"{a} − base_cal": d(a, "base_letter", jp, jc) for a in ("base_yn", "R_yn")}
        vs_cal.update({f"{a}_cal − base_cal": d(a, "base_letter", jc, jc) for a in ("base_yn", "R_yn")})
        s_all = np.concatenate([arr[a]["p"].ravel() for a in YN])
        numerics = {a: {"s_below_eps": int((arr[a]["p"] < EPS).sum()), "s_equal_1": int((arr[a]["p"] >= 1.0).sum()),
                        "prompts": int(arr[a]["p"].size)} for a in YN}
        pc = per_class(y, arr, k)
        other = {"readout: base_yn OvO > base_letter OvO (point)": bool(point["base_yn"][jo] > point["base_letter"][jo]),
                 "(c): R's dOvO larger under yes/no than under letters (point)": bool(rules["c"]["point"] > 0)}

        report["datasets"][flag] = {
            "n": n, "excluded": sorted(excl), "run_commits": commits, "guards": g, "s_numerics": numerics,
            "arms": {a: dict(zip(COLS, map(float, point[a]))) for a in ARMS}, "contrasts": contrasts,
            "primary_vs_base_cal": vs_cal, "rules": rules, "other_predictions_point": other, "per_class": pc}
        need(np.isfinite(s_all).all(), f"{flag}: non-finite s")

        def verdict(r):
            return "not interpreted (broken arm)" if r["supported"] is None else ("YES" if r["supported"] else "no")
        summary.append(f"| {flag} | {fmt(rules['readout']['point'], rules['readout']['lo'], rules['readout']['hi'])} | "
                       f"{verdict(rules['readout'])} | {fmt(rules['c']['point'], rules['c']['lo'], rules['c']['hi'])} | "
                       f"{verdict(rules['c'])} |")

        names = yesno_specs(flag)[0].option_texts
        body += [f"## {flag} (n = {n}; excluded {sorted(excl) or 'none'} [{CITE['exclusions']}]; runs at "
                 f"{', '.join(f'{a} `{c[:7]}`' for a, c in commits.items())})", "",
                 f"Guards [{CITE['guard']}]: " + "; ".join(
                     f"{a} compliance {g[a]['compliance']:.4f}, leakage {g[a]['leakage_mean']:.4f}"
                     f"{' **BROKEN — not interpreted**' if g[a]['broken'] else ' ok'}" for a in YN) + ".",
                 f"s numerics [{CITE['deciding']}]: " + "; ".join(
                     f"{a} {v['s_below_eps']} of {v['prompts']} below 1e-12 (clipped by the OvO), {v['s_equal_1']} "
                     f"equal to 1" for a, v in numerics.items()) + ".", "",
                 f"| arm | OvO (deciding) [{CITE['deciding']}] | OvR [{CITE['beside']}] | primary | primary cal "
                 f"[{CITE['beside']}] |", "|---|---|---|---|---|"]
        body += [f"| {a} | " + " | ".join(f"{x:.4f}" for x in point[a]) + " |" for a in ARMS]
        body += ["", f"| contrast | ΔOvO (deciding) | ΔOvR |", "|---|---|---|"]
        body += [f"| {c} | {fmt(*v['ovo'])} | {fmt(*v['ovr'])} |" for c, v in contrasts.items()]
        body += ["", f"Primary vs base_cal (A1's tune-CV base_cal, {a1_bc:.4f}) [{CITE['beside']}; {CITE['comparator']}]: "
                 + "; ".join(f"{c} {fmt(*v)}" for c, v in vs_cal.items()) + ".", "",
                 f"Rules [{CITE['rules']}; unadjusted, tune-only, diagnostic {CITE['report_only']}]: "
                 f"**readout** {fmt(rules['readout']['point'], rules['readout']['lo'], rules['readout']['hi'])} → "
                 f"{verdict(rules['readout'])}; **(c)** {fmt(rules['c']['point'], rules['c']['lo'], rules['c']['hi'])} → "
                 f"{verdict(rules['c'])}.",
                 f"Other D2 predictions, point estimates, report-only [{CITE['other']}]: "
                 + "; ".join(f"{q}: {'yes' if v else 'no'}" for q, v in other.items()) + ".", "",
                 f"Per class, report-only [{CITE['report_only']}; † = absence-type class, {CITE['expectation']}]:", "",
                 "| class | OvR AUC s_c base_yn | R_yn | OvR AUC letter p_c base | R | mean p(yes) base | R |",
                 "|---|---|---|---|---|---|---|"]
        for r in pc:
            mark = " †" if ABSENCE.get(flag) == r["class"] else ""
            body.append(f"| {r['class']} {names[r['class']]}{mark} | {r['auc_base_yn']:.4f} | {r['auc_R_yn']:.4f} | "
                        f"{r['auc_base_letter']:.4f} | {r['auc_R_letter']:.4f} | {r['mean_p_yes_base_yn']:.4f} | "
                        f"{r['mean_p_yes_R_yn']:.4f} |")
        body.append("")

    L += [f"### The six rule tests [{CITE['rules']}; {CITE['report_only']}]", ""] + summary + [""] + body
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "d2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # "−", "†" on a cp1252 console
    print("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-commit", required=True, help="the D2 code commit the runs must have used")
    args = ap.parse_args()
    dirs = [ROOT / d for d in YN.values()] + [ROOT / "results" / s for _, s in SMOKE_RUNS.values()]
    missing = [(d / f).relative_to(ROOT).as_posix() for d in dirs for f in FLAGS
               if not (d / f / "predictions.jsonl").exists()]
    if missing:
        raise SystemExit("D2 runs missing (see scripts/run_d2.py for the commands): " + ", ".join(missing))
    analyse(git("rev-parse", args.code_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
