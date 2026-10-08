"""D2 regression checks (plan approved 2026-09-29; Amendment 4f). Reads results only, no model.

    python scripts/check_d2_regression.py check         # (A) (B) (C) + static diff -> results/diag/D2/regression/regression.txt
    python scripts/check_d2_regression.py smoke-repro   # after the full D2 tune runs -> results/diag/D2/smoke_repro.txt

The frozen default path (`scripts/run_eval.py`, --prompt frozen) must be unchanged by the D2 code:
(A) rehearsal: val first 20/dataset, R (`var_tau10`) and `base_sdpa`, logits_sha1 vs the COMMITTED
    results/rehearsal/{var_tau10,base_sdpa}/summary.csv (60/60 each);
(B) smoke: the post-change re-run (base text-eager, R) vs the committed pre-change reference
    results/diag/D2/regression/smoke_pre_sha1.json, recorded at a clean HEAD before any D2 code (90/90 each);
(C) cached path (the one D2 uses): run_eval --cache-vision on the first 20 tune images/dataset, base and R, both attn
    var_eager like the saved runs; p_option exactly equal (max abs diff 0.0) and new_ids equal vs the saved
    results/tune/base and results/tune/var/rho0.5_summ0.1 (60/60 each);
static: the default path's files are byte-identical to the reference HEAD.
smoke-repro (Amendment 4f): the full D2 tune runs reproduce the 10-image tune smoke bitwise, logits_sha1 per prompt.
Exit 0 only if everything matches.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAGS = ("retinamnist", "pneumoniamnist", "dermamnist")
REG = ROOT / "results" / "diag" / "D2" / "regression"
FROZEN_PATHS = ("effvar/prompts.py", "effvar/scoring.py", "effvar/evaluate.py", "effvar/generation.py",
                "effvar/model.py", "effvar/methods", "effvar/var.py", "effvar/vision_cache.py", "effvar/data.py",
                "effvar/runio.py", "scripts/run_eval.py", "scripts/check_gates.py")
A_RUNS = {"var_tau10": "diag/D2/regression/rehearsal/var_tau10", "base_sdpa": "diag/D2/regression/rehearsal/base_sdpa"}
B_RUNS = {"base_text_eager": "diag/D2/regression/smoke_post/base_text_eager", "R": "diag/D2/regression/smoke_post/R"}
C_RUNS = {"base": ("tune/diag/D2/regression/base", "tune/base"),
          "R": ("tune/diag/D2/regression/R", "tune/var/rho0.5_summ0.1")}
SMOKE_RUNS = {"base_yn": ("tune/diag/D2/base_yn", "tune/diag/D2/smoke/base_yn"),
              "R_yn": ("tune/diag/D2/R_yn", "tune/diag/D2/smoke/R_yn")}


def load_rows(run: str, flag: str) -> dict[int, dict]:
    with open(ROOT / "results" / run / flag / "predictions.jsonl", encoding="utf-8") as f:
        return {int(r["index"]): r for r in map(json.loads, f)}


def compare_sha1(new: dict[int, dict], ref: dict[int, str]) -> tuple[int, int, list[int]]:
    """(identical, reference size, differing indices). An image missing from `new` counts as differing."""
    bad = [i for i in sorted(ref) if i not in new or new[i]["logits_sha1"] != ref[i]]
    return len(ref) - len(bad), len(ref), bad


def compare_p_option(new: dict[int, dict], saved: dict[int, dict]) -> tuple[int, int, float, list[int]]:
    """(identical, n, max abs diff of p_option, differing indices) over the images of `new`; identical = p_option
    exactly equal and the same generated ids."""
    worst, bad = 0.0, []
    for i in sorted(new):
        a, b = new[i]["p_option"], saved[i]["p_option"]
        d = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("inf")
        worst = max(worst, d)
        if d != 0.0 or new[i]["new_ids"] != saved[i]["new_ids"]:
            bad.append(i)
    return len(new) - len(bad), len(new), worst, bad


def smoke_repro(full: dict[int, dict], smoke: dict[int, dict]) -> dict:
    """Amendment 4f: every smoke image is in the full run with the same logits_sha1 for every class prompt."""
    prompts = hit = 0
    bad = []
    for i in sorted(smoke):
        k = len(smoke[i]["logits_sha1"])
        prompts += k
        if i not in full:
            bad.append(i)
            continue
        same = sum(a == b for a, b in zip(full[i]["logits_sha1"], smoke[i]["logits_sha1"]))
        hit += same if len(full[i]["logits_sha1"]) == k else 0
        if same != k or len(full[i]["logits_sha1"]) != k:
            bad.append(i)
    return {"images": len(smoke), "images_identical": len(smoke) - len(bad), "prompts": prompts,
            "prompts_identical": hit, "differing_indices": bad, "ok": not bad and len(smoke) > 0}


def run_smoke_repro() -> tuple[bool, list[str]]:
    lines, ok = ["smoke-repro (Amendment 4f): full D2 tune runs vs the 10-image tune smoke, logits_sha1 per prompt"], True
    for arm, (full_run, smoke_run) in SMOKE_RUNS.items():
        for flag in FLAGS:
            r = smoke_repro(load_rows(full_run, flag), load_rows(smoke_run, flag))
            ok &= r["ok"]
            lines.append(f"  {arm} {flag}: images {r['images_identical']}/{r['images']}, prompts "
                         f"{r['prompts_identical']}/{r['prompts']}" + (f" DIFFER {r['differing_indices']}" if not r["ok"] else ""))
    lines.append("PASS" if ok else "FAIL: the full runs do not reproduce the smoke; report it")
    return ok, lines


def check() -> tuple[bool, list[str]]:
    ok, lines = True, ["D2 regression of the frozen default path (plan approved 2026-09-29)"]
    ref = json.loads((REG / "smoke_pre_sha1.json").read_text(encoding="utf-8"))

    lines.append("(A) rehearsal (val first 20/dataset) vs committed results/rehearsal/*/summary.csv logits_sha1")
    for arm, run in A_RUNS.items():
        with open(ROOT / "results" / "rehearsal" / arm / "summary.csv", encoding="utf-8") as f:
            committed = list(csv.DictReader(f))
        hit = n = 0
        for flag in FLAGS:
            h, m, bad = compare_sha1(load_rows(run, flag), {int(r["index"]): r["logits_sha1"]
                                                             for r in committed if r["dataset"] == flag})
            hit, n = hit + h, n + m
            if bad:
                lines.append(f"  {arm} {flag}: DIFFER {bad}")
        ok &= hit == n == 60
        lines.append(f"  {arm}: {hit}/{n}")

    lines.append(f"(B) smoke post-change vs the pre-change reference at {ref['head'][:7]} (smoke_pre_sha1.json)")
    for arm, run in B_RUNS.items():
        hit = n = 0
        for flag in FLAGS:
            pre = ref["runs"][arm][flag]
            new = load_rows(run, flag)
            h, m, bad = compare_sha1(new, {int(i): v["logits_sha1"] for i, v in pre.items()})
            bad += [int(i) for i, v in pre.items() if int(i) in new and new[int(i)]["new_ids"] != v["new_ids"]]
            h = m - len(set(bad))
            hit, n = hit + h, n + m
            if bad:
                lines.append(f"  {arm} {flag}: DIFFER {sorted(set(bad))}")
        ok &= hit == n == 90
        lines.append(f"  {arm}: {hit}/{n}")

    lines.append("(C) cached default path (first 20 tune/dataset) vs saved results/tune outputs: p_option exact + ids")
    for arm, (run, saved) in C_RUNS.items():
        hit = n = 0
        worst = 0.0
        for flag in FLAGS:
            h, m, w, bad = compare_p_option(load_rows(run, flag), load_rows(saved, flag))
            hit, n, worst = hit + h, n + m, max(worst, w)
            if bad:
                lines.append(f"  {arm} {flag}: DIFFER {bad}")
        ok &= hit == n == 60
        lines.append(f"  {arm}: {hit}/{n}, max abs diff p_option {worst}")

    diff = subprocess.run(["git", "diff", "--stat", ref["head"], "HEAD", "--", *FROZEN_PATHS], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    ok &= not diff
    lines.append(f"static: git diff {ref['head'][:7]} {head[:7]} -- <default-path files>: "
                 + ("empty" if not diff else f"NOT EMPTY\n{diff}"))
    lines.append("PASS" if ok else "FAIL: stop and report; no D2 smoke")
    return ok, lines


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "check":
        ok, lines = check()
        out = REG / "regression.txt"
    elif mode == "smoke-repro":
        ok, lines = run_smoke_repro()
        out = ROOT / "results" / "diag" / "D2" / "smoke_repro.txt"
    else:
        print(__doc__)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
