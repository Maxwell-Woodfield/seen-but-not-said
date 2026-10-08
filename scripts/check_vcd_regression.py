"""Card V1 guard checks (notes/vcd_plan.md "Guards", Amendment V1-1). Reads results only: no model, no GPU, no metric.

    python scripts/check_vcd_regression.py guard1         # G3: identity smoke reproduces the saved clean runs
    python scripts/check_vcd_regression.py guard3         # G3: the smoke re-run (2nd process) is bitwise, steps 500, 999
    python scripts/check_vcd_regression.py guard8-smoke   # G3: probe > 0, no noised cache, clean cache never written
    python scripts/check_vcd_regression.py guard7         # after G4: the clean per-id re-run reproduces the saved runs
    python scripts/check_vcd_regression.py guard8         # after G4: counters, first passes empty, tokens, + the smoke part

Every mode also checks guard 6 on every run it reads. Each run's recorded code trees, at the start and at the end,
must equal those of the code commit recorded in the plan index (the G2 row of notes/vcd_plan.md), with a clean tree.
Every mode prints `git diff --stat <code commit> HEAD -- effvar scripts tests` and any replaced-run records.
Writes results/vcd/V1/regression/<mode>.txt and <mode>.json. The JSON holds ok, the code commit, HEAD and the sha256
of every file read, so scripts/diag_vcd.py can refuse a stale pass. Exit 0 only on PASS.
"""
import json
import sys
from pathlib import Path

import yaml

from effvar import FLAGS, ROOT
from effvar.generation import MAX_NEW_TOKENS
from effvar.vcd import (MISMATCH_ROWS, PLAN_INDEX, SAVED_RUNS, V1_EXP, code_commit_from_index, code_trees_at, g4_specs,
                        git, selection, sha256_file, smoke_specs)

RESULTS = ROOT / "results"
REG_DIR = "vcd/V1/regression"          # under results/
SKIP = {"peak_vram_gib", "vision_ms", "prefill_ms", "lm_prefill_ms", "decode_ms_per_token", "lm_ms", "e2e_ms"}
IDENTITY = {"base_identity": "base", "R_identity": "var"}
RERUN_PAIRS = (("base_s500", "base_s500_rerun"), ("base_s999", "base_s999_rerun"))
GUARD3_FIELDS = ("p_option", "logits_sha1", "p_ids", "logp_option", "logp_ids", "seed", "step")
G4_NOISED = ("base_s500", "R_s500", "base_s999", "base_s500_rep2")


class Reader:
    """Loads runs under a results root and remembers every file it read (for the fingerprints)."""

    def __init__(self, results_dir: Path):
        self.root = Path(results_dir)
        self.files: list[Path] = []

    def rows(self, exp: str, flag: str) -> dict[int, dict]:
        path = self.root / exp / flag / "predictions.jsonl"
        self.files.append(path)
        with open(path, encoding="utf-8") as f:
            return {int(r["index"]): r for r in map(json.loads, f)}

    def config(self, exp: str, flag: str) -> dict:
        path = self.root / exp / flag / "config.yaml"
        self.files.append(path)
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def exists(self, exp: str, flag: str) -> bool:
        d = self.root / exp / flag
        return (d / "predictions.jsonl").exists() and (d / "config.yaml").exists()


def tune_subsets() -> dict[str, list[int]]:
    from effvar.data import load_subset
    return {f: list(load_subset(f, "tune").indices) for f in FLAGS}


# --------------------------------------------------------------------------- shared checks
def provenance(cfgs: list[tuple[str, dict]], trees: dict) -> list[str]:
    """Guard 6: every run started and ended on a clean tree whose code equals the code commit's."""
    bad = []
    for label, cfg in cfgs:
        for when in ("git", "git_end"):
            g = cfg.get(when) or {}
            if g.get("dirty") or g.get("code_changes"):
                bad.append(f"{label} {when}: tree not clean")
            if g.get("code_trees") != trees:
                bad.append(f"{label} {when}: code trees differ from the code commit")
    return bad


def compare_rows(new: dict[int, dict], saved: dict[int, dict], skip=SKIP) -> dict:
    """Every saved field except VRAM and the (null) timings must be equal; p_option exactly. Images of `new` only."""
    worst, bad, missing = 0.0, [], set()
    for i in sorted(new):
        ref = saved[i]
        a, b = new[i]["p_option"], ref["p_option"]
        d = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("inf")
        worst = max(worst, d)
        fields = [k for k in ref if k not in skip]
        miss = [k for k in fields if k not in new[i]]
        missing.update(miss)
        if miss or d != 0.0 or any(new[i][k] != ref[k] for k in fields if k in new[i]):
            bad.append(i)
    return {"identical": len(new) - len(bad), "n": len(new), "max_abs_dp": worst, "differing": bad,
            "missing_fields": sorted(missing)}


def replaced_records(reader: Reader) -> list[str]:
    root = reader.root / V1_EXP / "_replaced"
    return sorted(p.relative_to(reader.root).as_posix() for p in root.glob("*/**/config.yaml")) if root.exists() \
        else []


# --------------------------------------------------------------------------- guards
def guard1(reader: Reader, trees: dict, subsets: dict) -> tuple[bool, list[str], list]:
    specs, ok = smoke_specs(), True
    lines = ["guard 1: identity smoke (coefficients (1, 0), uncached) vs the saved clean runs, first 10 per dataset"]
    cfgs = []
    for name, arm in IDENTITY.items():
        sub = specs[name]
        probe_max, hit, n, worst, detail = 0.0, 0, 0, 0.0, []
        for flag in FLAGS:
            if not reader.exists(sub.exp, flag):
                detail.append(f"  {name} {flag}: MISSING")
                ok = False
                continue
            new, cfg = reader.rows(sub.exp, flag), reader.config(sub.exp, flag)
            cfgs.append((f"{name}/{flag}", cfg))
            saved = reader.rows(SAVED_RUNS[arm], flag)
            want = subsets[flag][:10]
            if sorted(new) != sorted(want):
                detail.append(f"  {name} {flag}: images {sorted(new)} != first 10 {want}")
                ok = False
            if cfg["decoding"]["max_new_tokens"] != MAX_NEW_TOKENS or not cfg["subrun"]["identity"]:
                detail.append(f"  {name} {flag}: not an identity run at max_new_tokens {MAX_NEW_TOKENS}")
                ok = False
            if any(r["c1"] != 1.0 or r["c2"] != 0.0 for r in new.values()):
                detail.append(f"  {name} {flag}: coefficients are not (1, 0)")
                ok = False
            probe_max = max([probe_max] + [r["feat_maxabs_vs_clean_cache"] for r in new.values()])
            r = compare_rows(new, saved)
            hit, n, worst = hit + r["identical"], n + r["n"], max(worst, r["max_abs_dp"])
            if r["differing"] or r["missing_fields"]:
                detail.append(f"  {name} {flag}: DIFFER {r['differing']} missing fields {r['missing_fields']}")
        ok &= probe_max == 0.0 and hit == n == 30
        lines.append(f"  {name}: fresh vs cached clean features, max |delta| = {probe_max} (must be 0; if guard 1 "
                     "fails, compare these first)")
        lines.append(f"  {name}: {hit}/{n} rows identical, max |delta p_option| {worst}")
        lines += detail
    return ok, lines, cfgs


def guard3(reader: Reader, trees: dict, subsets: dict) -> tuple[bool, list[str], list]:
    specs, ok = smoke_specs(), True
    lines = [f"guard 3: smoke re-run in a second process reproduces {', '.join(GUARD3_FIELDS)} bitwise"]
    cfgs = []
    for a, b in RERUN_PAIRS:
        same = n = 0
        for flag in FLAGS:
            if not (reader.exists(specs[a].exp, flag) and reader.exists(specs[b].exp, flag)):
                lines.append(f"  {a} / {b} {flag}: MISSING")
                ok = False
                continue
            ra, rb = reader.rows(specs[a].exp, flag), reader.rows(specs[b].exp, flag)
            ca, cb = reader.config(specs[a].exp, flag), reader.config(specs[b].exp, flag)
            cfgs += [(f"{a}/{flag}", ca), (f"{b}/{flag}", cb)]
            if ca["process_record"] == cb["process_record"]:
                lines.append(f"  {a} / {b} {flag}: same process; the re-run must be a second process")
                ok = False
            if sorted(ra) != sorted(rb) or sorted(ra) != sorted(subsets[flag][:10]):
                lines.append(f"  {a} / {b} {flag}: image sets differ")
                ok = False
            for i in sorted(set(ra) & set(rb)):
                n += 1
                diff = [k for k in GUARD3_FIELDS if ra[i][k] != rb[i][k]]
                if diff:
                    lines.append(f"  {a} / {b} {flag} index {i}: DIFFER {diff}")
                else:
                    same += 1
        ok &= same == n == 30
        lines.append(f"  {a} vs {b}: {same}/{n} rows bitwise")
    return ok, lines, cfgs


def detect_layout(rows_by_flag: dict[str, set], subsets: dict) -> str | None:
    full = all(rows_by_flag.get(f) == set(subsets[f]) for f in FLAGS)
    fallback = all(rows_by_flag.get(f) == set(subsets[f][:50]) | ({258} if f == "retinamnist" else set())
                   for f in FLAGS)
    return "full" if full else "fallback" if fallback else None


def recorded_layout(reader: Reader, flags=FLAGS) -> str | None:
    vals = {reader.config(f"{V1_EXP}/clean_base", f).get("clean_rerun") for f in flags
            if reader.exists(f"{V1_EXP}/clean_base", f)}
    return vals.pop() if len(vals) == 1 else None


def guard7(reader: Reader, trees: dict, subsets: dict) -> tuple[bool, list[str], list]:
    ok, cfgs = True, []
    lines = ["guard 7 (+ Amendment V1-1(2)): the clean per-id re-run reproduces the saved p_option exactly"]
    layout = recorded_layout(reader)
    if layout is None:
        return False, lines + ["  clean_base: missing, or its datasets record different --clean-rerun choices"], cfgs
    specs = g4_specs(layout)
    present = {}
    for name, arm in (("clean_base", "base"), ("clean_R", "var")):
        sub = specs[name]
        want = selection(sub, subsets)
        hit = n = 0
        worst = 0.0
        for flag in want:
            if not reader.exists(sub.exp, flag):
                lines.append(f"  {name} {flag}: MISSING")
                ok = False
                continue
            new, cfg = reader.rows(sub.exp, flag), reader.config(sub.exp, flag)
            cfgs.append((f"{name}/{flag}", cfg))
            present[(arm, flag)] = set(new)
            if cfg["mode"] != "clean" or cfg["decoding"]["max_new_tokens"] != MAX_NEW_TOKENS:
                lines.append(f"  {name} {flag}: not a clean re-run at max_new_tokens {MAX_NEW_TOKENS}")
                ok = False
            r = compare_rows(new, reader.rows(SAVED_RUNS[arm], flag))
            hit, n, worst = hit + r["identical"], n + r["n"], max(worst, r["max_abs_dp"])
            if r["differing"] or r["missing_fields"]:
                lines.append(f"  {name} {flag}: DIFFER {r['differing']} missing fields {r['missing_fields']}")
        expected = sum(len(v) for v in want.values())
        ok &= hit == n == expected
        lines.append(f"  {name}: {hit}/{n} rows identical (expected {expected}), max |delta p_option| {worst}")
    detected = detect_layout({f: present.get(("base", f), set()) for f in FLAGS}, subsets)
    ok &= detected == layout
    lines.append(f"  clean_base layout: recorded --clean-rerun {layout}, detected from the rows {detected}")
    for arm, by_flag in MISMATCH_ROWS.items():
        for flag, idx in by_flag.items():
            miss = sorted(set(idx) - present.get((arm, flag), set()))
            ok &= not miss
            lines.append(f"  mismatch rows {arm} {flag} {list(idx)}: " + ("present" if not miss else f"MISSING {miss}"))
    return ok, lines, cfgs


def guard8(reader: Reader, trees: dict, subsets: dict) -> tuple[bool, list[str], list]:
    ok, cfgs = True, []
    lines = ["guard 8: silent-null checks"]
    layout = recorded_layout(reader)
    if layout is None:
        return False, lines + ["  clean_base: missing or inconsistent --clean-rerun record"], cfgs
    specs = g4_specs(layout)
    saved_digest = {reader.config(SAVED_RUNS[a], f)["vision_cache_stats"]["key_digest"]
                    for a in SAVED_RUNS for f in FLAGS}
    digests, mnt = {}, {}
    for name, sub in specs.items():
        want = selection(sub, subsets)
        hits = misses = 0
        for flag, idx in want.items():
            if not reader.exists(sub.exp, flag):
                lines.append(f"  {name} {flag}: MISSING")
                ok = False
                continue
            cfg = reader.config(sub.exp, flag)
            rows = reader.rows(sub.exp, flag)
            cfgs.append((f"{name}/{flag}", cfg))
            if sorted(rows) != sorted(idx):
                lines.append(f"  {name} {flag}: {len(rows)} rows, registered {len(idx)}")
                ok = False
            st = cfg["vision_cache_stats"] or {}
            hits, misses = hits + st.get("hits", 0), misses + st.get("misses", 0)
            digests.setdefault(name, set()).add(st.get("key_digest"))
            if name in G4_NOISED:
                mnt.setdefault(cfg["decoding"]["max_new_tokens"], []).append(f"{name}/{flag}")
        good = (hits, misses) == tuple(sub.expect)
        ok &= good
        lines.append(f"  {name}: hits {hits}, misses {misses} (registered {tuple(sub.expect)})"
                     + ("" if good else "  FAIL"))
    one = {n: (next(iter(d)) if len(d) == 1 else None) for n, d in digests.items()}   # None = missing or mixed
    same500 = one.get("base_s500") == one.get("R_s500") == one.get("base_s500_rep2") is not None
    split = one.get("base_s999") not in (None, one.get("base_s500"))
    clean_same = (one.get("clean_base") == one.get("clean_R") is not None and len(saved_digest) == 1
                  and one.get("clean_base") in saved_digest)
    ok &= same500 and split and clean_same
    lines.append(f"  cache directories: step 500 shared by base_s500 / R_s500 / rep2 {same500}; step 999 apart "
                 f"{split}; clean re-runs on the saved runs' clean cache {sorted(saved_digest)} {clean_same}")
    if len(mnt) == 1:
        lines.append(f"  max_new_tokens of the G4 noised runs: {next(iter(mnt))} (all {sum(map(len, mnt.values()))})")
    else:
        ok = False
        lines.append(f"  max_new_tokens of the G4 noised runs DIFFER: {mnt}")
    for name in ("base_s500", "base_s999"):          # the first noised passes start from an empty noised cache
        first = next(iter(selection(specs[name], subsets)))
        if reader.exists(specs[name].exp, first):
            ns = (reader.config(specs[name].exp, first).get("noised_cache_state") or {}).get("start")
            n0 = None if ns is None else ns.get(one.get(name), {"n_files": 0})["n_files"]
            ok &= n0 == 0
            lines.append(f"  {name}: its noised cache held {n0} files when it started (must be 0)")
    smoke_ok, smoke_lines, smoke_cfgs = smoke_part(reader)
    ok &= smoke_ok
    lines += smoke_lines
    cfgs += smoke_cfgs
    clean_ok, clean_line = clean_cache_unchanged(cfgs)
    ok &= clean_ok
    lines.append(clean_line)
    return ok, lines, cfgs


def smoke_part(reader: Reader) -> tuple[bool, list[str], list]:
    """Guard 8's smoke bullet, plus guard 3's "no noised cache is read or written": every noised smoke row's projected
    features differ from the clean cached ones; no noised cache directory held a file, or changed, during the smoke."""
    ok, lines, cfgs = True, [], []
    for name, sub in smoke_specs().items():
        for flag in FLAGS:
            if not reader.exists(sub.exp, flag):
                lines.append(f"  smoke {name} {flag}: MISSING")
                ok = False
                continue
            cfg, rows = reader.config(sub.exp, flag), reader.rows(sub.exp, flag)
            cfgs.append((f"smoke/{name}/{flag}", cfg))
            if not sub.identity:
                low = min(r["feat_maxabs_vs_clean_cache"] for r in rows.values())
                if not (low > 0.0 and cfg["decoding"]["max_new_tokens"] == MAX_NEW_TOKENS):
                    lines.append(f"  smoke {name} {flag}: min probe |delta| {low} (must be > 0) or tokens != 8")
                    ok = False
            ns = cfg.get("noised_cache_state")
            if ns is None or ns["start"] != ns["end"] or any(v["n_files"] for v in ns["start"].values()):
                lines.append(f"  smoke {name} {flag}: a noised cache was present or written during the smoke ({ns})")
                ok = False
    lines.append(f"  smoke: noised rows' projected features differ from the clean cached ones, and no noised cache "
                 f"was read or written: {ok}")
    return ok, lines, cfgs


def clean_cache_unchanged(cfgs: list) -> tuple[bool, str]:
    states = [s for _, cfg in cfgs for s in (cfg["clean_cache_state"]["start"], cfg["clean_cache_state"]["end"])]
    uniq = {json.dumps(s, sort_keys=True) for s in states}
    return len(uniq) == 1, (f"  clean cache directory unchanged across {len(states)} start/end records: "
                            f"{len(uniq) == 1} ({next(iter(uniq)) if len(uniq) == 1 else sorted(uniq)})")


def guard8_smoke(reader: Reader, trees: dict, subsets: dict) -> tuple[bool, list[str], list]:
    """G3: guard 8's smoke bullet (and guard 3's no-noised-cache rule) on the smoke and smoke-rerun outputs alone."""
    ok, lines, cfgs = smoke_part(reader)
    clean_ok, clean_line = clean_cache_unchanged(cfgs)
    return ok and clean_ok, ["guard 8 (smoke part, G3): silent-null checks on the smoke"] + lines + [clean_line], cfgs


GUARDS = {"guard1": guard1, "guard3": guard3, "guard8-smoke": guard8_smoke, "guard7": guard7, "guard8": guard8}


def run_mode(mode: str, results_dir: Path, code_commit: str, trees: dict, subsets: dict) -> tuple[bool, list, dict]:
    reader = Reader(results_dir)
    ok, lines, cfgs = GUARDS[mode](reader, trees, subsets)
    prov = provenance(cfgs, trees)
    ok &= not prov
    lines.append(f"guard 6: {len(cfgs)} run configs vs code commit {code_commit[:7]}: "
                 + ("clean, code trees equal at start and end" if not prov else "FAIL"))
    lines += [f"  {p}" for p in prov]
    rep = replaced_records(reader)
    lines.append(f"replaced runs on record: {rep or 'none'}")
    lines.append("PASS" if ok else "FAIL: stop, report, amend")
    inputs = {p.resolve().relative_to(Path(results_dir).resolve().parent).as_posix(): sha256_file(p)
              for p in dict.fromkeys(reader.files)}
    return ok, lines, {"mode": mode, "ok": ok, "code_commit": code_commit, "inputs": inputs}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in GUARDS:
        print(__doc__)
        return 2
    code_commit = git("rev-parse", code_commit_from_index(PLAN_INDEX.read_text(encoding="utf-8")) + "^{commit}")
    trees = code_trees_at(code_commit)
    ok, lines, record = run_mode(mode, RESULTS, code_commit, trees, tune_subsets())
    head = git("rev-parse", "HEAD")
    diff = git("diff", "--stat", code_commit, "HEAD", "--", "effvar", "scripts", "tests")
    lines.insert(0, f"code commit {code_commit[:7]} (plan index); HEAD {head[:7]}; effvar/ scripts/ tests/ changed "
                    f"since: {'none' if not diff else chr(10) + diff}")
    record["head"] = head
    out = RESULTS / REG_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{mode}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{mode}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
