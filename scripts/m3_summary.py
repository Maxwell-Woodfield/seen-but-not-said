"""Measurement 3 across batch sizes: R's overhead per image AND the work R did, per B. TIMING ONLY.

Each `ab_rounds.py --set m3 --batch B` run is its own interleaved, mirrored session with its own `S`
and `R_dup`, so every B's dR_LM is a within-session contrast. This lines them up, and puts beside each
the per-image workload of `R` (rows_edited, edit_events, n_vis, n_text): batching flips VAR's hard
thresholds (notes/sink_geometry.md), so a batch that got cheaper by editing LESS is not a regime
result. Every B uses the same images, so each image's rows_edited at B is matched against its own
value at B = 1, not just compared on average.

    python scripts/m3_summary.py [results/timing/m3_batch]

Reads the newest run under each B<k>/ and writes <root>/summary.md.
"""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _ab_report():
    spec = importlib.util.spec_from_file_location("ab_report", Path(__file__).with_name("ab_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_image_rows_edited(rows: list[dict], arm: str) -> dict[tuple, int]:
    """(dataset, index) -> rows_edited for `arm`, from its first round (identical batches every round)."""
    out: dict[tuple, int] = {}
    for r in rows:
        vs = r.get("var_stats")
        if r["arm"] != arm or not vs or "per_row" not in vs:
            continue
        for idx, n in zip(r["indices"], vs["per_row"]["rows_edited"]):
            out.setdefault((r["dataset"], idx), n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results/timing/m3_batch")
    args = ap.parse_args()
    rep = _ab_report()
    root = ROOT / args.root

    runs = {}
    for bdir in sorted(root.glob("B*")):
        latest = sorted(d for d in bdir.iterdir() if (d / "rows.jsonl").exists())
        if latest:
            run = latest[-1]
            manifest = yaml.safe_load((run / "manifest.yaml").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (run / "rows.jsonl").open(encoding="utf-8")]
            rep.add_run_metric(rows)
            runs[int(manifest["batch"])] = (run, rows)
    if not runs:
        raise SystemExit(f"no m3 runs under {root}")

    ref = per_image_rows_edited(runs[min(runs)][1], "R")
    out = [f"# Measurement 3 -- batch regime ({', '.join(f'B={b}' for b in sorted(runs))})", "",
           "TIMING ONLY. Per image = batch time / B. dR_LM = `R - S` within each B's own session; "
           "`R` on the per-row path at every B, including B = 1.", "",
           "| B | dR_LM as run [95% CI] | dR_LM D = 7 [95% CI] | S / R as run (ms/img) | R rows_edited/img | "
           "edit_events/img | n_vis | n_text | images with rows_edited != B=" + str(min(runs)) + " | run |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for B, (run, rows) in sorted(runs.items()):
        m_run, m_d7 = rep.round_means(rows, "lm_run_ms"), rep.round_means(rows, "lm_ms")
        stats = [r["var_stats"] for r in rows if r["arm"] == "R" and r.get("var_stats")]

        def mean(key):
            xs = [s[key] for s in stats if s.get(key) is not None]
            return f"{statistics.fmean(xs):.2f}" if xs else "n/a"
        mine = per_image_rows_edited(rows, "R")
        common = sorted(set(mine) & set(ref))
        changed = [k for k in common if mine[k] != ref[k]]
        dmax = max((abs(mine[k] - ref[k]) for k in changed), default=0)
        s_run = statistics.fmean(list(m_run["S"].values()))
        r_run = statistics.fmean(list(m_run["R"].values()))
        out.append(f"| {B} | {rep.fmt(rep.contrast(m_run, 'R', 'S'))} | {rep.fmt(rep.contrast(m_d7, 'R', 'S'))} | "
                   f"{s_run:.1f} / {r_run:.1f} | {mean('rows_edited')} | {mean('edit_events')} | {mean('n_vis')} | "
                   f"{mean('n_text')} | {len(changed)}/{len(common)} (max |d| {dmax}) | `{run.name}` |")
    out += ["", "Noise floor per B: `R_dup - R` in each run's own `report_lm_run_ms.md`."]
    (root / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
