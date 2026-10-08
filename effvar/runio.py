"""Result files for one (experiment, dataset): predictions.jsonl, config.yaml, metrics.json."""
import json
import shutil
import subprocess
from pathlib import Path

import torch
import transformers
import yaml

from effvar import MODEL_ID, ROOT

RESULTS_DIR = ROOT / "results"


def prepare_dir(exp_name: str, flag: str, overwrite: bool) -> Path:
    path = RESULTS_DIR / exp_name / flag
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def write_config(path: Path, *, method, spec, subset, token_sets, max_new_tokens: int, seed: int,
                 warmup_runs: int, limit: int | None, extra: dict | None = None) -> None:
    cfg = {
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "method": method.name,
        "method_params": method.params(),
        "attn": method.attn,
        "dataset": spec.flag,
        "subset": {"name": subset.name, "split": subset.split, "n": len(subset.indices)},
        "limit": limit,
        "prompt": spec.text(),
        "labels": list(spec.labels),
        "option_texts": list(spec.option_texts),
        "option_token_sets": token_sets,
        "decoding": {"greedy": True, "max_new_tokens": max_new_tokens},
        "seed": seed,
        "warmup_runs": warmup_runs,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "git": git_state(),
        **(extra or {}),
    }
    (path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def write_predictions(path: Path, rows: list[dict]) -> None:
    with (path / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_metrics(path: Path, metrics: dict) -> None:
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_predictions(path: Path) -> list[dict]:
    """Rows of `<path>/predictions.jsonl`, sorted by image index.

    `path` is a run/flag directory, e.g. results/test/var_tau10/dermamnist.
    """
    with (path / "predictions.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return sorted(rows, key=lambda r: r["index"])


def load_config(path: Path) -> dict:
    return yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
