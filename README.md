# Seen but Not Said

Code for the paper **Seen but Not Said: Visual Attention Redistribution and Contrastive Decoding in a Medical
Vision-Language Model** ([`paper.pdf`](paper.pdf); under double-blind review).

This repository holds what is needed to re-run the study: the `effvar` package, the command-line scripts, the tests,
the frozen subset files and the pinned environment. Model weights, MedMNIST images and run outputs are not included:
the steps below download the first two, and the scripts regenerate the third.

## Question and findings

We ask whether visual attention redistribution (VAR), a training-free fix reported to reduce hallucination in general
vision-language models, helps a medical one, and at what cost. We ported VAR to MedGemma-1.5-4b-it with p set to 0.4
(the released code's setting) and not varied, and tested it zero-shot on RetinaMNIST, PneumoniaMNIST and DermaMNIST
with a first-token option-letter readout and pre-registered deciding tests.

- VAR raised raw macro-F1 on PneumoniaMNIST and DermaMNIST validation images, but a per-class prior correction fitted
  on tune subsets of the training split gives the unmodified model the macro-F1 VAR reaches, and more on
  PneumoniaMNIST. On the prior-invariant one-vs-one AUC (OvO; post hoc for VAR), no difference was detectable on tune
  or validation (minimum detectable effects 0.006–0.020).
- VAR adds about 10% to its end-to-end time. No output-changing layer-restricted variant kept ≥ 90% of VAR's gain,
  and the only gain-preserving one is bitwise identical to VAR and saves 3.4 ms (0.46%) end to end.
- Linear probes show that label information decodable from image tokens is lost between the answer position's
  representation and the option logits. A pre-registered probe-based gate bounded what routing-type alterations could
  add, not readout-side ones.
- Visual contrastive decoding (VCD), pre-registered on the tune subsets at the same readout, passed its screen on no
  dataset (DermaMNIST underpowered) and lowered OvO on PneumoniaMNIST (α = 1; Holm-adjusted p = 0.001).
- With option-letter readouts, macro-F1 and quadratic weighted kappa cannot separate ranking effects from prior
  shifts, so such evaluations need a calibrated baseline and a prior-invariant ranking metric.

**Scope.** One model, three datasets, zero-shot classification and a first-token option readout. VAR was run at one
strength (p 0.4, the released code's setting), and VCD on the tune subsets only. The paper's section *Limitations and
Reproducibility* lists what was not tested.

## Methods in the code

MedGemma-1.5-4b-it pairs a 34-layer Gemma 3 decoder with a SigLIP image encoder. It runs zero-shot in bf16 with batch
size 1 and greedy decoding; each 224-px MedMNIST image is resized to 896 × 896 and encoded as 256 tokens. The frozen
prompts (`effvar/prompts.py`) list lettered options and prefill the answer with `Answer:`, so the first generated
token is the answer.

- **VAR port** (`effvar/var.py`, `effvar/methods/var.py`). Visual sink tokens are image tokens whose normalised
  activation in the sink dimensions exceeds a threshold τ. For each text token after the image, a head is
  image-centric if its attention to the image totals at least *summ*, with a share of at least ρ on non-sink image
  tokens. In those rows and heads, VAR moves a share p of the attention on all sink tokens, visual and text, to the
  non-sink image tokens in proportion to their attention. The code follows the released implementation's two hook
  sites on eager attention and is written in the VAR paper's convention (the released code inverts p and ρ). It is a
  port of the released implementation at its released settings (p_code 0.6 ≡ paper p 0.4, ρ 0.5); τ 10, *summ* 0.1
  and the sink dimensions {1313, 2511} were adapted to MedGemma on at most 20 validation images, and p was not varied.
  Dimension 443, which carries almost all of the residual norm, is zeroed before the detector's RMS normalisation, a
  disclosed deviation from the VAR paper's Appendix A.1. At p = 0 the port reproduces the unmodified model bitwise
  (G0, on 30 smoke images).
- **Layer-restricted variants** (`effvar_rank1_topm` in `effvar/var.py`; `scripts/sweep_topm.py`). VAR's edit runs in
  only m of the 24 layers it edits on these data (9–32), and the whole mechanism is skipped in the other layers.
  Timing ablations of its parts are in `effvar/ablate.py` and `scripts/ab_rounds.py`.
- **Probe-based diagnostics** (`effvar/diag.py`, `effvar/diag_stats.py`, `effvar/diag_subsets.py`, `effvar/yesno.py`,
  `scripts/diag_*.py`, `scripts/run_d2.py`). Calibrated baselines (per-class offsets fitted on tune), a shift
  decomposition, a content-free prior, linear probes on image-token and answer-row features (the routing and readout
  gaps, with a pre-registered gate), option-order reversal and per-class yes/no prompts.
- **VCD** (`effvar/vcd.py`, `effvar/vcd_cache.py`, `scripts/run_vcd.py`, `scripts/diag_vcd.py`). VCD compares the
  next-token distribution P given the image with P′ given a noised copy and scores tokens by (1 + α) log P − α log P′
  among those with P ≥ β max P. Here it is applied to the option letters at the first answer token, adding one
  prefill on the noised image to the clean pass; the noise follows VCD's released 1,000-step schedule with one seed
  per image. VCD is a separate method, not a variant of VAR.

The package is called `effvar` after the project's starting aim of testing lower-cost forms of VAR.

## Repository layout

```text
effvar/                  the package
  data.py                MedMNIST loading and the frozen subsets
  prompts.py             frozen per-dataset prompts (prompt_variants.py: the diagnostic variants)
  model.py               MedGemma loading (bf16; SDPA or eager attention) and image-token location
  generation.py          greedy generation with vision, prefill and decode timing
  scoring.py             answer parsing, first-token option probabilities, leakage and metrics
  evaluate.py, runio.py  one method on one subset; result files
  vision_cache.py        projected-image-feature cache (accuracy-only tune runs)
  var.py, methods/       the VAR port, its variants and the diagnostic family
  ablate.py, profile.py  timing-only ablations and region profiling
  diag*.py, yesno.py     diagnostic statistics, subsets and yes/no scoring
  vcd.py, vcd_cache.py   VCD and its noised-feature cache
scripts/                 command-line entry points; each docstring gives its usage
tests/                   pytest suite
data/subsets/            frozen subset files (split indices and labels)
data/const_sinks.json    modal sink sets, used only by one timing ablation
environment.yml          the pinned conda environment
paper.pdf                the paper
```

## Setup

The runs were made on Windows 11 with an NVIDIA GeForce RTX 3060 (12 GB), in a conda environment named `effvar` with
Python 3.11, CUDA 13.0 builds of PyTorch (torch 2.14.0, torchvision 0.29.0) and transformers 5.17.0. The package
requires transformers 5.17.0 or later; only the versions pinned in `environment.yml` were tested. The bf16 weights
take about 8.6 GB of GPU memory, so every run uses batch size 1.

```powershell
conda env create -f environment.yml
conda activate effvar
pip install -e . --no-deps
```

Without conda, install CUDA builds of `torch` and `torchvision` first, then run `pip install -e ".[test]"`.

### Model access

MedGemma is gated. Accept the Health AI Developer Foundations terms on the model page
(<https://huggingface.co/google/medgemma-1.5-4b-it>), then log in and download the model once:

```powershell
hf auth login
hf download google/medgemma-1.5-4b-it
python scripts/check_medgemma.py    # loads the model from the local cache and runs one tiny generation
```

`effvar/model.py` sets `HF_HUB_OFFLINE=1` unless it is already set, so runs read the model from the local Hugging Face
cache and never download it.

### Data

The three MedMNIST v2 datasets are downloaded at 224 px (about 1.4 GB) into `data/medmnist/`:

```powershell
python -c "from effvar import FLAGS; from effvar.data import DATA_DIR, load_split; DATA_DIR.mkdir(parents=True, exist_ok=True); [load_split(f, 'test', download=True) for f in FLAGS]"
```

`data/subsets/` holds the frozen subsets that every run uses. Do not regenerate them; check them against the download
with `python scripts/make_subsets.py --check` and `python scripts/make_diag_subsets.py --check`.

| Subset | Split | RetinaMNIST / PneumoniaMNIST / DermaMNIST | Used for |
|---|---|---|---|
| `smoke` | val | 30 / 30 / 30 | harness checks and the port gate |
| `tune` | train | 300 / 300 / 300 | exploration, configuration sweeps, calibration fits, VCD |
| `val` | val | 120 / 524 / 1003 (whole split) | the registered validation comparisons |
| `test` | test | 400 / 624 / 2005 (whole split) | the single test run |
| `probe` | train | 480 / 1000 / 1000 | linear probes |
| `confirm` | train | 300 / – / – | the RetinaMNIST confirmation set |

`<dataset>_tune_overlap.json` lists the two PneumoniaMNIST tune images that are pixel-identical to validation images
(excluded from calibration fits and the comparisons they govern, hence 298), and `<dataset>_test_duplicates.json`
the three exact-duplicate pairs in PneumoniaMNIST test (hence n = 621).

## Running the experiments

All scripts write under `results/`, which is not versioned. `run_eval.py` writes `results/<exp-name>/<dataset>/` with
`predictions.jsonl`, `config.yaml` (which records the git commit and whether the tree was dirty) and `metrics.json`.
Each script's docstring gives its options and registered settings.

```powershell
# Probe of MedGemma: sink dimensions, visual sinks, image-centric heads and base latency, on <= 20 val images
python scripts/probe_medgemma.py
# Port gate G0: var at p = 0 reproduces base bitwise on the smoke subsets
python scripts/check_gates.py --limit 10
# One method on one subset
python scripts/run_eval.py --method base --attn sdpa --subset val --exp-name val/base_sdpa
python scripts/run_eval.py --method var --attn var_eager --tau 10 --subset val --exp-name val/var_tau10
python scripts/run_eval.py --method effvar_rank1_topm --attn var_eager --layers 9-32 --subset val --exp-name val/effvar_topm24
# Tune runs (untimed, with the vision cache): base, then the rho x summ sweep that contains VAR's setting
python scripts/run_eval.py --method base --attn var_eager --subset tune --cache-vision --exp-name tune/base
python scripts/sweep_gate.py
```

| Part of the paper | Scripts in `scripts/` |
|---|---|
| Probe of MedGemma | `probe_medgemma.py`, `probe_sinks_followup.py` |
| VAR evaluation and gates | `run_eval.py`, `check_gates.py`, `sweep_gate.py` |
| Layer-restricted variants | `sweep_topm.py`, `check_lossless.py` |
| Timing | `ab_rounds.py`, `ab_report.py`, `check_ablate.py`, `check_batch.py`, `m3_summary.py`, `probe_batch_sinks.py` |
| Test split | `export_summary.py`, `test_report.py`, `find_duplicates.py` |
| Probe-based diagnostics | `make_diag_subsets.py`, `check_diag.py`, `check_tune_overlap.py`, `diag_a1.py`, `diag_a2.py`, `diag_b1.py`, `diag_b2_extract.py`, `diag_b2_probe.py`, `diag_depth.py`, `diag_c4.py`, `run_d2.py`, `diag_d2.py`, `check_d2_regression.py`, `diag_mde.py`, `diag_invariant_auc.py`, `s1_noise.py` |
| VCD | `run_vcd.py`, `check_vcd_regression.py`, `diag_vcd.py` |

Timed runs never use the vision cache: `--cache-vision` switches to the untimed path and may only write under
`results/tune/`. Several runners refuse to start while tracked files have uncommitted changes.

### VCD

`scripts/run_vcd.py` reuses the clean tune runs `results/tune/base` and `results/tune/var/rho0.5_summ0.1` (written by
the commands above). It runs a smoke check in two processes, then the tune runs; the paper's tune runs were made as
single runs, one process per arm (`--arm base|var` with `--step 500|999` or `--clean`; see the docstring).
`scripts/check_vcd_regression.py` checks each stage, and `scripts/diag_vcd.py` analyses the runs once the checks pass.

```powershell
python scripts/run_vcd.py --preset smoke
python scripts/run_vcd.py --preset smoke-rerun
python scripts/check_vcd_regression.py guard1
python scripts/check_vcd_regression.py guard3
python scripts/check_vcd_regression.py guard8-smoke
python scripts/run_vcd.py --preset g4 --clean-rerun full
python scripts/check_vcd_regression.py guard7
python scripts/check_vcd_regression.py guard8
python scripts/diag_vcd.py
```

These scripts also enforce the study's pre-registration guard: before loading the model they require a clean tree and
code identical to a registered code commit, which they read from a row ``| G2 | ... | done `<commit>` |`` in
`notes/vcd_plan.md`. That record is not part of this repository. To run them, create `notes/vcd_plan.md` (ignored by
git) containing one such row that names the commit you run from, for example ``| G2 | code | done `0123abc` |``,
with the output of `git rev-parse --short HEAD` in place of `0123abc`.

## Tests

```powershell
python -m pytest tests/
```

The tests check the VAR port against a transcription of the released code, the prompts and their tokenisation, the
subset files, scoring, the vision cache, and the VCD arithmetic and guards. Some tests need local files: the
subset-reproduction tests in `tests/test_data.py` need the MedMNIST download; the prompt, D2 and VCD tests need the
MedGemma processor in the Hugging Face cache; and `tests/test_vision_cache.py` loads the full model on the GPU.
Without a GPU, run `python -m pytest tests/ --ignore=tests/test_vision_cache.py`.

## Reproducibility and limitations

- Bitwise reproducibility was shown at batch size 1 only, on the machine and environment above. Batching was measured
  for latency alone, so accuracy at batch sizes above 1 is not established.
- This is the final version of the study's code. The paper's test results came from an earlier version on a clean
  tree, and later code changes were verified output-equivalent on validation rehearsals and tune reruns.
- The test split was run once, before the diagnostic and VCD experiments; its accuracy contrasts are unadjusted and
  exploratory. Exact-pixel duplicates were removed within test only (3 PneumoniaMNIST pairs); validation and tune were
  not checked for duplicates, and no split was checked for near-duplicates.
- VAR was run at one strength (p 0.4). Neither the value printed in the VAR paper nor the registered grid of p values
  was run, and the sink dimensions were adapted on at most 20 validation images.
- VCD used one noise schedule and one noise seed per image, on tune only.
- The pre-registration records (plans and amendments, committed to the project's research history before the outputs
  they govern; there was no external registry) are not part of this repository.
- The code and its outputs are for research only and are not intended for clinical use.

## Citation

The paper is under double-blind review; a citation entry will be added when it is published.

## Licence and acknowledgements

The code is released under the Apache License 2.0 ([`LICENSE`](LICENSE)). Code adapted from VAR and VCD, the model's
terms and the dataset licences are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

This work builds on:

- **VAR**: S. Kang, J. Kim, J. Kim and S. J. Hwang, "See What You Are Told: Visual Attention Sink in Large Multimodal
  Models", ICLR 2025; code at <https://github.com/seilk/VisAttnSink>.
- **VCD**: S. Leng et al., "Mitigating Object Hallucinations in Large Vision-Language Models through Visual
  Contrastive Decoding", CVPR 2024; code at <https://github.com/DAMO-NLP-SG/VCD>.
- **MedGemma**: A. Sellergren et al., "MedGemma Technical Report", arXiv:2507.05201, 2025; used under the Health AI
  Developer Foundations Terms of Use.
- **Gemma 3**: Gemma Team, "Gemma 3 Technical Report", arXiv:2503.19786, 2025.
- **SigLIP**: X. Zhai, B. Mustafa, A. Kolesnikov and L. Beyer, "Sigmoid Loss for Language Image Pre-Training",
  ICCV 2023.
- **MedMNIST**: J. Yang et al., "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification", Scientific Data 10, 41 (2023); J. Yang, R. Shi and B. Ni, "MedMNIST Classification Decathlon: A
  Lightweight AutoML Benchmark for Medical Image Analysis", ISBI 2021.
- **Source datasets**: DeepDRiD (R. Liu et al., Patterns 3(6), 100512, 2022) for RetinaMNIST; D. S. Kermany et al.,
  Cell 172(5), 1122–1131, 2018, for PneumoniaMNIST; HAM10000 (P. Tschandl, C. Rosendahl and H. Kittler, Scientific
  Data 5, 180161, 2018) and the ISIC 2018 challenge (N. Codella et al., arXiv:1902.03368, 2019) for DermaMNIST.
