# Third-party notices

This repository contains no model weights and no dataset images. It contains code adapted from two projects licensed
under the Apache License, Version 2.0, and it uses a gated model and public datasets that keep their own terms.

## VAR: VisAttnSink (Apache License 2.0)

Source: <https://github.com/seilk/VisAttnSink>, the code of Kang, Kim, Kim and Hwang, *See What You Are Told: Visual
Attention Sink in Large Multimodal Models*, ICLR 2025. Licensed under the Apache License, Version 2.0; the full text
is in [`LICENSE`](LICENSE).

Adapted parts: the sink-dimension detector (`DimProspector`), the image-centric head selection (`HeadFork`) and the
attention redistribution (`VARProcessor.attn_redist`) of `src/logic/logic.py`, and the two hook sites of
`src/model/llava.py`. Files containing adapted code, each citing the source lines it follows:

- `effvar/var.py`
- `effvar/methods/var.py`
- `tests/test_var.py` (`literal_attn_redist`, a transcription of `logic.py` lines 163-212, used as a test reference)

Changes made: ported from the LLaVA implementation to MedGemma's Gemma 3 decoder through a registered attention
implementation and a decoder-layer pre-hook; rewritten in the VAR paper's convention (the released code writes the
redistributed share and the head-selection ratio inverted); vectorised; sink dimensions and thresholds adapted to
MedGemma; an option to skip the last layer, a layer restriction (`effvar_rank1_topm`), an attention-defined sink
variant (`var_attn`), configuration variants for diagnostics (`effvar/diag.py`, `effvar/methods/diag.py`) and
instrumentation added.

## VCD (Apache License 2.0)

Source: <https://github.com/DAMO-NLP-SG/VCD>, the code of Leng et al., *Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024. Licensed under the Apache License, Version 2.0.

Re-implemented in `effvar/vcd.py`: the noise schedule of `vcd_utils/vcd_add_noise.py` (sigmoid schedule, 1000 steps),
the contrastive score (1 + α) log P − α log P′ and the plausibility cutoff P ≥ β max P, applied here to the answer
options at the first generated token.

## MedGemma (Health AI Developer Foundations Terms of Use)

HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at
https://developers.google.com/health-ai-developer-foundations/terms

This repository does not contain or redistribute MedGemma. The code loads `google/medgemma-1.5-4b-it` from the local
Hugging Face cache after you have accepted those terms and downloaded the model. The code and its outputs are for
research only and are not intended for clinical use.

## MedMNIST

- The `medmnist` package (<https://github.com/MedMNIST/MedMNIST>) is licensed under the Apache License 2.0.
- RetinaMNIST (from DeepDRiD) and PneumoniaMNIST (from Kermany et al., 2018) are licensed under CC BY 4.0, and
  DermaMNIST (from HAM10000) under CC BY-NC 4.0.
- No images are included. The files in `data/subsets/` list split indices and class labels from these datasets and
  remain subject to their licences; the DermaMNIST files may not be used commercially.
