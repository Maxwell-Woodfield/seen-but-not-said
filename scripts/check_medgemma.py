"""Setup check: load MedGemma-1.5-4b-it from the local Hugging Face cache and run one tiny generation.

Run from the project root with the `effvar` env active:
    python scripts/check_medgemma.py
"""
import os

# Offline mode: fail fast if the model isn't in the cache instead of silently re-downloading 8.6 GB.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time

import torch
from huggingface_hub import constants
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/medgemma-1.5-4b-it"


def main() -> None:
    print(f"HF cache dir : {constants.HF_HUB_CACHE}")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: reinstall the CUDA build of PyTorch.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU          : {props.name} | {props.total_memory / 2**30:.1f} GiB VRAM")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"Loaded in {time.perf_counter() - t0:.1f} s | "
          f"weights on GPU: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    image = Image.new("RGB", (224, 224), color=(128, 128, 128))  # plain grey test image
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    n_in = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    torch.cuda.synchronize()

    print(f"Prompt tokens: {n_in} (includes the image tokens)")
    print(f"Output       : {processor.decode(out[0][n_in:], skip_special_tokens=True)!r}")
    print(f"Generate time: {time.perf_counter() - t0:.2f} s | "
          f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("OK: MedGemma loads from the cache and generates on the GPU.")


if __name__ == "__main__":
    main()
