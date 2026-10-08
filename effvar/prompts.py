"""Frozen per-dataset prompts. Every method uses exactly these; the rendered text goes into config.yaml.

Answer labels are letters for all three datasets, chosen so each option has a unique single-token first
token in the Gemma tokenizer (notes/medgemma_probe.md §G). RetinaMNIST uses A-E for grades 0-4, with the
grade spelled out in the option text; the class index stays 0-4, so QWK is unaffected.

The assistant turn is prefilled with ANSWER_PREFIX, so the answer letter is the first generated token
(format check in notes/status.md; wording frozen 2026-09-17).
"""
from dataclasses import dataclass

import medmnist

ANSWER_PREFIX = "Answer:"


@dataclass(frozen=True)
class TaskSpec:
    flag: str
    modality: str
    question: str
    labels: tuple[str, ...]        # answer label for class index i
    option_texts: tuple[str, ...]  # human-readable option for class index i

    @property
    def instruction(self) -> str:
        return f"Answer with a single letter ({self.labels[0]}-{self.labels[-1]}) and nothing else. Do not explain."

    def text(self) -> str:
        options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(self.labels, self.option_texts))
        return f"This is a {self.modality}. {self.question}\nOptions:\n{options}\n{self.instruction}"

    def messages(self, image) -> list[dict]:
        return [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": self.text()}]},
                {"role": "assistant", "content": [{"type": "text", "text": ANSWER_PREFIX}]}]


def _derma_texts() -> tuple[str, ...]:
    labels = medmnist.INFO["dermamnist"]["label"]
    return tuple(labels[k] for k in sorted(labels, key=int))


SPECS = {
    "retinamnist": TaskSpec(
        flag="retinamnist",
        modality="retinal fundus photograph",
        question="What is the diabetic retinopathy grade?",
        labels=("A", "B", "C", "D", "E"),
        option_texts=("grade 0 - no diabetic retinopathy",
                      "grade 1 - mild non-proliferative diabetic retinopathy",
                      "grade 2 - moderate non-proliferative diabetic retinopathy",
                      "grade 3 - severe non-proliferative diabetic retinopathy",
                      "grade 4 - proliferative diabetic retinopathy"),
    ),
    "pneumoniamnist": TaskSpec(
        flag="pneumoniamnist",
        modality="pediatric chest X-ray",
        question="Which finding does this chest X-ray show?",
        labels=("A", "B"),
        option_texts=("normal", "pneumonia"),
    ),
    "dermamnist": TaskSpec(
        flag="dermamnist",
        modality="dermatoscopic image of a skin lesion",
        question="What is the diagnosis?",
        labels=("A", "B", "C", "D", "E", "F", "G"),
        option_texts=_derma_texts(),
    ),
}


def get_spec(flag: str) -> TaskSpec:
    spec = SPECS[flag]
    n_classes = len(medmnist.INFO[flag]["label"])
    assert len(spec.labels) == len(spec.option_texts) == n_classes, flag
    return spec
