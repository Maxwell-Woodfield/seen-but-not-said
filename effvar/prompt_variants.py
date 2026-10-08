"""Phase 6 prompt variants (notes/diagnosis_plan.md §2: prompt variants live here; `prompts.py` stays frozen).

Each variant is built from the frozen TaskSpec and changes only what its card varies.

- `reversed` (C4): the options are listed in reverse class order and lettered A.. in listing order, so
  letter A names the LAST class. `labels[i]` is still the letter of class i, so `option_token_sets`,
  `first_token_metrics` (p_option) and `parse_answer` all come back class-aligned: letters are mapped back
  to classes before anything is scored. The rendered text uses the frozen format string unchanged.
- `yesno_specs` (D2, Amendment 3b): one yes/no question per class. It is NOT in `VARIANTS`: it is not a letter
  prompt, so `scripts/run_eval.py` cannot select it; `scripts/run_d2.py` runs it and `effvar/yesno.py` scores it.
"""
from dataclasses import dataclass

from effvar.prompts import TaskSpec, get_spec

YESNO_QUESTION = "Does this {modality} show {option}?"
YESNO_INSTRUCTION = "Answer yes or no."


@dataclass(frozen=True)
class ReorderedSpec(TaskSpec):
    """A TaskSpec whose options are displayed in letter order rather than class order.

    `labels` and `option_texts` stay indexed by class; only `text()` and `instruction` change, via the
    frozen TaskSpec rendering of the letter-sorted options.
    """

    def display(self) -> TaskSpec:
        order = sorted(range(len(self.labels)), key=self.labels.__getitem__)
        return TaskSpec(flag=self.flag, modality=self.modality, question=self.question,
                        labels=tuple(self.labels[i] for i in order),
                        option_texts=tuple(self.option_texts[i] for i in order))

    @property
    def instruction(self) -> str:
        return self.display().instruction

    def text(self) -> str:
        return self.display().text()


def reversed_spec(flag: str) -> ReorderedSpec:
    """C4: class i gets the letter of class K-1-i in the frozen spec (class K-1 -> A)."""
    s = get_spec(flag)
    return ReorderedSpec(flag=s.flag, modality=s.modality, question=s.question,
                         labels=s.labels[::-1], option_texts=s.option_texts)


@dataclass(frozen=True)
class YesNoSpec(TaskSpec):
    """D2: the yes/no question about class `cls` of the frozen TaskSpec.

    `text()` = "Does this {modality} show {class}? Answer yes or no.", with the frozen modality and option text
    verbatim. `messages()` is inherited, so the chat template and the "Answer:" prefill are the frozen prompt's.
    `labels` / `option_texts` stay the frozen, class-indexed ones; score with effvar/yesno.py, never with
    `option_token_sets` / `parse_answer`.
    """

    cls: int = 0

    @property
    def instruction(self) -> str:
        return YESNO_INSTRUCTION

    def text(self) -> str:
        return f"{YESNO_QUESTION.format(modality=self.modality, option=self.option_texts[self.cls])} {self.instruction}"


def yesno_specs(flag: str) -> tuple[YesNoSpec, ...]:
    """D2: the K yes/no specs of a dataset, in class order (spec c asks about class c)."""
    s = get_spec(flag)
    return tuple(YesNoSpec(flag=s.flag, modality=s.modality, question=s.question, labels=s.labels,
                           option_texts=s.option_texts, cls=c) for c in range(len(s.labels)))


VARIANTS = {"frozen": get_spec, "reversed": reversed_spec}


def variant_spec(name: str, flag: str) -> TaskSpec:
    if name not in VARIANTS:
        raise ValueError(f"unknown prompt variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name](flag)
