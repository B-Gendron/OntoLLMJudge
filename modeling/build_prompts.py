"""
build_prompts.py
----------------
Prompt template loading and filling logic.

Template files live under prompt_templates/<family>/<task>.txt
so adding a new model family only requires creating a new sub-folder
with 4 template files — no code changes needed here.
"""

import json
from pathlib import Path

# Paths
SAMPLE_JSON_PATH   = Path("processed_data/output_sample.json")
MISTAKE_TYPES_PATH = Path("../prompts/modeling_mistakes.json")
PROMPTS_DIR        = Path("prompt_templates")
OUTPUT_DIR         = Path("predictor_prompts")

# Task names
TASKS = [
    "modeling_comprehension",
    "inference_capability",
    "mistake_correction_competence",
    "appropriate_vocabulary_usage",
]


def get_prompt_files(model_family: str) -> dict[str, Path]:
    """
        Returns a {task: Path} dict pointing to the template files for the given
        model family. Templates are expected at:
            prompt_templates/<model_family>/<task>.txt

        To support a new family, just create the folder and the 4 template files.
    """
    family_dir = PROMPTS_DIR / model_family
    if not family_dir.exists():
        raise FileNotFoundError(
            f"No prompt template folder found for model family '{model_family}'. "
            f"Expected: {family_dir.resolve()}"
        )
    prompt_files = {}
    for task in TASKS:
        path = family_dir / f"{task}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing template for task '{task}' in family '{model_family}'. "
                f"Expected: {path.resolve()}"
            )
        prompt_files[task] = path
    return prompt_files


# Loaders

def load_sample(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_mistake_types(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_template(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()



# Field formatters

def format_relevant_elements(relevant_elements: list[dict]) -> str:
    """
        Formats the relevant_elements list into a numbered plain-text block.
    """
    return "\n".join(
        f"{i}. {re['name']}: {re['comment']}"
        for i, re in enumerate(relevant_elements, 1)
    )

def format_unintended_outcomes(unintended_outcomes: list[dict]) -> str:
    """
        Formats the unintended_outcomes list into a numbered plain-text block.
    """
    return "\n".join(
        f"{i}. {uo['name']}: {uo['comment']}"
        for i, uo in enumerate(unintended_outcomes, 1)
    )


def build_prompts(model_family, sample_path=SAMPLE_JSON_PATH, mistake_types_path= MISTAKE_TYPES_PATH, output_dir=OUTPUT_DIR):
    """
        Loads a processed sample JSON and fills in all 4 prompt templates for the
        given model family. Saves each filled prompt as a .txt file in output_dir
        and returns them as a {task: filled_prompt} dict.
    """
    prompt_files  = get_prompt_files(model_family)
    sample        = load_sample(sample_path)
    mistake_types = load_mistake_types(mistake_types_path)

    common_fields = {
        "I":                   sample["I"],
        "OA":                  sample["OA"],
        "M_i":                 sample["M_i"],
        "mistake_description": sample["mistake_description"],
        "mistake_example":     mistake_types[sample["M_i"]]["example"],
        "generation":          sample["generation"],
    }

    extra_fields = {
        "modeling_comprehension":        {"relevant_elements":   format_relevant_elements(sample["relevant_elements"])},
        "inference_capability":          {"unintended_outcomes": format_unintended_outcomes(sample["unintended_outcomes"])},
        "mistake_correction_competence": {},
        "appropriate_vocabulary_usage":  {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filled_prompts = {}

    for task, template_path in prompt_files.items():
        template = load_template(template_path)
        filled   = template.format(**common_fields, **extra_fields[task])

        out_path = output_dir / f"prompt_{task}_{model_family}_filled.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(filled)
        print(f"Saved: {out_path}")

        filled_prompts[task] = filled

    return filled_prompts