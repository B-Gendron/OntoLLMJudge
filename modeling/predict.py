"""
predict.py
--------------
Runs the four EAS predictors on a single input sample, writes the results into a fresh copy of the Expert Assessment Schema ontology (using owlready2), fires the HermiT reasoner so that `hasScoringLevel` assertions are derived, and returns (plus optionally saves) the final scoring dictionary as a JSON file.

Usage
-----
    python predict.py \
        --I         "A Photographer takes pictures of at least 10 Guests." \
        --OA        "Photographer subClassOf (takesPictureOf min 10 Guest)" \
        --M_i       "opr_mistake_type_1" \
        --generation "The restriction is wrong because the range of takesPictureOf is Wedding, not Guest." \
        --output_json       prediction_outputs/scored_explanation.json \
        --adapter_dir       trained_adapters/best \
        --sample_id         explanation_42
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from owlready2 import get_ontology, sync_reasoner_pellet, sync_reasoner_hermit

# Local imports
from config import MODEL_CONFIGS, detect_model_family, get_label_token_ids
from build_prompts import load_mistake_types, load_template, get_prompt_files, MISTAKE_TYPES_PATH
from process_data_sample import process_sample, ONTOLOGY_PATH


BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Mappings of tasks to their ontology concepts / the ontology concepts of their outcomes

TASK_TO_QD_INDIVIDUAL = {
    "modeling_comprehension":        "qd1_modeling_comprehension",
    "inference_capability":          "qd2_inference_capability",
    "mistake_correction_competence": "qd3_mistake_correction_competence",
    "appropriate_vocabulary_usage":  "qd4_appropriate_vocabulary_usage",
}

LABEL_TO_SUBCLASS = {
    ("modeling_comprehension", "void"):    "AbsenceOfModelingComprehension",
    ("modeling_comprehension", "partial"): "PartiallyCorrectModelingComprehension",
    ("modeling_comprehension", "correct"): "CorrectModelingComprehension",
    ("modeling_comprehension", "wrong"):   "IncorrectModelingComprehension",
    ("inference_capability",   "void"):    "AbsenceOfInferenceCapability",
    ("inference_capability",   "partial"): "PartiallyCompleteInferenceCapability",
    ("inference_capability",   "correct"): "CompleteInferenceCapability",
    ("inference_capability",   "wrong"):   "IncorrectInferenceCapability",
    ("mistake_correction_competence", "void"):    "AbsenceOfMistakeCorrectionCompetence",
    ("mistake_correction_competence", "correct"): "CorrectMistakeCorrectionCompetence",
    ("mistake_correction_competence", "wrong"):   "IncorrectMistakeCorrectionCompetence",
    ("appropriate_vocabulary_usage", "correct"): "CorrectVocabularyUsage",
    ("appropriate_vocabulary_usage", "wrong"):   "IncorrectVocabularyUsage",
}

TASKS = list(TASK_TO_QD_INDIVIDUAL.keys())

ONTO_NS = "http://www.semanticweb.org/user/ontologies/2026/2/expertassessmentschema#"

SCORE_POINTS = {"correct": 1.0, "wrong": -1.0, "partial": 0.5, "void": 0.0}

QD_LABELS = {
    "qd1_modeling_comprehension":        "Modeling Comprehension",
    "qd2_inference_capability":          "Inference Capability",
    "qd3_mistake_correction_competence": "Mistake Correction Competence",
    "qd4_appropriate_vocabulary_usage":  "Appropriate Vocabulary Usage",
}


def _build_m_i_to_human_key(mistake_types: dict) -> dict:
    """
        Return {'opr_mistake_type_<N>': human_readable_key} for all entries.
    """
    index = {}
    for human_key, entry in mistake_types.items():
        n = entry["type_number"]
        m_i_key = "opr_mistake_type_other" if n == 7 else f"opr_mistake_type_{n}"
        index[m_i_key] = human_key
    return index


# 1.  Prompt construction
def build_prompt(row, task, mistake_types, mistake_types_path, prompt_files, ontology_path):
    """
        Build the inference prompt for one task.
    """
    m_i_index = _build_m_i_to_human_key(mistake_types)
    human_key = m_i_index.get(row["M_i"])
    if human_key is None:
        raise KeyError(
            f"Could not resolve M_i='{row['M_i']}'. "
            f"Known M_i keys: {list(m_i_index.keys())}"
        )
    entry = mistake_types[human_key]

    common = {
        "I":                   row["I"],
        "OA":                  row["OA"],
        "M_i":                 human_key,
        "mistake_description": entry["description"],
        "mistake_example":     entry["example"],
        "generation":          row["generation"],
    }

    if task in ("modeling_comprehension", "inference_capability"):
        enriched = process_sample(
            input_data=common,
            ontology_path=Path(ontology_path),
            mistake_types_path=Path(mistake_types_path),
            verbose=False,
        )
        fields = {**common, **enriched}
    else:
        fields = common

    template = load_template(prompt_files[task])
    return template.format(**fields)


# 2.  Model loading
def load_base_model(base_model_name, first_adapter_path):
    """
        Load the base model once and for all, and attach the first adapter.
        Returns (PeftModel, tokenizer).
    """
    print(f"  Loading base model  : {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(first_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Loading first adapter: {first_adapter_path}")
    model = PeftModel.from_pretrained(base, first_adapter_path,
                                      adapter_name=TASKS[0])
    model.eval()
    return model, tokenizer


def swap_adapter(model, adapter_path, adapter_name):
    """
        Load a new LoRA adapter into the already-loaded PeftModel and activate it.
    """
    print(f"  Loading LoRA adapter: {adapter_path}")
    model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)


# 3.  Prediction on one task
def predict_label(model, tokenizer, prompt, label_token_ids, label_set, max_length=2048):
    """
        Tokenize the prompt, forward-pass, return the label whose token has
        the highest logit at the last token position.
    """
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids      = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(input_ids.device)

    with torch.no_grad():
        outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits = outputs.logits[0, -1, :]

    label_ids    = [label_token_ids[l] for l in label_set]
    label_logits = last_logits[label_ids]
    best_idx     = label_logits.argmax().item()
    return label_set[best_idx]


# 4.  Ontology population + reasoning
def populate_and_reason(ontology_path, predictions, sample_id, sample_row, mistake_type_id, reasoner="hermit"):
    """
        1. Load the schema ontology fresh.
        2. Create Explanation + Mistake + QD individuals.
        3. Type each QD individual with the predicted scored subclass.
        4. Run the OWL reasoner so hasScoringLevel is derived via hasValue.
        5. Read back hasScoringLevel for each QD individual.

        Returns {qd_individual_name: scoring_level_str}.
    """
    onto = get_ontology(f"file://{Path(ontology_path).resolve()}").load()
    ns   = onto.get_namespace(ONTO_NS)

    def cls(local_name):
        c = ns[local_name]
        if c is None:
            raise ValueError(f"Class '{local_name}' not found in ontology namespace.")
        return c

    def prop(local_name):
        for p in onto.object_properties():
            if p.name == local_name:
                return p
        raise ValueError(
            f"Object property '{local_name}' not found in ontology. "
            f"Available properties: {[p.name for p in onto.object_properties()]}"
        )

    prop_explains_mistake  = prop("explainsMistake")
    prop_has_qd_instance   = prop("hasQualityDimensionInstance")
    prop_has_scoring_level = prop("hasScoringLevel")

    # Explanation individual
    expl_ind = cls("Explanation")(f"explanation_{sample_id}", namespace=onto)
    expl_ind.hasText = [sample_row["generation"]]

    # Mistake individual
    mistake_type_class_map = {
        "opr_mistake_type_1":     "OPRMistakeType1_DomainRangeInconsistency",
        "opr_mistake_type_2":     "OPRMistakeType2_DomainRangeMisplacement",
        "opr_mistake_type_3":     "OPRMistakeType3_EquivalenceMisuse",
        "opr_mistake_type_4":     "OPRMistakeType4_LogicalMisunderstanding",
        "opr_mistake_type_5":     "OPRMistakeType5_TrivialSatisfaction",
        "opr_mistake_type_6":     "OPRMistakeType6_WrongClass",
        "opr_mistake_type_other": "OPRMistakeTypeOther",
    }
    mistake_class_name = mistake_type_class_map.get(mistake_type_id, "OPRMistake")
    mistake_ind = cls(mistake_class_name)(f"mistake_{sample_id}", namespace=onto)
    prop_explains_mistake[expl_ind].append(mistake_ind)

    # Quality dimension individuals
    qd_individuals = {}
    for task, qd_name in TASK_TO_QD_INDIVIDUAL.items():
        label_str     = predictions[task]
        subclass_name = LABEL_TO_SUBCLASS.get((task, label_str))
        if subclass_name is None:
            raise ValueError(
                f"No subclass mapping for ({task}, {label_str}). "
                f"Check LABEL_TO_SUBCLASS."
            )
        qd_ind = cls(subclass_name)(f"{qd_name}_{sample_id}", namespace=onto)
        prop_has_qd_instance[expl_ind].append(qd_ind)
        qd_individuals[qd_name] = qd_ind

    # Reasoning
    print(" Running OWL reasoner …")
    try:
        if reasoner == "pellet":
            sync_reasoner_pellet(onto, infer_property_values=True, debug=0)
        else:
            sync_reasoner_hermit(onto, infer_property_values=True, debug=0)
    except Exception as exc:
        print(f"  [WARNING] Reasoner raised an exception: {exc}")
        print("  Falling back to raw predicted labels.")

    # Read inferred scores
    scores = {}
    for qd_name, qd_ind in qd_individuals.items():
        sl_values = prop_has_scoring_level[qd_ind]
        if sl_values:
            scores[qd_name] = sl_values[0].name
        else:
            task = next(t for t, n in TASK_TO_QD_INDIVIDUAL.items() if n == qd_name)
            scores[qd_name] = predictions[task]
            print(f"  [WARNING] hasScoringLevel not inferred for {qd_name}; "
                  f"using predicted label '{predictions[task]}' directly.")

    return scores


# 5.  Build the JSON output
def build_and_save_json(sample_row, scores, output_path):
    """
        Build the output dict and write it to a JSON file.
    """
    qd_details = {}
    total = 0.0
    for qd_name, label in scores.items():
        pts = SCORE_POINTS.get(label, 0.0)
        total += pts
        qd_details[qd_name] = {
            "label":       label,
            "points":      pts,
            "description": QD_LABELS.get(qd_name, qd_name),
        }

    output = {
        "input": {
            "I":          sample_row["I"],
            "OA":         sample_row["OA"],
            "M_i":        sample_row["M_i"],
            "generation": sample_row["generation"],
        },
        "scores": {
            "dimensions":  qd_details,
            "total":       total,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved → {out}")

    return output

def run_pipeline(sample_row, adapter_dir, sample_id="sample", output_json=None, max_length=2048, reasoner="hermit", ontology_path=ONTOLOGY_PATH, mistake_types_path=str(MISTAKE_TYPES_PATH)):
    """
        1. Build prompts
        2. Run the four predictors
        3. Populate the ontology
        4. Infer the scoring for each quality dimension
        5. Interpret, store and display the scores.
    """
    family       = detect_model_family(BASE_MODEL_NAME)
    model_cfg    = MODEL_CONFIGS[family]
    task_config  = model_cfg["task_config"]
    prompt_files = get_prompt_files(family)

    mistake_types = load_mistake_types(Path(mistake_types_path))

    first_adapter_path = str(Path(adapter_dir) / TASKS[0])
    print(f"\n── Loading base model (once for all tasks) ──────────────────")
    model, tokenizer = load_base_model(BASE_MODEL_NAME, first_adapter_path)

    predictions = {}
    for task in TASKS:
        adapter_path = str(Path(adapter_dir) / task)
        print(f"\n── Task: {task} ──────────────────────────────")

        if task != TASKS[0]:
            swap_adapter(model, adapter_path, adapter_name=task)

        label_token_ids = get_label_token_ids(task, tokenizer, task_config)
        label_set       = list(task_config[task]["labels"].values())

        prompt = build_prompt(
            row=sample_row,
            task=task,
            mistake_types=mistake_types,
            mistake_types_path=mistake_types_path,
            prompt_files=prompt_files,
            ontology_path=ontology_path,
        )

        print("  Predicting …")
        label_str = predict_label(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            label_token_ids=label_token_ids,
            label_set=label_set,
            max_length=max_length,
        )
        predictions[task] = label_str
        print(f"  → {label_str}")

    del model
    torch.cuda.empty_cache()

    print("\n── Populating ontology & running reasoner ──")
    scores = populate_and_reason(
        ontology_path=ontology_path,
        predictions=predictions,
        sample_id=sample_id,
        sample_row=sample_row,
        mistake_type_id=sample_row["M_i"],
        reasoner=reasoner,
    )

    if output_json:
        build_and_save_json(sample_row, scores, output_json)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Predict EAS scores for one explanation sample."
    )
    parser.add_argument("--I", required=True)
    parser.add_argument("--OA", required=True)
    parser.add_argument("--M_i", required=True, help="e.g. 'opr_mistake_type_3'")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--adapter_dir", default="trained_adapters/best")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--ontology_path", default=ONTOLOGY_PATH)
    parser.add_argument("--mistake_types_path", default=str(MISTAKE_TYPES_PATH))
    parser.add_argument("--sample_id", default="sample")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--reasoner", choices=["hermit", "pellet"], default="hermit")
    args = parser.parse_args()

    sample_row = {
        "I":          args.I,
        "OA":         args.OA,
        "M_i":        args.M_i,
        "generation": args.generation,
    }

    scores = run_pipeline(
        sample_row=sample_row,
        adapter_dir=args.adapter_dir,
        sample_id=args.sample_id,
        output_json=args.output_json,
        max_length=args.max_length,
        reasoner=args.reasoner,
        ontology_path=args.ontology_path,
        mistake_types_path=args.mistake_types_path,
    )

    total = 0.0
    print("\n══════════════════════════════════")
    print("  EAS SCORES")
    print("══════════════════════════════════")
    for qd_name, label in scores.items():
        pts = SCORE_POINTS.get(label, 0.0)
        total += pts
        print(f"  {QD_LABELS.get(qd_name, qd_name)}: {pts:g} pt ({label})")
    print(f"\n  Total score: {total:g} pts")
    print("══════════════════════════════════")

    return scores


if __name__ == "__main__":
    main()