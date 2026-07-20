# OntoLLMJudge

This repository provides the code for the paper **"OntoLLMJudge: A Framework for Neurosymbolic Evaluation of LLM Generations"**, submitted to the 4th ELMKE workshop @ ISWC 2026 (Bari, Italy).

OntoLLMJudge is a neurosymbolic framework that combines an OWL ontology formalising expert assessment criteria with fine-tuned neural predictors to automatically score LLM-generated outputs. This repository contains the full pipeline: ontology artefacts, relevant-element extraction, predictor training and evaluation, and the explanation-generation experiments.

---

## Repository Structure

```
OntoLLMJudge/
├── prompts/                            # Prompts used for explanation generation
│   ├── eas-instruct-llama.txt
│   ├── no-instruct-llama.txt
│   ├── expert-assessment-sheet.txt
│   └── modeling_mistakes.json          # Mistake-type definitions (used throughout)
│
├── generation/                         # Step 1 — Generate LLM explanations
│   ├── config/inference_config.yaml
│   ├── data/                           # Annotated datasets from student inputs
│   ├── outputs/                        # Generated explanation
│   ├── helpers.py
│   └── run_inference.py                # Generate the explanations from the student inputs
│
└── modeling/                           # Step 2 — Train and run predictors
    ├── ontologies/                     # Ontology artefacts (RDF/TTL)
    ├── prompt_templates/               # Model-specific system prompt templates
    ├── relevant_elements_extractor/    # Pipeline to enrich the ontology with the relevant elements
    │   ├── extract_and_populate_relevant_elements.sh
    │   ├── step1_rdfs_comment_extraction.py
    │   ├── step2_keyword_extraction.py
    │   └── step3_ontology_population.py
    ├── logs/                           # Training logs (best Qwen run + some Llama runs)
    ├── prediction_outputs/             # Scored explanations
    ├── config.py
    ├── build_prompts.py
    ├── train_baseline.py
    ├── train_predictors.py
    ├── train_predictors_5cv.py
    ├── pretrained_predictors.py
    ├── predict.py
    ├── process_data_sample.py
    ├── run_train_baseline.sh           # Train on the first 2 QDs without the ontology knowledge enrichment
    ├── run_training.sh                 # Train on a dataset split
    ├── run_training_5cv.sh             # Train with 5-fold CV
    ├── run_pretrained.sh               # Evaluate pre-trained model
    └── run_predict_example.sh          # Score a single explanation
```

---

## Setup

Clone this repository and install the required dependencies:

```bash
pip install transformers accelerate torch peft pyyaml pandas rdflib rapidfuzz spacy
python -m spacy download en_core_web_sm
```

> Java must be on `PATH` for the HermiT OWL reasoner used during prediction.

---

## Usage

### 1 — Generate explanations

Run inference with `Llama-3.1-8B-Instruct` using two prompt variants (with or without EAS instructions):

```bash
cd generation

# Both experiments (default)
python run_inference.py

# One experiment at a time
python run_inference.py --experiments no_instruct
python run_inference.py --experiments eas_instruct

# Custom config
python run_inference.py --config path/to/my_config.yaml
```

Key settings in `config/inference_config.yaml`:

| Key | Default | Description |
|---|---|---|
| `model.name` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `generation.max_new_tokens` | `256` | Max tokens to generate |
| `generation.temperature` | `0.1` | Sampling temperature |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |

Results are written to `generation/outputs/<experiment>/results.{csv,jsonl}`.

---

### 2 — Enrich the ontology with Relevant Elements

This three-step pipeline extracts OWL-relevant keywords from the ontology's `rdfs:comment` annotations and populates the ontology with `OPRModelingRelevantElement` instances.

```bash
cd modeling/relevant_elements_extractor

# With automatic dependency installation
./extract_and_populate_relevant_elements.sh --install-deps

# With custom paths
./extract_and_populate_relevant_elements.sh \
  --ontology ../ontologies/ExpertAssessmentSchema_UO.rdf \
  --vocab    opr_relevant_elements.ttl \
  --corpus   default_RE/corpus/ \
  --output   default_RE/enriched_onto_outputs/ \
  --enriched default_RE/enriched_onto_outputs/ExpertAssessmentSchema_enriched.rdf
```

Pre-computed outputs are already available under `default_RE/enriched_onto_outputs/`.

---

### 3 — Train a predictor

Edit the `TASK` variable at the top of the script to select a quality dimension (`modeling_comprehension`, `inference_capability`, `mistake_correction_competence`, or `appropriate_vocabulary_usage`), then run:

```bash
cd modeling

# Single train/validation split (VAL_SPLIT=0.2 by default)
bash run_training.sh

# 5-fold cross-validation
bash run_training_5cv.sh
```

Key parameters (set at the top of each script):

| Variable | Default | Description |
|---|---|---|
| `TASK` | `modeling_comprehension` | Quality dimension to train |
| `MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | HuggingFace model ID |
| `DATASET` | `../generation/data/2025_big_2026_small_annotated_data.csv` | Training data |
| `EPOCHS` | `40` | Number of training epochs |
| `LR` | `5e-5` | Learning rate |
| `LORA_R` | `4` | LoRA rank |
| `TRAIN_EMBEDDINGS` | `true` | Whether to update label token embeddings |

Trained LoRA adapters are saved to `modeling/trained_adapters/`.

---

### 4 — Evaluate the pre-trained baseline

```bash
cd modeling
bash run_pretrained.sh
```

Uses the same `TASK`, `MODEL`, and `DATASET` variables as `run_training.sh`. No adapter is loaded; the model is evaluated as-is.

---

### 5 — Score a single explanation

```bash
cd modeling
bash run_predict_example.sh
```

The script embeds a sample input (axiom, intended meaning, mistake type, and LLM-generated explanation) and runs all four predictors, querying the ontology via HermiT to resolve Relevant Elements and Unintended Outcomes. The scored result is written to `prediction_outputs/scored_explanation_<SAMPLE_ID>.json`. An example is available in the repository.

To score your own input, edit the variables `I`, `OA`, `M_I`, and `GENERATION` at the top of `run_predict_example.sh`.

---

## Ontology Artefacts

The `modeling/ontologies/` folder contains successive versions of the Expert Assessment Schema ontology.

All ontologies were validated with [OOPS! (OntOlogy Pitfall Scanner)](https://oops.linkeddata.es/). We addressed the pitfalls flagged by OOPS! and ensure that none of our ontologies presented in the table below have any remaining critical or important issues.

| File | Contents |
|---|---|
| `ExpertAssessmentSchema.rdf` | Core Expert-Assessment Schema formalization |
| `ExpertAssessmentSchema_UO.rdf` | + Unintended Outcomes (used for inference capability) |
| `ExpertAssessmentSchema_UO_RE.rdf` | + Relevant Elements (used for modeling comprehension) |
| `ExpertAssessmentSchema_withDemoIndivs.rdf` | Annotated sample individuals for illustration |