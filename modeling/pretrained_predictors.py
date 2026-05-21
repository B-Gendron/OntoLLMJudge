"""
pretrained_predictors.py
------------------------
Evaluates the pretrained baseline (no LoRA, no fine-tuning) on one EAS task.

Runs a single deterministic inference pass (argmax, no temperature)
and reports per-class and macro F1.

Example usage:
    python pretrained_predictors.py --task modeling_comprehension \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/eas_dataset.csv

    python pretrained_predictors.py --task mistake_correction_competence \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --dataset data/eas_dataset.csv \
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import (
    TASK_CONFIG_LLAMA,
    MODEL_CONFIGS,
    detect_model_family,
    get_label_str,
    get_label_token_ids,
)
from build_prompts import (
    load_mistake_types,
    load_template,
    get_prompt_files,
    MISTAKE_TYPES_PATH,
)
from process_data_sample import process_sample, ONTOLOGY_PATH


# Prompt building (identical to train_predictors.py)
def build_prompt_for_row(row, task, ontology_path, mistake_types, mistake_types_path, prompt_files):
    common_fields = {
        "I":                   row["I"],
        "OA":                  row["OA"],
        "M_i":                 row["M_i"],
        "mistake_description": mistake_types[row["M_i"]]["description"],
        "mistake_example":     mistake_types[row["M_i"]]["example"],
        "generation":          row["generation"],
    }

    if task in ("modeling_comprehension", "inference_capability"):
        enriched = process_sample(
            input_data=common_fields,
            ontology_path=ontology_path,
            mistake_types_path=mistake_types_path,
            verbose=False,
        )
        fields = {**common_fields, **enriched}
    else:
        fields = common_fields

    template = load_template(prompt_files[task])
    return template.format(**fields)


# Dataset
class EASValDataset(Dataset):
    def __init__(self, df, task, tokenizer, mistake_types, max_length,
                 ontology_path, mistake_types_path, task_config, prompt_files):
        self.samples = []
        for _, row in df.iterrows():
            prompt    = build_prompt_for_row(
                row.to_dict(), task, ontology_path,
                mistake_types, mistake_types_path, prompt_files,
            )
            label_str = get_label_str(task, row[task_config[task]["column"]], task_config)
            self.samples.append({"text": prompt, "label_str": label_str})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length):
    texts      = [s["text"]      for s in batch]
    label_strs = [s["label_str"] for s in batch]

    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "label_strs":     label_strs,
    }


def run_once(model, tokenizer, val_loader, label_token_ids, label_set, device):
    """
        Run a single deterministic pass (argmax, no temperature).
        Returns (all_true, all_preds) lists of label strings.
    """
    label_ids_list = [label_token_ids[l] for l in label_set]

    all_true  = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the last token position — val prompts have no appended label,
            # so with left-padding the last token is always the prompt's final token.
            last_logits = outputs.logits[:, -1, :]          # [B, vocab_size]

            label_logits = last_logits[:, label_ids_list]   # [B, n_classes]
            pred_indices = label_logits.argmax(dim=-1).tolist()
            preds        = [label_set[i] for i in pred_indices]

            all_preds.extend(preds)
            all_true.extend(batch["label_strs"])

    return all_true, all_preds


def evaluate(task, model_name, dataset_path, val_split=0.2, batch_size=4, max_length=1024, ontology_path=ONTOLOGY_PATH, mistake_types_path=str(MISTAKE_TYPES_PATH)):
    """
        Run the full evaluation
    """

    # Config
    family       = detect_model_family(model_name)
    model_cfg    = MODEL_CONFIGS[family]
    task_config  = model_cfg["task_config"]
    prompt_files = get_prompt_files(family)

    assert task in task_config, (
        f"Unknown task '{task}'. Choose from: {list(task_config.keys())}"
    )
    print(f"Model family : {family}")
    print(f"Model        : {model_name}")
    print(f"Task         : {task}")

    mistake_types = load_mistake_types(Path(mistake_types_path))

    # Train/val dataset splits
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[task_config[task]["column"]])
    _, val_df = train_test_split(
        df, test_size=val_split, random_state=42,
        stratify=df[task_config[task]["column"]],
    )
    print(f"Val samples  : {len(val_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset class and dataloader
    val_dataset = EASValDataset(
        val_df, task, tokenizer, mistake_types, max_length,
        ontology_path, mistake_types_path, task_config, prompt_files,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    # Model w/o adapters
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False
    device = next(model.parameters()).device

    label_token_ids = get_label_token_ids(task, tokenizer, task_config)
    label_set       = list(task_config[task]["labels"].values())

    print(f"\nLabel token IDs:")
    for l, tid in label_token_ids.items():
        print(f"  '{l}' → {tid} → '{tokenizer.decode([tid])}'")

    # Sigle run (deterministic)
    all_true, all_preds = run_once(
        model, tokenizer, val_loader,
        label_token_ids, label_set, device,
    )

    # Display results
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"PRETRAINED BASELINE RESULTS")
    print(f"Task: {task}  |  Model: {model_name}")
    print(sep)
    print(classification_report(all_true, all_preds, labels=label_set, zero_division=0))

    # Display prediction distribution
    pred_counts = Counter(all_preds)
    true_counts = Counter(all_true)
    n = len(all_preds)
    print(f"  {'Label':<30} {'True %':>8} {'Pred %':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for label_str in label_set:
        true_pct = 100 * true_counts.get(label_str, 0) / n
        pred_pct = 100 * pred_counts.get(label_str, 0) / n
        flag     = " ← collapsed?" if pred_pct > 90 else ""
        print(f"  {label_str:<30} {true_pct:>7.1f}% {pred_pct:>7.1f}%{flag}")
    print(sep)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained baseline (no fine-tuning) on one EAS task. "
                    "Single deterministic pass, argmax decoding."
    )
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG_LLAMA.keys()))
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data used as validation (must match training). Default: 0.2.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--ontology_path", default=ONTOLOGY_PATH)
    parser.add_argument("--mistake_types_path", default=str(MISTAKE_TYPES_PATH))
    args = parser.parse_args()

    evaluate(
        task=args.task,
        model_name=args.model,
        dataset_path=args.dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        ontology_path=args.ontology_path,
        mistake_types_path=args.mistake_types_path,
    )