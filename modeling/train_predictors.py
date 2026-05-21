"""
train_predictors.py
-------------------
Fine-tunes a causal LM with LoRA on one EAS evaluation task.
Supports Llama and Qwen model families out of the box; see config.py to add more.

Example usage:
    python train_predictors.py --task modeling_comprehension \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/eas_dataset.csv

    python train_predictors.py --task modeling_comprehension \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --dataset data/eas_dataset.csv
"""

import os

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from config import (
    TASK_CONFIG_LLAMA,
    MODEL_CONFIGS,
    detect_model_family,
    get_label_str,
    get_label_token_ids,
    register_label_tokens,
)
from build_prompts import (
    load_mistake_types,
    load_template,
    get_prompt_files,
    MISTAKE_TYPES_PATH,
)
from process_data_sample import process_sample, ONTOLOGY_PATH


# Class weights

def compute_class_weights_local(task, train_df, label_token_ids, label_set, device, vocab_size, weight_power=2.0):
    """
        Build a [vocab_size] weight tensor for F.cross_entropy.

        weight[token_id] = (total / (n_classes * count)) ** weight_power

        weight_power=1.0 → standard inverse frequency
        weight_power=2.0 → squared inverse frequency (stronger push toward minority)

        All token positions that are not label tokens keep weight 1.0 so that
        the -100 masked positions in the sequence loss are unaffected.
    """
    from config import get_label_str as _get_label_str
    label_counts = Counter(
        _get_label_str(task, v, task_config_ref)
        for v in train_df[task_config_ref[task]["column"]]
    )
    total    = sum(label_counts.values())
    n_classes = len(label_set)

    weights = torch.ones(vocab_size, dtype=torch.bfloat16, device=device)
    for label_str in label_set:
        count    = label_counts.get(label_str, 1)
        w        = (total / (n_classes * count)) ** weight_power
        token_id = label_token_ids[label_str]
        weights[token_id] = w

    print(f"Class weights (power={weight_power}):")
    for label_str in label_set:
        count = label_counts.get(label_str, 0)
        print(f"  {label_str}: {weights[label_token_ids[label_str]]:.4f}  (n={count})")

    return weights


# Prompt building

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
        fields = common_fields  # ← no merge needed, avoids duplicate keys

    template = load_template(prompt_files[task])
    return template.format(**fields)


# Dataset class
class EASDataset(Dataset):
    def __init__(self, df, task, tokenizer, mistake_types, max_length, ontology_path, mistake_types_path, task_config, prompt_files, training=True):
        self.samples = []

        for _, row in df.iterrows():
            prompt    = build_prompt_for_row(
                row.to_dict(), task, ontology_path,
                mistake_types, mistake_types_path, prompt_files,
            )
            label_str = get_label_str(task, row[task_config[task]["column"]], task_config)

            # Templates already end with the assistant turn opener, so we just
            # append the label for training or leave the prompt as-is for inference.
            text = f"{prompt}{label_str}" if training else prompt

            self.samples.append({
                "text":        text,
                "label_str":   label_str,
                "label_value": float(row[task_config[task]["column"]]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length, training=True):
    texts        = [s["text"]        for s in batch]
    label_strs   = [s["label_str"]   for s in batch]
    label_values = [s["label_value"] for s in batch]

    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels_tensor   = None
    label_positions = None
    if training:
        labels_tensor   = encoded["input_ids"].clone()
        label_positions = []
        for i, label_str in enumerate(label_strs):
            label_id  = tokenizer.convert_tokens_to_ids(label_str)
            seq       = labels_tensor[i].tolist()
            pos = None
            for j in range(len(seq) - 1, -1, -1):
                if seq[j] == label_id:
                    pos = j
                    break
            if pos is not None:
                labels_tensor[i, :pos] = -100   # mask everything before the label token
            else:
                labels_tensor[i, :] = -100      # fallback: mask everything
                pos_list = encoded["attention_mask"][i].nonzero(as_tuple=True)[0]
                pos = pos_list[-1].item()
            label_positions.append(pos)
        label_positions = torch.tensor(label_positions, dtype=torch.long)

    return {
        "input_ids":       encoded["input_ids"],
        "attention_mask":  encoded["attention_mask"],
        "labels":          labels_tensor,            # [B, seq_len] for sequence loss
        "label_positions": label_positions,          # [B] for focused train eval
        "label_strs":      label_strs,
        "label_values":    torch.tensor(label_values, dtype=torch.float),
    }



# Validation loop with sanity checks
def validate_with_diagnostics(epoch, task, model, tokenizer, val_loader, label_token_ids, label_set, device, n_token_samples=3):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"DIAGNOSTIC — Epoch {epoch} | Task: {task}")
    print(sep)

    print("\n[1] Label token ID check")
    all_ids = list(label_token_ids.values())
    for label_str, token_id in label_token_ids.items():
        print(f"  '{label_str}' → token_id={token_id} → decoded='{tokenizer.decode([token_id])}'")
    if len(set(all_ids)) == len(all_ids):
        print("  ✓ All label token IDs are unique.")
    else:
        print("  ✗ WARNING: Duplicate token IDs detected! Labels may collide during argmax.")

    label_ids_list    = [label_token_ids[l] for l in label_set]
    all_preds         = []
    all_true          = []
    last_token_logged = 0

    print(f"\n[2] Last-token identity (first {n_token_samples} val samples)")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Diag-val epoch {epoch}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs      = model(input_ids=input_ids, attention_mask=attention_mask)
            last_logits  = outputs.logits[:, -1, :]
            label_logits = last_logits[:, label_ids_list]
            pred_indices = label_logits.argmax(dim=-1).tolist()
            preds        = [label_set[i] for i in pred_indices]

            all_preds.extend(preds)
            all_true.extend(batch["label_strs"])

            for i in range(input_ids.size(0)):
                if last_token_logged >= n_token_samples:
                    break
                non_pad        = attention_mask[i].nonzero(as_tuple=True)[0]
                last_real_pos  = non_pad[-1].item()
                last_token_id  = input_ids[i, last_real_pos].item()
                final_token_id = input_ids[i, -1].item()
                pad_id         = tokenizer.pad_token_id
                print(f"  Sample {last_token_logged}: "
                      f"last real='{tokenizer.decode([last_token_id])}' (id={last_token_id}) "
                      f"{'✓' if last_token_id != pad_id else '✗ PAD'} | "
                      f"pos[-1]='{tokenizer.decode([final_token_id])}' (id={final_token_id}) "
                      f"{'✓' if final_token_id != pad_id else '✗ PAD'}")
                last_token_logged += 1

            if batch_idx == 0:
                print(f"\n[3] Logit spread over label tokens (first batch)")
                ll = label_logits.float()
                for j, label_str in enumerate(label_set):
                    col = ll[:, j]
                    print(f"  '{label_str}': mean={col.mean():.3f}  std={col.std():.3f}  "
                          f"min={col.min():.3f}  max={col.max():.3f}")
                spread = ll.max(dim=-1).values - ll.min(dim=-1).values
                print(f"  Max-min spread: mean={spread.mean():.3f}  "
                      f"min={spread.min():.3f}  max={spread.max():.3f}")
                print("  (Near 0 → not differentiating labels; large → confident)")

    print(f"\n[4] Prediction vs. true label distribution")
    pred_counts = Counter(all_preds)
    true_counts = Counter(all_true)
    n = len(all_preds)
    print(f"  {'Label':<30} {'True %':>8} {'Pred %':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for label_str in label_set:
        true_pct = 100 * true_counts.get(label_str, 0) / n
        pred_pct = 100 * pred_counts.get(label_str, 0) / n
        flag = " ← collapsed?" if pred_pct > 90 else ""
        print(f"  {label_str:<30} {true_pct:>7.1f}% {pred_pct:>7.1f}%{flag}")

    print(f"\n[Classification report — Epoch {epoch}]")
    report = classification_report(all_true, all_preds, labels=label_set, zero_division=0, output_dict=True)
    print(classification_report(all_true, all_preds, labels=label_set, zero_division=0))
    print(sep)
    return report["macro avg"]["f1-score"]


# Main training
def train(task, model_name, dataset_path, output_dir, epochs=3, batch_size=4, lr=2e-4, max_length=2048, lora_r=16, lora_alpha=32, lora_dropout=0.05, val_split=0.1, train_embeddings=False, weight_power=2.0, ontology_path=ONTOLOGY_PATH, mistake_types_path=str(MISTAKE_TYPES_PATH)):

    # Setup config
    family      = detect_model_family(model_name)
    model_cfg   = MODEL_CONFIGS[family]
    task_config = model_cfg["task_config"]
    lora_targets       = model_cfg["lora_target_modules"]
    needs_registration = model_cfg["register_labels"]
    prompt_files       = get_prompt_files(family)

    global task_config_ref
    task_config_ref = task_config

    assert task in task_config, (
        f"Unknown task '{task}'. Choose from: {list(task_config.keys())}"
    )
    print(f"Model family : {family}")
    print(f"Task         : {task}")

    output_path = Path(output_dir) / task
    output_path.mkdir(parents=True, exist_ok=True)

    mistake_types = load_mistake_types(Path(mistake_types_path))

    # Train/val dataset splits
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[task_config[task]["column"]])
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=25,
        stratify=df[task_config[task]["column"]],
    )
    print(f"#Samples — train: {len(train_df)} | val: {len(val_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset class and dataloader
    dataset_kwargs = dict(
        task=task, tokenizer=tokenizer, mistake_types=mistake_types,
        max_length=max_length, ontology_path=ontology_path,
        mistake_types_path=mistake_types_path,
        task_config=task_config, prompt_files=prompt_files,
    )
    train_dataset = EASDataset(train_df, training=True,  **dataset_kwargs)
    val_dataset   = EASDataset(val_df,   training=False, **dataset_kwargs)

    label_col = task_config[task]["column"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length, training=True),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length, training=False),
    )

    # Model with adapters
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    if needs_registration:
        register_label_tokens(task, tokenizer, model, task_config, train_embeddings)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    device          = next(model.parameters()).device
    label_token_ids = get_label_token_ids(task, tokenizer, task_config)
    label_set       = list(task_config[task]["labels"].values())
    label_ids_list  = [label_token_ids[l] for l in label_set]

    vocab_size          = model.config.vocab_size
    class_weight_tensor = compute_class_weights_local(
        task=task,
        train_df=train_df,
        label_token_ids=label_token_ids,
        label_set=label_set,
        device=device,
        vocab_size=vocab_size,
        weight_power=weight_power,
    )

    # Optimizer and scheduler
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    best_macro_f1 = -1.0
    best_epoch    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        train_preds = []
        train_true  = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)
            label_positions = batch["label_positions"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Loss computation
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                weight=class_weight_tensor,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Train eval: gather logits at the exact label token position
            with torch.no_grad():
                batch_size_actual = input_ids.size(0)
                pos_idx = label_positions.unsqueeze(-1).unsqueeze(-1).expand(
                    batch_size_actual, 1, outputs.logits.size(-1)
                )
                label_pos_logits = outputs.logits.gather(1, pos_idx).squeeze(1)
                focused_logits   = label_pos_logits[:, label_ids_list]
                pred_indices     = focused_logits.argmax(dim=-1).tolist()
                train_preds.extend([label_set[i] for i in pred_indices])
                train_true.extend(batch["label_strs"])

        print(f"\nEpoch {epoch} — avg train loss: {total_loss / len(train_loader):.4f}")
        print(f"\n[Train classification report — Epoch {epoch}]")
        print(classification_report(train_true, train_preds, labels=label_set, zero_division=0))

        val_macro_f1 = validate_with_diagnostics(
            epoch=epoch,
            task=task,
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            label_token_ids=label_token_ids,
            label_set=label_set,
            device=device,
        )

        # Save best checkpoint (overwrites previous best adapters)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch    = epoch
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  ✓ Best checkpoint updated — epoch {best_epoch}, macro F1={best_macro_f1:.4f}")
        else:
            print(f"  No improvement (best: epoch {best_epoch}, macro F1={best_macro_f1:.4f})")

    print(f"\nTraining complete. Best checkpoint: epoch {best_epoch}, macro F1={best_macro_f1:.4f}")
    print(f"LoRA adapters saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA on one EAS evaluation task. Model family (Llama / Qwen) is detected automatically from --model.")
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG_LLAMA.keys()))
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--output_dir", default="adapters")
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int,   default=2048)
    parser.add_argument("--lora_r", type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_power", type=float, default=2.0, help="Exponent for inverse-frequency class weights. 1.0=standard, 2.0=squared (stronger minority boost).")
    parser.add_argument("--train_embeddings", action="store_true", default=False, help="Train newly added label token embeddings (Llama only).")
    parser.add_argument("--ontology_path", default=ONTOLOGY_PATH)
    parser.add_argument("--mistake_types_path", default=str(MISTAKE_TYPES_PATH))
    args = parser.parse_args()

    train(
        task=args.task,
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        val_split=args.val_split,
        train_embeddings=args.train_embeddings,
        weight_power=args.weight_power,
        ontology_path=args.ontology_path,
        mistake_types_path=args.mistake_types_path,
    )