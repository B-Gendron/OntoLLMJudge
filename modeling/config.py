"""
config.py
---------
General configuration for all model families and tasks.

To add a new model family:
  1. Add a TASK_CONFIG_NEW_MODEL dict below.
  2. Add an entry to MODEL_CONFIGS.
  3. Add an elif branch in detect_model_family().
  4. Create a prompt_templates/new_model/ folder with the 4 template files.
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Task configs — one per model family
# ---------------------------------------------------------------------------

# Llama uses custom label tokens registered into the vocabulary (e.g. CORRECT_ALL).
# Embeddings are initialised from seed words to avoid random/identical initialisations.
TASK_CONFIG_LLAMA = {
    "modeling_comprehension": {
        "column":      "QD1_MODELING_COMPR",
        "labels":      {1: "CORRECT_ALL", 0.5: "CORRECT_SUBSET", 0: "NO_INTERPRETATION", -1: "INCORRECT_INTERPRETATION"},
        "seed_tokens": {"CORRECT_ALL": "correct", "CORRECT_SUBSET": "partial", "NO_INTERPRETATION": "void", "INCORRECT_INTERPRETATION": "wrong"},
    },
    "inference_capability": {
        "column":      "QD2_INF_CAPACITY",
        "labels":      {1: "CORRECT_ALL", 0.5: "CORRECT_SUBSET", 0.0: "NO_IDENTIFICATION", -1: "INCORRECT_IDENTIFICATION"},
        "seed_tokens": {"CORRECT_ALL": "correct", "CORRECT_SUBSET": "partial", "NO_IDENTIFICATION": "void", "INCORRECT_IDENTIFICATION": "wrong"},
    },
    "mistake_correction_competence": {
        "column":      "QD3_MISTAKE_CORR",
        "labels":      {1: "CORRECT_SUGGESTION", 0.0: "NO_SUGGESTION"},  # -1 INCORRECT_SUGGESTION removed (only 1 occurrence)
        "seed_tokens": {"CORRECT_SUGGESTION": "correct", "NO_SUGGESTION": "void"},
    },
    "appropriate_vocabulary_usage": {
        "column":      "QD4_VOCAB_USAGE",
        "labels":      {1: "APPROPRIATE", -1: "INAPPROPRIATE"},
        "seed_tokens": {"APPROPRIATE": "correct", "INAPPROPRIATE": "wrong"},
    },
}

# Qwen uses existing vocabulary words as labels (correct / partial / void / wrong)
# to avoid calling resize_token_embeddings(), which conflicts with accelerate's
# device-dispatch hooks on large MoE models.
TASK_CONFIG_QWEN = {
    "modeling_comprehension": {
        "column":      "QD1_MODELING_COMPR",
        "labels":      {1: "correct", 0.5: "partial", 0: "void", -1: "wrong"},
        "seed_tokens": {"correct": "correct", "partial": "partial", "void": "void", "wrong": "wrong"},
    },
    "inference_capability": {
        "column":      "QD2_INF_CAPACITY",
        "labels":      {1: "correct", 0.5: "partial", 0.0: "void", -1: "wrong"},
        "seed_tokens": {"correct": "correct", "partial": "partial", "void": "void", "wrong": "wrong"},
    },
    "mistake_correction_competence": {
        "column":      "QD3_MISTAKE_CORR",
        "labels":      {1: "correct", 0.0: "void"},
        "seed_tokens": {"correct": "correct", "void": "void"},
    },
    "appropriate_vocabulary_usage": {
        "column":      "QD4_VOCAB_USAGE",
        "labels":      {1: "correct", -1: "wrong"},
        "seed_tokens": {"correct": "correct", "wrong": "wrong"},
    },
}


# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------
# Each entry maps a family name to:
#   task_config          — label/column definitions for that family
#   lora_target_modules  — which attention projections to adapt
#   register_labels      — whether custom label tokens must be added to vocab
MODEL_CONFIGS = {
    "llama": {
        "task_config":         TASK_CONFIG_LLAMA,
        "lora_target_modules": ["q_proj", "v_proj"],  # only q_proj for 8B+
        "register_labels":     True,                  # needs vocab extension
    },
    "qwen": {
        "task_config":         TASK_CONFIG_QWEN,
        "lora_target_modules": ["q_proj", "v_proj"],  # MoE A3B is smaller
        "register_labels":     False,                 # labels already in vocab
    },
}


def detect_model_family(model_name: str) -> str:
    """
    Infers the model family from the HuggingFace model name or local path.
    Add a new elif branch here when supporting additional model families.
    """
    name = model_name.lower()
    if "llama" in name:
        return "llama"
    elif "qwen" in name:
        return "qwen"
    else:
        raise ValueError(
            f"Cannot detect model family from '{model_name}'. "
            f"Supported families: {list(MODEL_CONFIGS.keys())}. "
            f"Add a new branch to detect_model_family() if needed."
        )


# ---------------------------------------------------------------------------
# Label helpers  (all take task_config as an explicit argument so they work
# for any family without relying on a global)
# ---------------------------------------------------------------------------

def get_label_str(task: str, value, task_config: dict) -> str:
    labels = task_config[task]["labels"]
    value  = float(value)
    for k, v in labels.items():
        if float(k) == value:
            return v
    raise ValueError(f"Unknown label value {value} for task '{task}'. Valid: {list(labels.keys())}")


def get_label_value(task: str, label_str: str, task_config: dict) -> float:
    labels = task_config[task]["labels"]
    for k, v in labels.items():
        if v == label_str:
            return float(k)
    raise ValueError(f"Unknown label string '{label_str}' for task '{task}'.")


def get_label_token_ids(task: str, tokenizer, task_config: dict) -> dict:
    """
    Returns {label_str: token_id} for every label in the task.
    For Llama, labels are custom-registered tokens looked up via
    convert_tokens_to_ids(). For Qwen, they are existing vocab words.
    Raises if any label is missing from the vocabulary.
    """
    label_token_ids = {}
    for label_str in task_config[task]["labels"].values():
        token_id = tokenizer.convert_tokens_to_ids(label_str)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Label token '{label_str}' not found in vocabulary. "
                f"For Llama: did register_label_tokens() run? "
                f"For Qwen: seed token may be missing from base vocabulary."
            )
        label_token_ids[label_str] = token_id
    return label_token_ids


def compute_class_weights(task: str, train_df, label_token_ids: dict,
                          device, task_config: dict, vocab_size: int):
    label_set   = list(task_config[task]["labels"].values())
    column      = task_config[task]["column"]
    y           = [get_label_str(task, v, task_config) for v in train_df[column]]
    raw_weights = compute_class_weight("balanced", classes=np.array(label_set), y=y)

    print("\nClass weights (inverse frequency):")
    for label_str, w in zip(label_set, raw_weights):
        print(f"  {label_str}: {w:.4f}  (n={y.count(label_str)})")

    # Use the full vocab size, not max token id + 1
    weight_tensor = torch.zeros(vocab_size, dtype=torch.bfloat16, device=device)
    for label_str, w in zip(label_set, raw_weights):
        weight_tensor[label_token_ids[label_str]] = w

    return weight_tensor

# ---------------------------------------------------------------------------
# Label token registration (Llama only)
# ---------------------------------------------------------------------------

def register_label_tokens(task: str, tokenizer, model,
                           task_config: dict, train_embeddings: bool = False):
    """
    Adds task label strings as atomic vocabulary tokens and initialises their
    input embeddings and lm_head projections from the mean of their seed word's
    BPE vectors. This avoids the near-identical random initialisations that
    HuggingFace's resize_token_embeddings() produces by default.

    Must be called BEFORE get_peft_model() so that the resize happens on the
    unwrapped model and accelerate's dispatch hooks see the final tensor shapes.

    No-op if all label tokens are already in the vocabulary.
    """
    seed_tokens  = task_config[task]["seed_tokens"]
    label_tokens = list(task_config[task]["labels"].values())

    new_tokens = [t for t in label_tokens if t not in tokenizer.get_vocab()]
    if not new_tokens:
        print("All label tokens already in vocabulary, skipping registration.")
        return []

    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    for t in new_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        assert tid != tokenizer.unk_token_id, (
            f"Token '{t}' failed to register — check that add_tokens() persisted."
        )

    new_token_ids = []
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        lm_head    = model.get_output_embeddings()
        for label_str in new_tokens:
            seed_word = seed_tokens[label_str]
            seed_ids  = tokenizer.encode(seed_word, add_special_tokens=False)
            seed_mean = embeddings.weight[seed_ids].mean(dim=0)
            new_id    = tokenizer.convert_tokens_to_ids(label_str)
            embeddings.weight[new_id] = seed_mean
            lm_head.weight[new_id]   = seed_mean
            new_token_ids.append(new_id)
            print(f"  Initialised '{label_str}' from '{seed_word}' (seed ids: {seed_ids})")

    print(f"Registered {len(new_tokens)} new label tokens.")

    if train_embeddings:
        embeddings.weight.requires_grad = True
        lm_head.weight.requires_grad    = True

        def _mask_embedding_grad(grad):
            mask = torch.zeros_like(grad)
            for tid in new_token_ids:
                mask[tid] = 1.0
            return grad * mask

        embeddings.weight.register_hook(_mask_embedding_grad)
        lm_head.weight.register_hook(_mask_embedding_grad)
        print(f"Embeddings+lm_head trainable for token ids: {new_token_ids} (others masked).")

    return new_token_ids