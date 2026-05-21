#!/usr/bin/env python3
"""
run_inference.py
─────────────────────────────────────────────
Runs explanation generation.

Usage
-----
    python run_inference.py                               # use the default config
    python run_inference.py --config path/to/cfg.yaml     # usa a custom config
    python run_inference.py --experiments no_instruct     # run one experiment
    python run_inference.py --experiments eas_instruct    # run one experiment

The script will:
  1. Load the dataset (data/expl_gen_ready_for_prompting.csv)
  2. For each enabled experiment, iterate over all rows, render the prompt,
     call the model, and stream results to outputs/<experiment>/results.jsonl
  3. Convert each JSONL to a tidy CSV at the end
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_messages, ensure_output_dirs, jsonl_to_csv, load_config, load_dataset, load_text, project_path, save_result

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Load model
def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device_map = cfg.get("device", "auto")
    log.info("Loading model (device_map=%s) …", device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    log.info("Model loaded.")
    return model, tokenizer


def build_pipeline(model, tokenizer, cfg: dict):
    gen_cfg = cfg["generation"]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        do_sample=gen_cfg["do_sample"],
        repetition_penalty=gen_cfg["repetition_penalty"],
        return_full_text=False,   # return only the newly generated tokens
    )
    return pipe


#  Generation helper
def generate_explanation(pipe, messages: list[dict], tokenizer) -> str:
    """Apply the chat template and run generation; return the assistant text."""
    # apply_chat_template converts the messages list → a single prompt string
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipe(prompt)
    return outputs[0]["generated_text"].strip()


# Generate explanations
def run_experiment(
    experiment_name: str,
    template_path: Path,
    df,
    pipe,
    tokenizer,
    out_dir: Path,
    cfg: dict,
    eas_text: str | None = None,
) -> None:
    log.info("─" * 60)
    log.info("Experiment: %s  (%d rows)", experiment_name, len(df))
    log.info("Template  : %s", template_path)
    log.info("Output    : %s", out_dir)
    log.info("─" * 60)

    # Clear any previous run's JSONL (idempotent re-runs)
    jsonl_path = out_dir / cfg["output"]["jsonl_filename"]
    if jsonl_path.exists():
        log.warning("Overwriting existing results file: %s", jsonl_path)
        jsonl_path.unlink()

    for idx, row in df.iterrows():
        t0 = time.perf_counter()

        messages = build_messages(template_path, row, eas_text=eas_text)
        explanation = generate_explanation(pipe, messages, tokenizer)

        elapsed = time.perf_counter() - t0
        log.info(
            "[%d/%d] row=%d  M_i='%s'  (%.1fs)",
            idx + 1,
            len(df),
            idx,
            str(row["M_i"])[:60],
            elapsed,
        )

        record = {
            "row_id":      int(idx),
            "experiment":  experiment_name,
            "OA":          str(row["OA"]),
            "I":           str(row["I"]),
            "M_i":         str(row["M_i"]),
            "explanation": explanation,
        }
        save_result(record, out_dir, cfg)

    jsonl_to_csv(out_dir, cfg)
    log.info("Saved results to %s", out_dir)


# Arguments handling
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ontology explanation-generation experiments."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "inference_config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=["no_instruct", "eas_instruct"],
        default=None,
        help=(
            "Which experiments to run. Overrides the config's experiments flags. "
            "If omitted, the config file controls which experiments are enabled."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.experiments is not None:
        cfg["experiments"]["no_instruct"]  = "no_instruct"  in args.experiments
        cfg["experiments"]["eas_instruct"] = "eas_instruct" in args.experiments

    if not any(cfg["experiments"].values()):
        log.error("No experiments enabled. Check your config or --experiments flag.")
        sys.exit(1)

    # set the seed for reproducibility
    import random, numpy as np
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data
    df = load_dataset(cfg)
    log.info("Loaded dataset: %d rows", len(df))

    # load model once for all experiments
    model, tokenizer = load_model_and_tokenizer(cfg)
    pipe = build_pipeline(model, tokenizer, cfg)

    out_dirs = ensure_output_dirs(cfg)

    eas_sheet_path = project_path(cfg, cfg["prompts"]["eas_sheet"])
    eas_text       = load_text(eas_sheet_path)

    #  Experiment 1: no-instruct
    if cfg["experiments"]["no_instruct"]:
        run_experiment(
            experiment_name="no_instruct",
            template_path=project_path(cfg, cfg["prompts"]["no_instruct"]),
            df=df,
            pipe=pipe,
            tokenizer=tokenizer,
            out_dir=out_dirs["no_instruct"],
            cfg=cfg,
            eas_text=None,          # no EAS in the prompt in this case
        )

    #  Experiment 2: eas-instruct
    if cfg["experiments"]["eas_instruct"]:
        run_experiment(
            experiment_name="eas_instruct",
            template_path=project_path(cfg, cfg["prompts"]["eas_instruct"]),
            df=df,
            pipe=pipe,
            tokenizer=tokenizer,
            out_dir=out_dirs["eas_instruct"],
            cfg=cfg,
            eas_text=eas_text,      # the EAS is present in the prompt
        )

    log.info("All experiments done.")


if __name__ == "__main__":
    main()