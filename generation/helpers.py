"""
Shared helpers for config loading, prompt rendering and formatting input/output.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

MISTAKE_TYPES_PATH = Path("../prompts/modeling_mistakes.json")


# Load config
def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and resolve all relative paths against its parent dir."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Store the project root
    cfg["_project_root"] = str(config_path.parent.parent)
    return cfg


def project_path(cfg: dict, *parts: str) -> Path:
    """Build an absolute path from the project root + relative parts."""
    return Path(cfg["_project_root"]).joinpath(*parts)


# Load data
def load_dataset(cfg: dict) -> pd.DataFrame:
    path = project_path(cfg, cfg["data"]["path"])
    df = pd.read_csv(path, index_col=0)
    required = {"OA", "I", "M_i"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.reset_index(drop=True)


# Load mistake-type dict
def load_mistake_types(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Load and render prompt
def load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()

def _strip_llama_scaffold(template: str) -> dict[str, str]:
    """
    Parse a Llama-3 chat template into {'system': ..., 'user': ...}.
    Falls back gracefully if the markers aren't found.
    """
    sys_match = re.search(
        r"<\|start_header_id\|>system<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
        template,
        re.DOTALL,
    )
    user_match = re.search(
        r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
        template,
        re.DOTALL,
    )

    system_text = sys_match.group(1).strip() if sys_match else template.strip()
    user_text   = user_match.group(1).strip() if user_match else ""
    return {"system": system_text, "user": user_text}


def build_messages(
    template_path: Path,
    row: pd.Series,
    eas_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Render a prompt template for one data row and return a messages list
    compatible with the HuggingFace `apply_chat_template` API.

    Substitution variables in the template:
        {OA}   – original axiom
        {I}    – student's intended meaning
        {M_i}  – mistake type
        {mistake_name} - mistake full name from the modeling mistakes dictionnary
        {mistake_description} - mistake description from the modeling mistakes dictionnary
        {mistake_example} - mistake example from the modeling mistakes dictionnary
        {EAS}  – expert assessment sheet (eas-instruct only)

    Variables that require an ontology lookup ({name(M_i)}, {desc(M_i)},
    {example(M_i)}) are left as-is; they are placeholders for a future
    enrichment step.
    """
    raw = load_text(template_path)
    parts = _strip_llama_scaffold(raw)

    # get the mistake names and descriptions
    mistake_types = load_mistake_types(MISTAKE_TYPES_PATH)

    substitutions: dict[str, str] = {
        "OA":  str(row["OA"]),
        "I":   str(row["I"]),
        "M_i": str(row["M_i"]),
        "mistake_name": mistake_types[str(row["M_i"])]["name"],
        "mistake_description": mistake_types[str(row["M_i"])]["description"],
        "mistake_example": mistake_types[str(row["M_i"])]["example"],
    }
    if eas_text is not None:
        substitutions["EAS"] = eas_text

    def _sub(text: str) -> str:
        for key, val in substitutions.items():
            text = text.replace("{" + key + "}", val)
        return text

    system_content = _sub(parts["system"])
    user_content   = _sub(parts["user"])

    messages = [{"role": "system", "content": system_content}]
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def ensure_output_dirs(cfg: dict) -> dict[str, Path]:
    """Create output directories and return a dict of {experiment_name: path}."""
    root = project_path(cfg, cfg["output"]["root"])
    dirs: dict[str, Path] = {}

    if cfg["experiments"]["no_instruct"]:
        p = root / cfg["output"]["no_instruct_dir"]
        p.mkdir(parents=True, exist_ok=True)
        dirs["no_instruct"] = p

    if cfg["experiments"]["eas_instruct"]:
        p = root / cfg["output"]["eas_instruct_dir"]
        p.mkdir(parents=True, exist_ok=True)
        dirs["eas_instruct"] = p

    return dirs


def save_result(record: dict[str, Any], out_dir: Path, cfg: dict) -> None:
    """Append one result record to the JSONL file in out_dir."""
    jsonl_path = out_dir / cfg["output"]["jsonl_filename"]
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_to_csv(out_dir: Path, cfg: dict) -> None:
    """Convert the JSONL results file to a tidy CSV."""
    jsonl_path = out_dir / cfg["output"]["jsonl_filename"]
    if not jsonl_path.exists():
        return
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if records:
        pd.DataFrame(records).to_csv(
            out_dir / cfg["output"]["csv_filename"], index=False
        )