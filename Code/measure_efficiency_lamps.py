#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Measure efficiency metrics for the LAMPS pipeline components.

This script reports, for each agent in LAMPS:
  - Average latency per invocation (seconds)
  - Throughput (calls per second)
  - Peak memory usage (GB)
  - Average token counts per decision (for LLM based agents)

It is designed to be used with the datasets employed in the LAMPS experiments.
Given a CSV file, it samples a subset of rows and uses:
  - The package name as input to the Fetcher Agent,
  - The file list and code snippet information as input to the Extractor Agent,
  - The setup.py (or equivalent) source code as input to the Classifier Agent,
  - The classifier outputs as input to the Verdict Agent.

Usage example:

  python measure_efficiency_lamps.py \
      --csv data/D1-6000snippets.csv \
      --code-column "Setup.py" \
      --package-column "Package" \
      --max-samples 100

The script assumes that:
  - LLaMA 3 8B Instruct is available as 'meta-llama/Llama3-8B-Instruct'
  - The fine tuned CodeBERT model is available in 'models/codebert-malware-detector'

These identifiers can be customised by editing the constants below.
"""

import argparse
import os
import time
import statistics
from typing import List, Dict, Any

import psutil
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)

# -------------------------------------------------------------------
# Configuration: model identifiers and defaults
# -------------------------------------------------------------------

LLAMA_MODEL_ID = "meta-llama/Llama3-8B-Instruct"
CODEBERT_MODEL_PATH = "models/codebert-malware-detector"  # adjust if needed

DEFAULT_CODE_COLUMN = "Setup.py"       # column that contains code
DEFAULT_PACKAGE_COLUMN = "Package"     # column that contains package name
DEFAULT_FILELIST_COLUMN = "file_list"  # optional, used for Extractor prompt
DEFAULT_MAX_SAMPLES = 50


# -------------------------------------------------------------------
# LLaMA wrapper with token counting
# -------------------------------------------------------------------

class LlamaWrapper:
    """Wrapper around Meta LLaMA 3 for agent style prompts with token counting."""

    def __init__(self, model_id: str = LLAMA_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
        )

    def run(self, prompt: str) -> Dict[str, Any]:
        """Run a single generation and return text, latency, and token counts."""
        enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = enc["input_ids"].shape[-1]

        start = time.time()
        out = self.pipeline(prompt)[0]["generated_text"]
        end = time.time()

        out_enc = self.tokenizer(out, return_tensors="pt")
        total_tokens = out_enc["input_ids"].shape[-1]
        completion_tokens = max(0, total_tokens - prompt_tokens)

        return {
            "output": out.strip(),
            "latency_s": end - start,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }


# -------------------------------------------------------------------
# CodeBERT classifier wrapper with token counting
# -------------------------------------------------------------------

class CodeBERTClassifier:
    """Wrapper around fine tuned CodeBERT for malware classification."""

    def __init__(self, model_path: str = CODEBERT_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def build_prompt(self, code: str) -> str:
        # Align with the prompting template used in the LAMPS classifier agent
        return (
            "You are a security expert. Classify the following Python code as "
            "either malicious or benign.\n\n"
            "Malicious code may include obfuscation, data exfiltration, network "
            "exploitation, or hidden behaviour.\n\n"
            "Code:\n"
            f"{code[:512]}"
        )

    def run(self, code: str) -> Dict[str, Any]:
        """Run a single classification and return label, score, latency, tokens."""
        prompt = self.build_prompt(code)
        enc = self.tokenizer(prompt, return_tensors="pt")
        num_tokens = enc["input_ids"].shape[-1]

        start = time.time()
        result = self.pipeline(prompt)[0]
        end = time.time()

        return {
            "label": result["label"],
            "score": float(result["score"]),
            "latency_s": end - start,
            "tokens": int(num_tokens),
        }


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def bytes_to_gb(b: int) -> float:
    return b / (1024.0 ** 3)


def mean_and_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f} (±0.0000)"
    return f"{statistics.mean(values):.4f} (±{statistics.stdev(values):.4f})"


def safe_truncate(text: str, max_len: int = 512) -> str:
    if text is None:
        return ""
    text = str(text)
    return text[:max_len]


# -------------------------------------------------------------------
# Agent specific measurement routines
# -------------------------------------------------------------------

def measure_fetcher_agent(
    llama: LlamaWrapper,
    packages: List[str],
    process: psutil.Process,
) -> Dict[str, Any]:
    """Measure efficiency for the Fetcher Agent (LLaMA based)."""
    latencies = []
    prompt_tokens = []
    completion_tokens = []
    peak_mem = process.memory_info().rss

    for pkg in packages:
        prompt = (
            "You are the Fetcher Agent in the LAMPS pipeline. Given the name and "
            "metadata of a Python package, identify the URL of its source archive "
            "on PyPI. Respond concisely.\n\n"
            f"Package name: {pkg}"
        )
        before = process.memory_info().rss
        result = llama.run(prompt)
        after = process.memory_info().rss

        latencies.append(result["latency_s"])
        prompt_tokens.append(result["prompt_tokens"])
        completion_tokens.append(result["completion_tokens"])
        peak_mem = max(peak_mem, before, after)

    return {
        "name": "Fetcher Agent (LLaMA)",
        "latencies": latencies,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "peak_mem_bytes": peak_mem,
    }


def measure_extractor_agent(
    llama: LlamaWrapper,
    packages: List[str],
    file_lists: List[str],
    process: psutil.Process,
) -> Dict[str, Any]:
    """Measure efficiency for the Extractor Agent (LLaMA based)."""
    latencies = []
    prompt_tokens = []
    completion_tokens = []
    peak_mem = process.memory_info().rss

    for pkg, files in zip(packages, file_lists):
        files_str = safe_truncate(files, max_len=300)
        prompt = (
            "You are the Extractor Agent in the LAMPS pipeline. Given the contents "
            "of a source archive, identify and list only the Python source files "
            "that should be analysed for potential malicious behaviour. Respond "
            "with a concise explanation and a list of file paths.\n\n"
            f"Package name: {pkg}\n"
            f"Archive file list (truncated): {files_str}"
        )
        before = process.memory_info().rss
        result = llama.run(prompt)
        after = process.memory_info().rss

        latencies.append(result["latency_s"])
        prompt_tokens.append(result["prompt_tokens"])
        completion_tokens.append(result["completion_tokens"])
        peak_mem = max(peak_mem, before, after)

    return {
        "name": "Extractor Agent (LLaMA)",
        "latencies": latencies,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "peak_mem_bytes": peak_mem,
    }


def measure_classifier_agent(
    classifier: CodeBERTClassifier,
    codes: List[str],
    process: psutil.Process,
) -> Dict[str, Any]:
    """Measure efficiency for the Classifier Agent (CodeBERT)."""
    latencies = []
    tokens = []
    peak_mem = process.memory_info().rss

    for code in codes:
        code_snippet = safe_truncate(code, max_len=1000)
        before = process.memory_info().rss
        result = classifier.run(code_snippet)
        after = process.memory_info().rss

        latencies.append(result["latency_s"])
        tokens.append(result["tokens"])
        peak_mem = max(peak_mem, before, after)

    return {
        "name": "Classifier Agent (CodeBERT)",
        "latencies": latencies,
        "tokens": tokens,
        "peak_mem_bytes": peak_mem,
    }


def measure_verdict_agent(
    llama: LlamaWrapper,
    packages: List[str],
    classifier_labels: List[str],
    process: psutil.Process,
) -> Dict[str, Any]:
    """Measure efficiency for the Verdict Agent (LLaMA based)."""
    latencies = []
    prompt_tokens = []
    completion_tokens = []
    peak_mem = process.memory_info().rss

    for pkg, label in zip(packages, classifier_labels):
        prompt = (
            "You are the Verdict Agent in the LAMPS pipeline. Given the per file "
            "classification results for a package, decide whether the package as "
            "a whole should be considered malicious or benign and provide a short "
            "justification.\n\n"
            f"Package: {pkg}\n"
            f"File-level summary (truncated): label={label}"
        )
        before = process.memory_info().rss
        result = llama.run(prompt)
        after = process.memory_info().rss

        latencies.append(result["latency_s"])
        prompt_tokens.append(result["prompt_tokens"])
        completion_tokens.append(result["completion_tokens"])
        peak_mem = max(peak_mem, before, after)

    return {
        "name": "Verdict Agent (LLaMA)",
        "latencies": latencies,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "peak_mem_bytes": peak_mem,
    }


# -------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure efficiency metrics for LAMPS agents using a CSV dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file (e.g., D1 or D2).",
    )
    parser.add_argument(
        "--code-column",
        type=str,
        default=DEFAULT_CODE_COLUMN,
        help=f"Name of the column that contains the code snippet (default: {DEFAULT_CODE_COLUMN}).",
    )
    parser.add_argument(
        "--package-column",
        type=str,
        default=DEFAULT_PACKAGE_COLUMN,
        help=f"Name of the column that contains the package name (default: {DEFAULT_PACKAGE_COLUMN}).",
    )
    parser.add_argument(
        "--filelist-column",
        type=str,
        default=DEFAULT_FILELIST_COLUMN,
        help=f"Optional column with archive file list used to build Extractor prompts (default: {DEFAULT_FILELIST_COLUMN}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of rows to sample from the CSV (default: {DEFAULT_MAX_SAMPLES}).",
    )

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("CSV file is empty.")

    # Sample rows deterministically (first N rows)
    df_sample = df.head(args.max_samples).copy()

    if args.package_column in df_sample.columns:
        packages = df_sample[args.package_column].astype(str).tolist()
    else:
        packages = [f"pkg_{i}" for i in range(len(df_sample))]

    if args.filelist_column in df_sample.columns:
        file_lists = df_sample[args.filelist_column].astype(str).tolist()
    else:
        file_lists = ["" for _ in range(len(df_sample))]

    if args.code_column not in df_sample.columns:
        raise ValueError(
            f"Specified code column '{args.code_column}' not found in CSV. "
            f"Available columns: {list(df_sample.columns)}"
        )
    codes = df_sample[args.code_column].fillna("").astype(str).tolist()

    # Prepare process and models
    process = psutil.Process(os.getpid())

    print("Loading LLaMA model and tokenizer...")
    llama = LlamaWrapper(LLAMA_MODEL_ID)

    print("Loading fine tuned CodeBERT model and tokenizer...")
    codebert = CodeBERTClassifier(CODEBERT_MODEL_PATH)

    # Measure each agent
    print("\nMeasuring Fetcher Agent...")
    fetch_stats = measure_fetcher_agent(llama, packages, process)

    print("Measuring Extractor Agent...")
    extract_stats = measure_extractor_agent(llama, packages, file_lists, process)

    print("Measuring Classifier Agent...")
    classifier_stats = measure_classifier_agent(codebert, codes, process)
    # Use classifier labels as a simple summary input for Verdict Agent
    clf_labels = classifier_stats["tokens"]  # tokens per decision, not ideal but deterministic
    # For Verdict, better to use actual labels; re-run light classification for labels only:
    clf_labels_clean = []
    for code in codes:
        res = codebert.run(safe_truncate(code, 512))
        clf_labels_clean.append(res["label"])

    print("Measuring Verdict Agent...")
    verdict_stats = measure_verdict_agent(llama, packages, clf_labels_clean, process)

    # Collect and print summary
    components = [fetch_stats, extract_stats, classifier_stats, verdict_stats]

    print("\n=== LAMPS Efficiency Summary ===\n")
    header = (
        f"{'Component':40s}  "
        f"{'Avg latency (s)':>18s}  "
        f"{'Throughput (calls/s)':>22s}  "
        f"{'Peak memory (GB)':>18s}  "
        f"{'Tokens / decision':>18s}"
    )
    print(header)
    print("-" * len(header))

    for stats in components:
        name = stats["name"]
        latencies = stats.get("latencies", [])
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0.0
        peak_mem_gb = bytes_to_gb(stats["peak_mem_bytes"])

        # Token counts
        if "tokens" in stats:
            token_values = stats["tokens"]
        elif "prompt_tokens" in stats and "completion_tokens" in stats:
            token_values = [
                pt + ct
                for pt, ct in zip(
                    stats["prompt_tokens"], stats["completion_tokens"]
                )
            ]
        else:
            token_values = []

        avg_tokens = statistics.mean(token_values) if token_values else 0.0

        print(
            f"{name:40s}  "
            f"{avg_latency:18.4f}  "
            f"{throughput:22.2f}  "
            f"{peak_mem_gb:18.3f}  "
            f"{avg_tokens:18.1f}"
        )

    print("\nNote: values are averages over the first "
          f"{min(args.max_samples, n_rows)} rows of {os.path.basename(args.csv)} "
          "under the configuration used for the main experiments.")


if __name__ == "__main__":
    main()