"""
Evaluate the base Qwen2.5-VL model without the LoRA adapter.

Usage:
    python scripts/evaluate_zeroshot.py
"""
from __future__ import annotations

import json
import os
import argparse

import torch

from evaluate import BASE_MODEL, PROJECT_ROOT, print_scores, run_eval

RESULTS_PATH = PROJECT_ROOT / "results_zeroshot_v2.json"


def load_base_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32
    use_4bit = use_cuda and os.getenv("EVAL_4BIT", "1") != "0"

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model without LoRA: {BASE_MODEL}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    model, processor = load_base_model()
    output = run_eval(model, processor, label="zero_shot_v2")
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_scores(output["scores"])
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
