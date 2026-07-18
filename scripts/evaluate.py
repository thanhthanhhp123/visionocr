"""
Evaluate the fine-tuned invoice extraction model on the held-out test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --update-readme
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_JSONL = PROJECT_ROOT / "datasets/test.jsonl"
ADAPTER_PATH = PROJECT_ROOT / "models/qwen-lora-invoice-adapter-v2"
LOCAL_BASE_MODEL = PROJECT_ROOT.parent / "models/Qwen2.5-VL-3B-Instruct"
BASE_MODEL = os.getenv(
    "BASE_MODEL_ID",
    str(LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.exists() else "Qwen/Qwen2.5-VL-3B-Instruct"),
)
RESULTS_PATH = PROJECT_ROOT / "results_finetuned_v2.json"
README_PATH = PROJECT_ROOT / "README.md"

PROMPT_TEMPLATE = (
    "Trích xuất thông tin hóa đơn Việt Nam từ ảnh.\n"
    "{ocr_hint}"
    "Trả về JSON với format:\n"
    '{{"store_name":"","date":"YYYY-MM-DD","total":0,"discount":0,'
    '"items":[{{"name":"","unit_price":0,"quantity":0,"total_price":0}}]}}'
)


def load_model():
    from peft import PeftModel
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

    print(f"Loading base model: {BASE_MODEL}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    return model, processor


def messages_and_gold(sample: dict) -> tuple[list[dict], dict]:
    if "messages" in sample:
        return [sample["messages"][0]], json.loads(sample["messages"][1]["content"][0]["text"])

    if "image" not in sample or "label" not in sample:
        raise KeyError("Expected sample to contain either 'messages' or both 'image' and 'label'")

    ocr_hint = ""
    if sample.get("ocr_text"):
        ocr_hint = f"OCR text tham khảo:\n{sample['ocr_text']}\n\n"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resolve_image_path(sample["image"])},
                {"type": "text", "text": PROMPT_TEMPLATE.format(ocr_hint=ocr_hint)},
            ],
        }
    ]
    return messages, sample["label"]


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return str(path)

    local_path = PROJECT_ROOT / "datasets/images" / path.name
    if local_path.exists():
        return str(local_path)

    return str(path)


def predict(messages: list[dict], model, processor) -> tuple[dict, float, str]:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    latency = time.perf_counter() - start

    response = processor.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_json(response), latency, response


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        return {}


def normalize_text(value) -> str:
    return str(value or "").lower().strip()


def normalize_date(value) -> str:
    value = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return value


def numeric_match(pred, gold, tol=0.01) -> bool:
    try:
        gold_value = float(gold)
        return abs(float(pred) - gold_value) / (abs(gold_value) + 1e-9) <= tol
    except Exception:
        return False


def item_f1(pred_items: list, gold_items: list) -> float:
    if not gold_items:
        return 1.0 if not pred_items else 0.0

    matched, used = 0, set()
    for pred_item in pred_items or []:
        for index, gold_item in enumerate(gold_items):
            if index in used:
                continue
            if fuzz.token_sort_ratio(pred_item.get("name", ""), gold_item.get("name", "")) >= 70:
                matched += 1
                used.add(index)
                break

    precision = matched / len(pred_items) if pred_items else 0.0
    recall = matched / len(gold_items)
    return 2 * precision * recall / (precision + recall + 1e-9)


def run_eval(model, processor, label: str = "fine_tuned_v2") -> dict:
    samples = [json.loads(line) for line in open(TEST_JSONL, encoding="utf-8")]
    results = {"store": [], "date": [], "total": [], "item_f1": [], "parse_ok": [], "latency": []}
    predictions = []

    print(f"Evaluating {len(samples)} samples [{label}]...")
    for index, sample in enumerate(samples, start=1):
        messages, gold = messages_and_gold(sample)
        pred, latency, raw_response = predict(messages, model, processor)
        parse_ok = bool(pred)

        results["parse_ok"].append(int(parse_ok))
        results["latency"].append(latency)
        results["store"].append(int(parse_ok and normalize_text(pred.get("store_name")) == normalize_text(gold.get("store_name"))))
        results["date"].append(int(parse_ok and normalize_date(pred.get("date")) == normalize_date(gold.get("date"))))
        results["total"].append(int(parse_ok and numeric_match(pred.get("total"), gold.get("total"))))
        results["item_f1"].append(item_f1(pred.get("items", []), gold.get("items", [])) if parse_ok else 0.0)
        predictions.append({"index": index - 1, "gold": gold, "pred": pred, "raw_response": raw_response, "latency_sec": latency})

        if index % 5 == 0 or index == len(samples):
            print(f"  {index}/{len(samples)} done...")

    scores = {
        "label": label,
        "samples": len(samples),
        "store_name_accuracy": float(np.mean(results["store"])),
        "date_accuracy": float(np.mean(results["date"])),
        "total_amount_accuracy": float(np.mean(results["total"])),
        "item_name_f1": float(np.mean(results["item_f1"])),
        "json_parse_rate": float(np.mean(results["parse_ok"])),
        "avg_latency_sec": float(np.mean(results["latency"])),
        "latency_label": "HF + LoRA",
    }
    return {"scores": scores, "predictions": predictions}


def print_scores(scores: dict):
    print("\n=== EVALUATION RESULTS ===")
    print(f"Samples              : {scores['samples']}")
    print(f"Store Name Accuracy  : {scores['store_name_accuracy']:.1%}")
    print(f"Date Accuracy        : {scores['date_accuracy']:.1%}")
    print(f"Total Amount Accuracy: {scores['total_amount_accuracy']:.1%}")
    print(f"Item Name F1         : {scores['item_name_f1']:.3f}")
    print(f"JSON Parse Rate      : {scores['json_parse_rate']:.1%}")
    print(f"Avg Latency          : {scores['avg_latency_sec']:.2f}s ({scores['latency_label']})")


def update_readme(scores: dict):
    readme = README_PATH.read_text(encoding="utf-8")
    replacement = f"""## Evaluation Results

Results on {scores['samples']}-sample held-out test set (Vietnamese retail invoices):

| Metric | Score |
|---|---|
| Store Name Accuracy | {scores['store_name_accuracy']:.1%} |
| Date Accuracy | {scores['date_accuracy']:.1%} |
| Total Amount Accuracy | {scores['total_amount_accuracy']:.1%} |
| Item Name F1 | {scores['item_name_f1']:.3f} |
| JSON Parse Rate | {scores['json_parse_rate']:.1%} |
| Avg Latency (HF + LoRA) | {scores['avg_latency_sec']:.2f}s |
| Avg Latency (AWQ + vLLM) | — |

*AWQ + vLLM latency will be updated after merge and quantization.*
"""
    updated = re.sub(
        r"## Evaluation Results\n.*?\n---",
        replacement + "\n---",
        readme,
        flags=re.DOTALL,
    )
    README_PATH.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-readme", action="store_true", help="Update README.md with fine-tuned metrics.")
    args = parser.parse_args()

    model, processor = load_model()
    output = run_eval(model, processor)
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_scores(output["scores"])
    print(f"\nSaved to {RESULTS_PATH}")

    if args.update_readme:
        update_readme(output["scores"])
        print(f"Updated {README_PATH}")


if __name__ == "__main__":
    main()
