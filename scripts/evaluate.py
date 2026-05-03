"""
scripts/evaluate.py
Evaluate fine-tuned model on test set.
Run after training completes.

Usage:
    python scripts/evaluate.py
"""
import json
import torch
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz

TEST_JSONL   = "datasets/test.jsonl"
ADAPTER_PATH = "./models/qwen-lora-invoice-adapter"
BASE_MODEL   = "Qwen/Qwen2.5-VL-3B-Instruct"


# ── Load model ───────────────────────────────────────────────────────────────
def load_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print("Loading model...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    return model, processor


def predict(messages: list, model, processor) -> dict:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    resp = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    try:
        return json.loads(resp)
    except Exception:
        import re
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        return json.loads(m.group()) if m else {}


# ── Metrics ──────────────────────────────────────────────────────────────────
def numeric_match(pred, gold, tol=0.01) -> bool:
    try:
        return abs(float(pred) - float(gold)) / (float(gold) + 1e-9) <= tol
    except Exception:
        return False


def normalized_em(pred: str, gold: str) -> bool:
    return (pred or "").lower().strip() == (gold or "").lower().strip()


def item_f1(pred_items: list, gold_items: list) -> float:
    if not gold_items:
        return 1.0 if not pred_items else 0.0
    matched, used = 0, set()
    for p in pred_items:
        for i, g in enumerate(gold_items):
            if i in used:
                continue
            if fuzz.token_sort_ratio(p.get("name", ""), g.get("name", "")) >= 70:
                matched += 1
                used.add(i)
                break
    prec = matched / len(pred_items) if pred_items else 0.0
    rec  = matched / len(gold_items)
    return 2 * prec * rec / (prec + rec + 1e-9)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    model, processor = load_model()

    results = {"store": [], "date": [], "total": [], "item_f1": [], "parse_ok": []}

    test_samples = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    print(f"Evaluating on {len(test_samples)} test samples...")

    for i, sample in enumerate(test_samples):
        user_msg = [sample["messages"][0]]   # only user turn
        gold     = json.loads(sample["messages"][1]["content"][0]["text"])

        pred = predict(user_msg, model, processor)
        parse_ok = 1 if pred else 0
        results["parse_ok"].append(parse_ok)

        if pred:
            results["store"].append(int(normalized_em(pred.get("store_name"), gold.get("store_name"))))
            results["date"].append(int(pred.get("date", "") == gold.get("date", "")))
            results["total"].append(int(numeric_match(pred.get("total"), gold.get("total"))))
            results["item_f1"].append(item_f1(pred.get("items", []), gold.get("items", [])))

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(test_samples)} done...")

    print(f"\n{'='*45}")
    print(f"  Evaluation on {len(test_samples)}-invoice test set")
    print(f"{'='*45}")
    print(f"  JSON Parse Rate : {np.mean(results['parse_ok']):.1%}")
    print(f"  Store Name Acc  : {np.mean(results['store']):.1%}")
    print(f"  Date Accuracy   : {np.mean(results['date']):.1%}")
    print(f"  Total Accuracy  : {np.mean(results['total']):.1%}")
    print(f"  Item F1         : {np.mean(results['item_f1']):.3f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
