# scripts/evaluate_zeroshot.py
# Dùng base model KHÔNG có LoRA adapter
import json, torch, numpy as np
from rapidfuzz import fuzz
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
TEST_JSONL = "datasets/test.jsonl"

def load_base_model():
    print("Loading BASE model (no LoRA)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor

def predict(messages, model, processor):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:    return json.loads(resp)
    except:
        import re
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        return json.loads(m.group()) if m else {}

def numeric_match(p, g, tol=0.01):
    try:    return abs(float(p) - float(g)) / (float(g) + 1e-9) <= tol
    except: return False

def item_f1(pred_items, gold_items):
    if not gold_items: return 1.0 if not pred_items else 0.0
    matched, used = 0, set()
    for p in pred_items:
        for i, g in enumerate(gold_items):
            if i in used: continue
            if fuzz.token_sort_ratio(p.get("name",""), g.get("name","")) >= 70:
                matched += 1; used.add(i); break
    prec = matched / len(pred_items) if pred_items else 0.0
    rec  = matched / len(gold_items)
    return 2 * prec * rec / (prec + rec + 1e-9)

def run_eval(model, processor, label="BASE"):
    results = {"store":[], "date":[], "total":[], "item_f1":[], "parse_ok":[]}
    samples = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    print(f"Evaluating {len(samples)} samples [{label}]...")

    for i, sample in enumerate(samples):
        user_msg = [sample["messages"][0]]
        gold     = json.loads(sample["messages"][1]["content"][0]["text"])
        pred     = predict(user_msg, model, processor)

        results["parse_ok"].append(1 if pred else 0)
        if pred:
            results["store"].append(int(
                (pred.get("store_name","") or "").lower().strip() ==
                (gold.get("store_name","") or "").lower().strip()
            ))
            results["date"].append(int(pred.get("date","") == gold.get("date","")))
            results["total"].append(int(numeric_match(pred.get("total"), gold.get("total"))))
            results["item_f1"].append(item_f1(pred.get("items",[]), gold.get("items",[])))

        if (i+1) % 5 == 0: print(f"  {i+1}/{len(samples)}...")

    scores = {k: round(float(np.mean(v)), 4) for k, v in results.items() if v}
    return scores

if __name__ == "__main__":
    model, processor = load_base_model()
    scores = run_eval(model, processor, label="ZERO-SHOT")

    # Lưu kết quả ra file để so sánh sau
    out = {"label": "zero_shot", "scores": scores}
    with open("results_zeroshot.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== ZERO-SHOT RESULTS ===")
    for k, v in scores.items():
        print(f"  {k:15s}: {v:.1%}" if v < 2 else f"  {k:15s}: {v:.3f}")
    print(f"\nSaved to results_zeroshot.json")