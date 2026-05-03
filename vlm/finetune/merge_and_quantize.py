"""
Phase 2 — Post-training: Merge LoRA adapter then quantize with AWQ.

Steps:
    1. Merge LoRA weights into base model
    2. Quantize merged model to INT4 (AWQ)
    3. Save to models/qwen-awq-invoice/

Usage:
    python vlm/finetune/merge_and_quantize.py

Requirements:
    pip install autoawq autoawq-kernels
"""
import torch
from pathlib import Path

BASE_MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH   = "./models/qwen-lora-invoice-adapter"
MERGED_PATH    = "./models/qwen-merged"
AWQ_OUT_PATH   = "./models/qwen-awq-invoice"
CALIB_IMAGES   = "./datasets/images"    # used for AWQ calibration


# ── Step 1: Merge LoRA ───────────────────────────────────────────────────────
def merge_lora():
    print("Step 1: Merging LoRA adapter into base model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # merge on CPU to avoid VRAM spike
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_PATH)

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    processor.save_pretrained(MERGED_PATH)
    print(f"Merged model saved to {MERGED_PATH}")


# ── Step 2: AWQ Quantization ─────────────────────────────────────────────────
def quantize_awq():
    print("Step 2: AWQ INT4 quantization (~30 min on RTX 3090)...")
    from awq import AutoAWQForCausalLM
    from transformers import AutoProcessor
    import json

    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH, low_cpu_mem_usage=True, use_cache=False
    )
    processor = AutoProcessor.from_pretrained(MERGED_PATH)

    # Build calibration data from invoice images (no labels needed)
    calib_messages = []
    image_files = list(Path(CALIB_IMAGES).glob("*.jpg"))[:128]
    if not image_files:
        image_files = list(Path(CALIB_IMAGES).glob("*.png"))[:128]

    for img in image_files:
        calib_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img)},
                {"type": "text",  "text": "Trích xuất thông tin hóa đơn."},
            ],
        }])

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(processor.tokenizer, quant_config=quant_config, calib_data=calib_messages)
    model.save_quantized(AWQ_OUT_PATH)
    processor.save_pretrained(AWQ_OUT_PATH)
    print(f"AWQ model saved to {AWQ_OUT_PATH}")


if __name__ == "__main__":
    merge_lora()
    quantize_awq()
    print("\nDone! Load with vLLM:")
    print(f'  LLM(model="{AWQ_OUT_PATH}", quantization="awq")')
