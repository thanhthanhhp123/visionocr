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
import argparse
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_base_candidates = [
    PROJECT_ROOT.parent / "models/Qwen2.5-VL-3B-Instruct",
    PROJECT_ROOT / "models/Qwen2.5-VL-3B-Instruct",
]
BASE_MODEL_ID = os.getenv(
    "BASE_MODEL_ID",
    next((str(path) for path in _base_candidates if path.exists()), "Qwen/Qwen2.5-VL-3B-Instruct"),
)
ADAPTER_PATH = Path(os.getenv("LORA_ADAPTER_PATH", PROJECT_ROOT / "models/qwen-lora-invoice-adapter-v2"))
MERGED_PATH = Path(os.getenv("MERGED_MODEL_PATH", PROJECT_ROOT / "models/qwen-merged-invoice-v2"))
AWQ_OUT_PATH = Path(os.getenv("AWQ_MODEL_PATH", PROJECT_ROOT / "models/qwen-awq-invoice-v2"))
CALIB_IMAGES = PROJECT_ROOT / "datasets/images"


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
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_PATH)

    processor = AutoProcessor.from_pretrained(str(ADAPTER_PATH))
    processor.save_pretrained(MERGED_PATH)
    print(f"Merged model saved to {MERGED_PATH}")


# ── Step 2: AWQ Quantization ─────────────────────────────────────────────────
def quantize_awq():
    print("Step 2: AWQ INT4 quantization (~30 min on RTX 3090)...")
    from transformers import AutoProcessor, activations

    # AutoAWQ 0.2.x imports this activation removed by Transformers 4.57.
    # Qwen2.5-VL uses SiLU, so an equivalent GELU fallback preserves the
    # importer compatibility without changing the model's active layers.
    if not hasattr(activations, "PytorchGELUTanh"):
        activations.PytorchGELUTanh = activations.GELUActivation

    from awq import AutoAWQForCausalLM
    from awq.models.qwen2_5_vl import Qwen2_5_VLAWQForCausalLM

    # AutoAWQ's Qwen2.5-VL adapter targets an older transformers layout where
    # decoder layers hung directly off `model.model`. The installed
    # transformers version nests them under `model.model.language_model`
    # instead, so `get_model_layers`/`move_embed` raise AttributeError
    # without this patch.
    def _get_model_layers(model):
        return model.model.language_model.layers

    def _move_embed(model, device):
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
        model.visual = model.visual.to(device)
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to(device)

    Qwen2_5_VLAWQForCausalLM.get_model_layers = staticmethod(_get_model_layers)
    Qwen2_5_VLAWQForCausalLM.move_embed = staticmethod(_move_embed)

    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH, low_cpu_mem_usage=True, use_cache=False
    )
    processor = AutoProcessor.from_pretrained(MERGED_PATH)

    # Build calibration data from invoice images (no labels needed)
    calib_messages = []
    image_files = list(CALIB_IMAGES.glob("*.jpg"))[:128]
    if not image_files:
        image_files = list(CALIB_IMAGES.glob("*.png"))[:128]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true", help="Merge LoRA without AWQ quantization.")
    parser.add_argument("--quantize-only", action="store_true", help="Quantize an existing merged model.")
    args = parser.parse_args()
    if args.merge_only and args.quantize_only:
        parser.error("Choose only one of --merge-only or --quantize-only")
    if not args.quantize_only:
        merge_lora()
    if not args.merge_only:
        quantize_awq()
        print("\nDone! Load with vLLM:")
        print(f'  LLM(model="{AWQ_OUT_PATH}", quantization="awq")')
