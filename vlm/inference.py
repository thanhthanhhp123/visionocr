"""
vLLM inference engine for invoice extraction.
Loads the fine-tuned Qwen2.5-VL (AWQ quantized after Phase 2).
"""
from __future__ import annotations
import os, json, re

MODEL_PATH = os.getenv("AWQ_MODEL_PATH", "./models/qwen-awq-invoice")
FALLBACK_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")

_engine = None
_processor = None


def _get_engine():
    global _engine, _processor
    if _engine is None:
        from vllm import LLM
        from transformers import AutoProcessor

        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
        quant = "awq" if "awq" in model_path.lower() else None

        _engine = LLM(
            model=model_path,
            quantization=quant,
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            limit_mm_per_prompt={"image": 1},
        )
        _processor = AutoProcessor.from_pretrained(model_path)
    return _engine, _processor


SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất thông tin hóa đơn Việt Nam. "
    "Chỉ trả về JSON, không giải thích thêm."
)

OUTPUT_FORMAT = (
    '{"store_name":"","date":"YYYY-MM-DD","total":0,"discount":0,'
    '"items":[{"name":"","unit_price":0,"quantity":0,"total_price":0}]}'
)


def extract_invoice(image_path: str, ocr_text: str = "") -> dict:
    """
    Run Qwen2.5-VL inference via vLLM.
    Returns parsed invoice dict or empty dict on failure.
    """
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info

    engine, processor = _get_engine()

    prompt_text = (
        f"Trích xuất thông tin hóa đơn Việt Nam từ ảnh.\n"
        + (f"OCR text tham khảo:\n{ocr_text}\n\n" if ocr_text else "")
        + f"Trả về JSON format:\n{OUTPUT_FORMAT}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling = SamplingParams(temperature=0.1, max_tokens=512)
    outputs = engine.generate({"prompt": text, "multi_modal_data": {"image": image_path}},
                              sampling)

    response = outputs[0].outputs[0].text.strip()
    return _parse_json(response)


def _parse_json(text: str) -> dict:
    """Robust JSON parsing — 3 fallback strategies."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # 3. Strip markdown fences
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        pass

    return {}
