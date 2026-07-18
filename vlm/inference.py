"""Invoice inference with an AWQ/vLLM fast path and a LoRA v2 fallback."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AWQ_MODEL_PATH = Path(os.getenv("AWQ_MODEL_PATH", PROJECT_ROOT / "models/qwen-awq-invoice-v2"))
ADAPTER_PATH = Path(os.getenv("LORA_ADAPTER_PATH", PROJECT_ROOT / "models/qwen-lora-invoice-adapter-v2"))
MERGED_MODEL_PATH = Path(os.getenv("MERGED_MODEL_PATH", PROJECT_ROOT / "models/qwen-merged-invoice-v2"))

_base_candidates = [
    PROJECT_ROOT.parent / "models/Qwen2.5-VL-3B-Instruct",
    PROJECT_ROOT / "models/Qwen2.5-VL-3B-Instruct",
]
BASE_MODEL = os.getenv("BASE_MODEL_ID") or next(
    (str(candidate) for candidate in _base_candidates if candidate.exists()),
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
BACKEND = os.getenv("INFERENCE_BACKEND", "auto").lower()

_engine = None
_processor = None
_hf_model = None
_hf_processor = None

OUTPUT_FORMAT = (
    '{"store_name":"","date":"YYYY-MM-DD","total":0,"discount":0,'
    '"items":[{"name":"","unit_price":0,"quantity":0,"total_price":0}]}'
)


def backend_name() -> str:
    if BACKEND in {"vllm", "awq"}:
        return "vllm_awq"
    if BACKEND in {"hf", "transformers", "lora"}:
        return "hf_lora"
    return "vllm_awq" if AWQ_MODEL_PATH.exists() else "hf_lora"


def _get_vllm_engine():
    global _engine, _processor
    if _engine is None:
        if not AWQ_MODEL_PATH.exists():
            raise RuntimeError(f"AWQ model is unavailable: {AWQ_MODEL_PATH}")
        from transformers import AutoProcessor
        from vllm import LLM

        _engine = LLM(
            model=str(AWQ_MODEL_PATH),
            quantization="awq",
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            limit_mm_per_prompt={"image": 1},
        )
        _processor = AutoProcessor.from_pretrained(AWQ_MODEL_PATH)
    return _engine, _processor


def _get_hf_model():
    global _hf_model, _hf_processor
    if _hf_model is None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32
        quantization = None
        if use_cuda and os.getenv("HF_4BIT", "1") != "0":
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        model_source = str(MERGED_MODEL_PATH) if MERGED_MODEL_PATH.exists() else BASE_MODEL
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=dtype,
            quantization_config=quantization,
            device_map="auto" if use_cuda else None,
        )
        if MERGED_MODEL_PATH.exists():
            _hf_model = base
            _hf_processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH)
        else:
            if not ADAPTER_PATH.exists():
                raise RuntimeError(f"LoRA adapter is unavailable: {ADAPTER_PATH}")
            _hf_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            _hf_processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
        _hf_model.eval()
    return _hf_model, _hf_processor


def _messages(image_path: str, ocr_text: str) -> list[dict]:
    prompt = "Trích xuất thông tin hóa đơn Việt Nam từ ảnh.\n"
    if ocr_text:
        prompt += f"OCR text tham khảo:\n{ocr_text}\n\n"
    prompt += f"Chỉ trả về JSON theo format:\n{OUTPUT_FORMAT}"
    return [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]


def _extract_vllm(messages: list[dict]) -> str:
    from vllm import SamplingParams

    engine, processor = _get_vllm_engine()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_path = messages[0]["content"][0]["image"]
    outputs = engine.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image_path}}],
        SamplingParams(temperature=0.0, max_tokens=512),
    )
    return outputs[0].outputs[0].text.strip()


def _extract_hf(messages: list[dict]) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _get_hf_model()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def extract_invoice(image_path: str, ocr_text: str = "") -> dict:
    """Extract and parse a structured invoice using the configured runtime backend."""
    messages = _messages(image_path, ocr_text)
    backend = backend_name()
    try:
        response = _extract_vllm(messages) if backend == "vllm_awq" else _extract_hf(messages)
    except Exception:
        if BACKEND != "auto" or backend != "vllm_awq":
            raise
        response = _extract_hf(messages)
    return _parse_json(response)


def _parse_json(text: str) -> dict:
    """Robust JSON parsing with direct, fenced, and embedded-object variants."""
    for candidate in (text, re.sub(r"```json|```", "", text).strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
