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
CALIB_JSONL = PROJECT_ROOT / "datasets/train.jsonl"


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

    # AutoAWQ's calibration harness (AwqQuantizer.init_quant) wraps decoder
    # layer 0 in a `Catcher` proxy to capture its input activations. The
    # installed transformers reads `decoder_layer.attention_type` off every
    # layer -- including the wrapped one -- to pick a causal mask before
    # invoking it, but `Catcher` never copied that attribute from the module
    # it wraps. This is a faithful copy of AwqQuantizer.init_quant (awq
    # 0.2.x) with a single added line restoring `attention_type` on the
    # Catcher instance; everything else is unchanged from the library.
    from awq.quantize.quantizer import AwqQuantizer
    from awq.utils.calib_data import get_calib_dataset
    from awq.utils.utils import clear_memory, get_best_device
    import torch.nn as nn

    def _init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.attention_type = getattr(module, "attention_type", None)

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)
                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            pass
        modules[0] = modules[0].module

        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")
        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(best_device)
        elif "qwen" in self.awq_model.model_type:
            layer_kwargs["attention_mask"] = None

        return modules, layer_kwargs, inps

    AwqQuantizer.init_quant = _init_quant

    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH, low_cpu_mem_usage=True, use_cache=False
    )
    processor = AutoProcessor.from_pretrained(MERGED_PATH)

    # AutoAWQ's calibration path (awq.utils.calib_data.get_calib_dataset) only
    # accepts plain text — a list of strings, a list of token-id lists, or a
    # HF dataset name. It cannot consume multimodal chat messages, and only
    # the language-model layers are quantized anyway (modules_to_not_convert
    # includes "visual"), so plain text through the language model is the
    # right calibration signal. Reuse the already-prepared training prompts
    # (same instruction template + target JSON the model was fine-tuned on)
    # instead of loading images.
    import json

    calib_texts = []
    with open(CALIB_JSONL, encoding="utf-8") as f:
        for line in f:
            if len(calib_texts) >= 128:
                break
            sample = json.loads(line)
            user_text = next(
                c["text"] for c in sample["messages"][0]["content"] if c["type"] == "text"
            )
            assistant_text = sample["messages"][1]["content"][0]["text"]
            calib_texts.append(f"{user_text}\n{assistant_text}")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(processor.tokenizer, quant_config=quant_config, calib_data=calib_texts)
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
