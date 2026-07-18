"""
Phase 2 — LoRA fine-tuning for Qwen2.5-VL-3B on Vietnamese invoices.

Usage:
    python vlm/finetune/train.py
    # or, from visionocr/vlm:
    python finetune/train.py

Requirements:
    pip install -r requirements-train.txt
    GPU with 16GB+ VRAM recommended (RTX 3090 / 4090 / A100)
"""

import sys
from pathlib import Path

import torch
import mlflow
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm.finetune.dataset import InvoiceDataset  # noqa: E402

# ── Hyperparameters ─────────────────────────────────────────────────────────
MODEL_ID = "/home/s3002152/LeeHoang_/vlm_invoice/models/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL = str(PROJECT_ROOT / "datasets/train.jsonl")
VAL_JSONL = str(PROJECT_ROOT / "datasets/val.jsonl")
OUTPUT_DIR = str(PROJECT_ROOT / "checkpoints/qwen-lora-invoice-v2")
ADAPTER_OUT = str(PROJECT_ROOT / "models/qwen-lora-invoice-adapter-v2")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 5e-5
EPOCHS = 5
GRAD_ACCUM = 8  # effective batch = 8
MAX_PIXELS = 768 * 28 * 28  # reduce if OOM
# ────────────────────────────────────────────────────────────────────────────


def main():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 4-bit quantization to fit in consumer GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_PIXELS,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = InvoiceDataset(TRAIN_JSONL, processor)
    val_ds = InvoiceDataset(VAL_JSONL, processor)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # MLflow tracking
    mlflow.set_experiment("qwen-vl-invoice-lora")

    class MLflowTrainer(Trainer):
        def log(self, logs, start_time=None):
            super().log(logs)
            mlflow.log_metrics(
                {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                step=self.state.global_step,
            )

    with mlflow.start_run(run_name=f"qwen2.5-vl-3b-lora-r{LORA_R}"):
        mlflow.log_params(
            {
                "model": MODEL_ID,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lr": LR,
                "epochs": EPOCHS,
                "grad_accum": GRAD_ACCUM,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
            }
        )

        trainer = MLflowTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=single_sample_collator,
        )

        print("Starting training...")
        trainer.train()

        # Save LoRA adapter (~80MB, not full 6GB model)
        model.save_pretrained(ADAPTER_OUT)
        processor.save_pretrained(ADAPTER_OUT)
        mlflow.log_artifacts(ADAPTER_OUT)
        print(f"Adapter saved to {ADAPTER_OUT}")


def single_sample_collator(features):
    if len(features) != 1:
        raise ValueError(
            "This training script expects per-device batch size 1 for variable-size images"
        )

    feature = features[0]
    return {
        "input_ids": feature["input_ids"].unsqueeze(0),
        "attention_mask": feature["attention_mask"].unsqueeze(0),
        "labels": feature["labels"].unsqueeze(0),
        "pixel_values": feature["pixel_values"],
        "image_grid_thw": feature["image_grid_thw"].unsqueeze(0),
    }


if __name__ == "__main__":
    main()
