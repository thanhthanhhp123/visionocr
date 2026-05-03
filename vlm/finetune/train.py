"""
Phase 2 — LoRA fine-tuning for Qwen2.5-VL-3B on Vietnamese invoices.

Usage:
    python vlm/finetune/train.py

Requirements:
    pip install -r requirements-train.txt
    GPU with 16GB+ VRAM recommended (RTX 3090 / 4090 / A100)
"""
import torch
import mlflow
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from vlm.finetune.dataset import InvoiceDataset

# ── Hyperparameters ─────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL    = "datasets/train.jsonl"
VAL_JSONL      = "datasets/val.jsonl"
OUTPUT_DIR     = "./checkpoints/qwen-lora-invoice"
ADAPTER_OUT    = "./models/qwen-lora-invoice-adapter"

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LR             = 2e-4
EPOCHS         = 5
GRAD_ACCUM     = 8       # effective batch = 8
MAX_PIXELS     = 768 * 28 * 28   # reduce if OOM
# ────────────────────────────────────────────────────────────────────────────


def main():
    # 4-bit quantization to fit in consumer GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_PIXELS,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = InvoiceDataset(TRAIN_JSONL, processor)
    val_ds   = InvoiceDataset(VAL_JSONL,   processor)
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
        # bf16=True,
        fp16=True,
        gradient_checkpointing=True,
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
        mlflow.log_params({
            "model":        MODEL_ID,
            "lora_r":       LORA_R,
            "lora_alpha":   LORA_ALPHA,
            "lr":           LR,
            "epochs":       EPOCHS,
            "grad_accum":   GRAD_ACCUM,
            "train_samples": len(train_ds),
            "val_samples":   len(val_ds),
        })

        trainer = MLflowTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        print("Starting training...")
        trainer.train()

        # Save LoRA adapter (~80MB, not full 6GB model)
        model.save_pretrained(ADAPTER_OUT)
        processor.save_pretrained(ADAPTER_OUT)
        mlflow.log_artifacts(ADAPTER_OUT)
        print(f"Adapter saved to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
