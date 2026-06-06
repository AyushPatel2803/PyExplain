"""
fine_tune.py
============
Step 3 of the pipeline.

Fine-tunes the base causal language model on the code-explanation data using
PyTorch + PEFT/LoRA, on CPU. Only the small LoRA adapter weights are trained;
the base model stays frozen, which keeps memory and disk usage low.

The trained adapter is saved to output/ (config.OUTPUT_DIR).

Run:
    python fine_tune.py
"""

import os

import config  # imported first so HF_HOME is set before HuggingFace imports

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def tokenize_record(record, tokenizer):
    """Tokenise one record, masking the prompt tokens out of the loss."""
    full = tokenizer(
        record["text"],
        truncation=True,
        max_length=config.MAX_SEQ_LEN,
        add_special_tokens=False,  # chat template already added them
    )
    # Length of the prompt portion (everything before the assistant answer).
    prompt_ids = tokenizer(
        record["prompt"],
        truncation=True,
        max_length=config.MAX_SEQ_LEN,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = list(full["input_ids"])
    # -100 is ignored by the loss, so the model is only trained on the answer.
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    full["labels"] = labels
    return full


def main() -> None:
    set_seed(config.SEED)
    # Use all available CPU cores for the math-heavy ops.
    torch.set_num_threads(os.cpu_count() or 4)

    for path in (config.TRAIN_PATH, config.VAL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run prepare_data.py first."
            )

    print(f"[fine_tune] Loading tokenizer + model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    use_4bit = use_gpu and config.LOAD_IN_4BIT  # QLoRA only makes sense on a GPU
    if use_4bit:
        print(f"[fine_tune] GPU detected: {torch.cuda.get_device_name(0)} — "
              f"training the 7B model in 4-bit (QLoRA).")
    elif use_gpu:
        print(f"[fine_tune] GPU detected: {torch.cuda.get_device_name(0)} — "
              f"training on GPU in float16.")
    else:
        print("[fine_tune] No GPU detected — training on CPU in float32 (slow).")

    model_kwargs = {"dtype": torch.float16 if use_gpu else torch.float32}
    if use_4bit:
        # 4-bit shrinks the frozen base model ~4x so the 7B fits in 6 GB VRAM.
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = {"": 0}  # put the whole model on GPU 0

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **model_kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if use_4bit:
        # Prepares the quantized model for training: enables gradient
        # checkpointing and makes the inputs require grad so gradients reach
        # the LoRA adapters.
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters to the attention projection layers.
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_4bit:
        # With the base model frozen + gradient checkpointing, the embedding
        # output does not require grad, which breaks the backward graph. This
        # hook makes the inputs require grad so gradients flow to the adapters.
        # (prepare_model_for_kbit_training already handles this for 4-bit.)
        model.enable_input_require_grads()

    # ---- Data ----
    print("[fine_tune] Loading and tokenising datasets")
    raw = load_dataset(
        "json",
        data_files={"train": config.TRAIN_PATH, "validation": config.VAL_PATH},
    )
    tokenized = raw.map(
        lambda r: tokenize_record(r, tokenizer),
        remove_columns=raw["train"].column_names,
        desc="Tokenising",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    # ---- Training ----
    args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=1,
        # Eval/save once at the END only. On a ~14 GB RAM machine the extra
        # memory spike from per-epoch checkpoint saves + eval was enough to get
        # the process OOM-killed mid-run, so we avoid those spikes here.
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        use_cpu=not use_gpu,
        fp16=use_gpu,  # half-precision training on the GPU (faster, less memory)
        # Paged 8-bit optimizer: keeps optimizer memory tiny and offloads to
        # regular RAM during spikes, which is what lets a 7B QLoRA run fit in
        # a 6 GB GPU. Falls back to the normal optimizer when not on a GPU.
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    print("[fine_tune] Starting training (CPU training is slow — please wait)...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"[fine_tune] Final eval metrics: {metrics}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"[fine_tune] Saved LoRA adapter + tokenizer to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
