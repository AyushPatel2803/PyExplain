"""
prepare_data.py
===============
Step 2 of the pipeline.

Converts the raw code/explanation pairs from data/code_explanations.json
into the chat-formatted training records the transformer expects, then
splits them into train/validation JSONL files.

For each pair we build a 3-turn chat conversation (system / user / assistant)
and render it with the model's own chat template via the tokenizer. We store:

  * "text"   - the full rendered conversation (prompt + assistant answer),
               used as the training target.
  * "prompt" - the same conversation up to (and including) the assistant
               turn header but WITHOUT the answer, so fine_tune.py can mask
               the prompt tokens out of the loss (the model is only trained
               to produce the explanation, not to echo the question).

Run:
    python prepare_data.py
"""

import json
import os
import random

import config  # imported first so HF_HOME is set before HuggingFace imports

from transformers import AutoTokenizer


def build_records(pairs, tokenizer):
    records = []
    for pair in pairs:
        user_msg = f"Explain the following Python code:\n\n```python\n{pair['code']}\n```"
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": pair["explanation"]},
        ]

        # Full conversation including the assistant's answer (training target).
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Same conversation but without the answer; add_generation_prompt=True
        # appends the assistant-turn header so lengths line up for masking.
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        records.append({"text": text, "prompt": prompt})
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(
            f"{config.RAW_DATA_PATH} not found. Run collect_data.py first."
        )

    with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"[prepare_data] Loaded {len(pairs)} pairs. "
          f"Loading tokenizer for chat template: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    records = build_records(pairs, tokenizer)

    # Deterministic shuffle + split.
    random.seed(config.SEED)
    random.shuffle(records)
    n_val = max(1, int(len(records) * config.VAL_FRACTION))
    val_records = records[:n_val]
    train_records = records[n_val:]

    os.makedirs(config.DATA_DIR, exist_ok=True)
    write_jsonl(config.TRAIN_PATH, train_records)
    write_jsonl(config.VAL_PATH, val_records)

    print(f"[prepare_data] Wrote {len(train_records)} train -> {config.TRAIN_PATH}")
    print(f"[prepare_data] Wrote {len(val_records)} val   -> {config.VAL_PATH}")

    # Show one rendered example so the format is easy to inspect.
    print("\n[prepare_data] Example rendered conversation:\n")
    print(train_records[0]["text"][:600])


if __name__ == "__main__":
    main()
