"""
test_model.py
=============
Step 4 of the pipeline.

Loads the base model and the LoRA-fine-tuned adapter and compares their
code-explanation output side by side on held-out snippets, so you can see
that fine-tuning improved the explanations.

Run:
    python test_model.py
"""

import os

import config  # imported first so HF_HOME is set before HuggingFace imports

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Snippets the model has NOT been trained on, to test generalisation.
TEST_SNIPPETS = [
    "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
    "result = sorted(data, key=lambda item: item['age'], reverse=True)",
    "async def fetch(url):\n    async with session.get(url) as resp:\n        return await resp.json()",
]


def build_prompt(tokenizer, code):
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Explain the following Python code:\n\n```python\n{code}\n```",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)

    if not os.path.isdir(config.OUTPUT_DIR) or not os.listdir(config.OUTPUT_DIR):
        raise FileNotFoundError(
            f"No fine-tuned adapter in {config.OUTPUT_DIR}. Run fine_tune.py first."
        )

    print(f"[test_model] Loading tokenizer from {config.OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use the GPU if available. The 7B model is loaded in 4-bit to fit in 6 GB
    # of video memory; on CPU it falls back to plain float32.
    use_gpu = torch.cuda.is_available()
    use_4bit = use_gpu and config.LOAD_IN_4BIT
    device = "cuda" if use_gpu else "cpu"

    base_kwargs = {"dtype": torch.float16 if use_gpu else torch.float32}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_kwargs["device_map"] = {"": 0}

    # Memory-efficient: load the base model ONCE, attach the LoRA adapter, and
    # toggle it on/off with disable_adapter() to compare. This avoids holding
    # two full copies of the model in memory at the same time.
    print(f"[test_model] Loading base model on {device}: {config.MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **base_kwargs)

    print(f"[test_model] Attaching fine-tuned adapter from {config.OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base, config.OUTPUT_DIR)
    if not use_4bit:
        model.to(device)
    model.eval()
    _ = use_4bit  # (4-bit model is already on GPU via device_map)

    for i, code in enumerate(TEST_SNIPPETS, 1):
        prompt = build_prompt(tokenizer, code)
        print("\n" + "=" * 78, flush=True)
        print(f"TEST {i}  | Code:\n{code}", flush=True)
        print("-" * 78, flush=True)
        # disable_adapter() = the original, untrained model behaviour.
        with model.disable_adapter():
            print("BASE MODEL:\n" + generate(model, tokenizer, prompt), flush=True)
        print("-" * 78, flush=True)
        # adapter active = your fine-tuned model.
        print("FINE-TUNED MODEL:\n" + generate(model, tokenizer, prompt), flush=True)

    print("\n" + "=" * 78)
    print("[test_model] Done. Compare the two outputs above for each snippet.")


if __name__ == "__main__":
    main()
