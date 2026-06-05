"""
quick_test.py
=============
Quick check: how good is the 7B model with our prompt but WITHOUT fine-tuning?

Loads Qwen2.5-Coder-7B in 4-bit (no LoRA adapter) and runs it on a couple of
snippets so we can judge whether fine-tuning is even necessary.
"""

import config  # first, so HF_HOME is set before HuggingFace imports

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SNIPPETS = [
    # single function
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    # the multi-part script the fine-tuned 1.5B got wrong
    'test_snippets = [\n'
    '    "x = [i**2 for i in range(5)]",\n'
    '    "def reverse(s): return s[::-1]",\n'
    ']\n\n'
    'for code in test_snippets:\n'
    '    print(f"\\nCode: {code}")\n'
    '    print(explain_code(code, model, tokenizer))',
]


def main():
    print(f"Loading {config.MODEL_NAME} in 4-bit (no fine-tuning)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, quantization_config=bnb, device_map={"": 0},
        dtype=torch.float16,
    )
    model.eval()
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    for i, code in enumerate(SNIPPETS, 1):
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Explain the following Python code:\n\n```python\n{code}\n```"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print("\n" + "=" * 78, flush=True)
        print(f"SNIPPET {i}:\n{code}", flush=True)
        print("-" * 78, flush=True)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        print("7B (no fine-tune) EXPLANATION:\n"
              + tokenizer.decode(new, skip_special_tokens=True).strip(), flush=True)

    print("\n" + "=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
