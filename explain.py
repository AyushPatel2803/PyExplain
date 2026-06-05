"""
explain.py
==========
Interactive code explainer using the fine-tuned model.

Paste a piece of Python code, press Enter twice, and the trained model
explains it. Type 'quit' (or just press Enter on an empty line twice) to exit.

Run:
    python explain.py
"""

import os
import random

import config  # imported first so HF_HOME is set before HuggingFace imports

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Friendly random send-offs shown when the user quits.
GOODBYES = [
    "Hope you learned something new today!",
    "Happy coding — see you next time!",
    "Keep exploring that code. Catch you later!",
    "Nice session! Come back anytime you're curious.",
    "That's a wrap — go build something cool!",
    "Until next time, keep asking great questions!",
    "Every bit of code you understand makes you stronger. See you!",
    "Curiosity is how coders are made. Come back soon!",
    "You're getting better at this — keep it up!",
    "Done for now? Your code will be here when you return.",
    "Great work today. Go put that knowledge to use!",
    "Stay curious, keep learning, and happy coding!",
]


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)

    # Use the GPU if available. The 7B model is loaded in 4-bit so it fits in
    # 6 GB of video memory; on CPU it falls back to plain float32.
    use_gpu = torch.cuda.is_available()
    use_4bit = use_gpu and config.LOAD_IN_4BIT
    device = "cuda" if use_gpu else "cpu"

    where = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Loading the model on {where} (this takes a moment)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    base = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **base_kwargs)

    # Attach the fine-tuned adapter if one exists and matches this model;
    # otherwise just run the base model with the system prompt.
    model = base
    have_adapter = os.path.exists(
        os.path.join(config.OUTPUT_DIR, "adapter_config.json")
    )
    if config.USE_ADAPTER and have_adapter:
        try:
            model = PeftModel.from_pretrained(base, config.OUTPUT_DIR)
            print("(using your fine-tuned adapter)")
        except Exception:
            print("(the saved adapter doesn't match this model — running the "
                  "base model with the prompt instead)")
            model = base
    else:
        print("(no fine-tuned adapter yet — running the base model + prompt)")
    if not use_4bit:
        model.to(device)  # 4-bit is already placed on the GPU by device_map
    model.eval()
    # We do greedy (non-random) decoding, so clear the sampling defaults that
    # would otherwise trigger a harmless "flags ignored" warning on each call.
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    print("\n" + "=" * 50)
    print("            PyExplain  -  Python code, explained")
    print("=" * 50)
    print("Ready!")
    print("Paste your Python code (blank lines inside are fine),")
    print("then type RUN on its own line to get the explanation.")
    print("Type 'quit' to exit.\n")

    while True:
        print("--- Paste code, then type RUN. (CLEAR = start over, quit = exit) ---")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            stripped = line.strip().lower()
            if stripped == "quit":
                print(random.choice(GOODBYES))
                return
            if stripped == "clear":  # throw away what's been typed and restart
                lines = []
                print("(Cleared — paste your code again.)")
                continue
            if stripped == "run":  # explicit submit so pasted blank lines are kept
                break
            lines.append(line)

        code = "\n".join(lines).strip()
        if not code:
            print("(Nothing pasted — try again.)\n")
            continue

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Explain the following Python code:\n\n```python\n{code}\n```",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("\nThinking...\n")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print("EXPLANATION:\n" + answer + "\n")


if __name__ == "__main__":
    main()
