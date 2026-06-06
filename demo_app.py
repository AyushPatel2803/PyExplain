"""
demo_app.py — PyExplain Live Demo

Loads the published PyExplain 7B (base + LoRA adapter) and launches a Gradio web
app with a public share link. Used by demo_colab.ipynb (and runnable directly).
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER = "AyushPatel28/PyExplain-qwen-coder-7b"

SYSTEM_PROMPT = (
    "You are an expert Python engineer explaining code to someone who has ZERO "
    "knowledge of Python or programming. Explain everything in the simplest, "
    "plainest everyday language so a complete beginner can fully understand. "
    "Use the proper programming terms, but the instant you use a term, explain "
    "in plain words what it means. Start with one plain sentence on what the "
    "code does overall, then explain the actual code part by part. IMPORTANT: "
    "do NOT include any worked example, do NOT trace the code with numbers, and "
    "do NOT calculate any result by hand — explain only in words."
)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()
for a in ("temperature", "top_p", "top_k"):
    if hasattr(model.generation_config, a):
        setattr(model.generation_config, a, None)


def explain(code):
    if not (code or "").strip():
        return "Paste some Python code first."
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Explain the following Python code:\n```python\n{code}\n```"}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**inp, max_new_tokens=350, do_sample=False, repetition_penalty=1.15,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=explain,
    inputs=gr.Code(language="python", label="Your Python code"),
    outputs=gr.Textbox(label="Explanation", lines=12),
    title="🐍 PyExplain Live Demo",
    description="Fine-tuned Qwen2.5-Coder-7B (QLoRA). Paste Python code and get a plain-English explanation.",
    examples=[["def reverse(s):\n    return s[::-1]"],
              ["squares = [x * x for x in range(5)]"]],
    flagging_mode="never",
)

if __name__ == "__main__":
    # quiet=True hides Gradio's default output (including the useless local URL);
    # we then print ONLY the public share link, clearly.
    _, _, public_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True)
    print("\n" + "=" * 64)
    print("  PyExplain Live Demo is ready! Open this public link:")
    print("  -> " + str(public_url))
    print("=" * 64 + "\n")
    demo.block_thread()  # keep the server running while you use the link

