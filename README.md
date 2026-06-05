# PyExplain — A Local AI That Explains Python Code

PyExplain is a small, self-contained project that **fine-tunes a code language
model to explain Python code in plain, beginner-friendly English** — and runs
entirely **locally** on a consumer laptop GPU (no cloud, no API keys).

You paste in a snippet of Python, and the model explains the overall purpose
and then walks through the code part by part.

```
==================================================
            PyExplain  -  Python code, explained
==================================================
Ready!
--- Paste code, then type RUN. (CLEAR = start over, quit = exit) ---
def reverse(s):
    return s[::-1]
RUN

EXPLANATION:
This function reverses a string. It takes one input, `s` (the text to
reverse), and returns it back-to-front. The `[::-1]` is a "slice" that walks
through the characters from end to start. So reverse("cat") gives "tac".
```

## What it does

- Fine-tunes a code model with **LoRA / QLoRA** on a hand-written dataset of
  Python *code → explanation* pairs.
- Produces explanations that aim to be **simple, accurate, and complete**
  (overall purpose + a part-by-part walkthrough of the actual code).
- Includes a friendly interactive tool (`PyExplain.bat` / `explain.py`) and a
  base-vs-fine-tuned comparison script.

## Tech stack

- **Python**, **PyTorch** (CUDA build)
- **Hugging Face Transformers**, **PEFT** (LoRA/QLoRA), **Datasets**
- **bitsandbytes** for 4-bit quantization
- Base model: **Qwen2.5-Coder** (the code is model-agnostic — set `MODEL_NAME`
  in `config.py`; it has run on TinyLlama-1.1B, Qwen2.5-Coder-1.5B, and
  Qwen2.5-Coder-7B).

## Built on modest hardware

Developed and trained on an **ASUS TUF FX505DU laptop**: NVIDIA GTX 1660 Ti
(**6 GB VRAM**), Ryzen 7, ~14 GB RAM. The 7B model is loaded in **4-bit
(QLoRA)** so it fits in 6 GB of video memory — a practical example of
fine-tuning a 7B model on consumer hardware.

## How it works (the pipeline)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `collect_data.py` | Builds the dataset of code→explanation pairs |
| 2 | `prepare_data.py` | Formats them into chat format, splits train/validation |
| 3 | `fine_tune.py` | Fine-tunes the model with LoRA/QLoRA, saves the adapter to `output/` |
| 4 | `test_model.py` | Compares the base model vs the fine-tuned model side by side |
| — | `explain.py` | Interactive explainer (launched by `PyExplain.bat`) |

All shared settings (model name, paths, hyper-parameters, the system prompt)
live in `config.py`.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

# 2. Install PyTorch (GPU build — pick the index for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. Install the rest
pip install -r requirements.txt
```

## Usage

```bash
python collect_data.py     # 1. build the dataset
python prepare_data.py     # 2. format + split
python fine_tune.py        # 3. train the LoRA adapter -> output/
python test_model.py       # 4. compare base vs fine-tuned
python explain.py          # interactive explainer (or double-click PyExplain.bat on Windows)
```

In the interactive tool: paste your code, type **`RUN`** to get the
explanation, **`CLEAR`** to start over, **`quit`** to exit.

## Notes & limitations

- Best on **focused snippets** (a function, a loop, a class). Very long,
  multi-part scripts are harder for a small local model.
- Small local models can occasionally get a detail wrong — this is a learning /
  portfolio project, not a production tool.
- The base model weights (several GB) are downloaded on first run and are **not**
  included in this repo.

## License

MIT
