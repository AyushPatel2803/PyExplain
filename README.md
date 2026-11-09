# PyExplain — A Local AI That Explains Python Code

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen2.5--Coder--7B%20(LoRA)-yellow)](https://huggingface.co/AyushPatel28/PyExplain-qwen-coder-7b)
&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyExplain is a small, self-contained project that **fine-tunes a code language
model (Qwen2.5-Coder) with LoRA/QLoRA to explain Python code in plain,
beginner-friendly English**. It uses open models and no paid APIs.

You paste in a snippet of Python, and the model explains the overall purpose
and then walks through the code part by part.

The smaller 1.5B model was fine-tuned locally on a 6 GB laptop GPU; the 7B was
fine-tuned on a free cloud GPU (Google Colab) and can run locally in 4-bit.

🤗 **Fine-tuned model on Hugging Face:**
[AyushPatel28/PyExplain-qwen-coder-7b](https://huggingface.co/AyushPatel28/PyExplain-qwen-coder-7b)

## ▶️ Run it in Colab (free GPU)

| | |
|---|---|
| **PyExplain Live-Demo** — launches a public web link to try the 7B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AyushPatel2803/PyExplain/blob/main/demo_colab.ipynb) |
| **PyExplain** — train the 7B yourself (QLoRA) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AyushPatel2803/PyExplain/blob/main/PyExplain.ipynb) |
| **PyExplain Evaluation** — score base vs fine-tuned on 20 held-out snippets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AyushPatel2803/PyExplain/blob/main/eval_colab.ipynb) |

> In Colab, set **Runtime → Change runtime type → T4 GPU** before running the cells.

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

## Evaluation

PyExplain was scored on 20 held-out beginner Python snippets (none in the
training data), comparing the base Qwen2.5-Coder-7B-Instruct against the
PyExplain adapter with the same system prompt and greedy decoding.

| Metric | Base | PyExplain |
|---|---:|---:|
| Flesch Reading Ease (higher = easier) | 76.8 | **88.3** |
| Jargon terms per answer | 5.4 | **0.2** |
| Average length (words) | 157 | **58** |
| Answers cut off at the token limit | 3 | **0** |

PyExplain's answer was easier to read on **19/20** snippets and used less
jargon on **19/20**.

**Example: `for_squares`**

```python
for n in range(1, 6):
    print(n, 'squared is', n * n)
```

<table>
<tr><th>Base</th><th>PyExplain</th></tr>
<tr valign="top"><td>

This Python script prints out the squares of numbers from 1 to 5.

Here's how it works:
- `for n in range(1, 6):` This line starts a loop. It says "For each number `n`, starting at 1 and going up to but not including 6," do the stuff inside the loop.
- `print(n, 'squared is', n * n)` Inside the loop, this line prints three things:
  - The current number `n`
  - The string `'squared is'`
  - The square of `n` (which is calculated as `n * n`)

</td><td>

This prints squares of numbers from 1 to 5. The `range` makes a list [1,2,3,4,5]. The `for` loops over them. Each time, `n` gets next item. It prints `n`, says "squared is", then shows `n*n`. So it prints:
```
1 squared is 1
2 squared is 4
3 squared is 9
4 squared is 16
5 squared is 25
```

</td></tr>
</table>

**Example: `class_bank_account`**

```python
class BankAccount:
    def __init__(self, owner):
        self.owner = owner
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount
        return self.balance

account = BankAccount('Maya')
account.deposit(50)
```

<table>
<tr><th>Base</th><th>PyExplain</th></tr>
<tr valign="top"><td>

This Python code defines a class called `BankAccount` which represents a bank account. A class is like a blueprint for creating objects.

The `__init__` method initializes a new instance of the `BankAccount`. It sets up two attributes:
- `owner`: The name of the person who owns the account.
- `balance`: The current amount of money in the account, starting at zero.

The `deposit` method allows you to put money into the account. When you call this method, you pass in how much money (`amount`) you want to deposit. Inside the method, it adds this amount to the balance using the `+=` operator, which is short for "add this amount to whatever is already there". Then, it returns the updated balance.

Finally, the script creates an instance of `BankAccount`, naming it `account`, and assigns it to Maya. It then calls the `deposit` method on this account object, putting $50 into her account.

</td><td>

This creates a bank account where Maya owns it. The balance starts at zero. When she deposits money, her total goes up by whatever she puts in. So after putting $50 in, her new total would be $50. It's like having a piggy bank that adds coins inside whenever you put them in.

</td></tr>
</table>

**Limitations.** These scores measure readability, not correctness. A manual
review found 2 factual errors in PyExplain's answers: on `try_zero_division` it
describes the wrong condition for the error (it says the error happens when
nobody has paid, but it happens when the number of people is zero), and on
`listcomp_filter` it includes 19 in the list of temperatures over 20 (the base
model made the same mistake). Automatic correctness checking is future work.

Full results: [results/eval_summary.md](results/eval_summary.md) (averages and
per-snippet scores) and [results/eval_results.csv](results/eval_results.csv)
(every answer and its scores). Reproduce with
[`eval_colab.ipynb`](eval_colab.ipynb) or `python eval.py` on a CUDA GPU.

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

Developed on an **ASUS TUF FX505DU laptop**: NVIDIA GTX 1660 Ti (**6 GB VRAM**),
Ryzen 7, ~14 GB RAM. The 1.5B was fine-tuned locally; the **7B was fine-tuned
with QLoRA on a free Google Colab GPU (Tesla T4)** and runs locally in 4-bit
(~4.5 GB) — a practical example of training and running a 7B model on consumer
hardware.

The published model is the **7B**:
[🤗 AyushPatel28/PyExplain-qwen-coder-7b](https://huggingface.co/AyushPatel28/PyExplain-qwen-coder-7b).
To reproduce the 7B training yourself, open
[`PyExplain.ipynb`](PyExplain.ipynb) in Google Colab.

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

## How to use PyExplain (the explainer)

This is the main way to use the project — paste Python code and get an
explanation back.

**Launch it:**

- **Windows:** double-click **`PyExplain.bat`**, or run:
  ```bash
  python explain.py
  ```
- **macOS / Linux:**
  ```bash
  python explain.py
  ```

**Then:**

1. Wait for the model to load (a few seconds on CPU/small models; a couple of
   minutes for a 7B in 4-bit). When you see **`Ready!`**, it's good to go.
2. **Paste your Python code** (multi-line is fine — blank lines inside are kept).
3. Type **`RUN`** on its own line and press Enter to get the explanation.
4. Type **`CLEAR`** to discard what you pasted and start over, or **`quit`** to exit.

**Example session:**

```text
--- Paste code, then type RUN. (CLEAR = start over, quit = exit) ---
def average(nums):
    return sum(nums) / len(nums)
RUN

EXPLANATION:
This function calculates the average of a list of numbers. It takes one input,
`nums` (a list of numbers), adds them all up with `sum(nums)`, counts how many
there are with `len(nums)`, and divides the total by the count. So
average([2, 4, 6]) gives 4.
```

> Tip: the model loads once at startup, so keep the window open and explain as
> many snippets as you like in one session.

## Train your own adapter (optional)

To rebuild the dataset and fine-tune the model yourself, run the pipeline in
order:

```bash
python collect_data.py     # 1. build the dataset of code -> explanation pairs
python prepare_data.py     # 2. format into chat format + split train/validation
python fine_tune.py        # 3. fine-tune with LoRA/QLoRA -> saves adapter to output/
python test_model.py       # 4. compare the base model vs the fine-tuned model
```

Change the base model, paths, or hyper-parameters in **`config.py`**.

## Notes & limitations

- Best on **focused snippets** (a function, a loop, a class). Very long,
  multi-part scripts are harder for a small local model.
- Small local models can occasionally get a detail wrong — this is a learning /
  portfolio project, not a production tool.
- The base model weights (several GB) are downloaded on first run and are **not**
  included in this repo.

## License

MIT
