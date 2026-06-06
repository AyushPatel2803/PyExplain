"""
Shared configuration for the code-explanation fine-tuning project.

Everything that the four pipeline scripts (collect_data, prepare_data,
fine_tune, test_model) need to agree on lives here, so there is a single
place to change the base model, paths, or hyper-parameters.

--------------------------------------------------------------------------
SWITCHING TO MISTRAL 7B
--------------------------------------------------------------------------
This project is intentionally model-agnostic. To fine-tune Mistral 7B
(on a machine with a GPU or >=32 GB RAM) simply change MODEL_NAME to:

    MODEL_NAME = "mistralai/Mistral-7B-v0.3"

and, if you have a CUDA GPU, set LOAD_IN_4BIT = True (requires
`pip install bitsandbytes`). No other code needs to change.

On the current CPU-only machine (~14 GB RAM) we default to TinyLlama
1.1B, which shares the Llama/Mistral architecture and chat format but
actually fits in memory and trains in minutes.
"""

import os

# --------------------------------------------------------------------------
# Cache location
# --------------------------------------------------------------------------
# The system drive (C:) on this machine is nearly full, so redirect all
# HuggingFace downloads (models, tokenizers, datasets) to the project drive.
# This MUST run before transformers/huggingface_hub are imported, so every
# pipeline script imports `config` FIRST.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, "hf_cache"))

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# Qwen2.5-Coder-7B reads and understands code far better than the smaller
# 1.5B. It does not fit on a 6 GB GPU in normal precision, so we load it in
# 4-bit (see LOAD_IN_4BIT), which shrinks it ~4x to fit.
# (History: TinyLlama-1.1B -> Qwen-Coder-1.5B -> Qwen-Coder-7B.)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")

# 4-bit quantization (QLoRA). Requires a CUDA GPU + bitsandbytes. This is what
# lets the 7B model fit in 6 GB of video memory. The scripts only actually use
# it when a GPU is present; on CPU they fall back to normal loading.
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") == "1"

# Whether to attach the fine-tuned LoRA adapter from OUTPUT_DIR. When False (or
# when no matching adapter exists), the scripts run the plain base model with
# just the system prompt — useful for comparing "base model + prompt" against
# the fine-tuned version.
USE_ADAPTER = os.environ.get("USE_ADAPTER", "1") == "1"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "code_explanations.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")          # final LoRA adapter
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")  # training checkpoints

# --------------------------------------------------------------------------
# Tokenisation / training hyper-parameters
# --------------------------------------------------------------------------
MAX_SEQ_LEN = 512          # truncate code+explanation to this many tokens
                           # (512 keeps 7B 4-bit training within 6 GB of VRAM;
                           #  our code+explanation pairs are well under this)
VAL_FRACTION = 0.15        # held-out fraction for evaluation
SEED = 42

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Attention/MLP projection names shared by Llama, Mistral and TinyLlama.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Trainer
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "4"))
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8       # effective batch size = 8
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.0

# System prompt that frames the task for the model.
# Goal: keep the expert, technically-precise voice AND the correct terminology,
# but make every technical term understandable to someone with zero Python
# knowledge by explaining it in plain language right where it's used.
SYSTEM_PROMPT = (
    "You are an expert Python engineer explaining code to someone who has ZERO "
    "knowledge of Python or programming. Explain everything in the simplest, "
    "plainest everyday language so a complete beginner with no coding concepts "
    "can fully understand. Be technically correct and use the proper "
    "programming terms (e.g. function, argument, return value, loop, "
    "recursion), but every single time you use a term, immediately explain in "
    "plain words what that term means (for example: 'a function — a reusable "
    "block of code that takes inputs and gives back a result'). Also clearly "
    "explain what each function or piece of code IS and what it DOES. Do NOT "
    "mention your audience or say things like 'so a child can understand' — "
    "just give the explanation directly. Prefer a direct plain explanation by "
    "default. "
    "If something is simple enough to state directly, just state it — do not "
    "add an analogy that makes it longer or more confusing. Use a real-world "
    "analogy ONLY as a last resort when a concept is genuinely hard to grasp "
    "plainly, and even then only if the analogy is accurate and truly "
    "clarifies it. Do not skip details or get facts wrong. Start with one "
    "plain sentence on the overall purpose — what the code does and what it "
    "produces. Then ALWAYS explain the actual code itself, part by part: go "
    "through each meaningful piece (each line or section) and say what it is "
    "and what it does. Never give only the overall purpose or a vague summary "
    "of the idea — the part-by-part walkthrough of the real code is required, "
    "even for longer multi-part scripts. You may end with ONE short example "
    "showing an input and its final result on a single line (for example: "
    "'fibonacci(6) gives 8'). Do NOT trace or expand the code's execution by "
    "hand, do NOT show long chains of intermediate calculations, and never "
    "include arithmetic that could be wrong — just state the final result. "
    "Keep each part's explanation concise."
)
