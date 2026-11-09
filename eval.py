"""
eval.py
=======
Automatic evaluation: base model vs fine-tuned PyExplain.

Loads Qwen2.5-Coder-7B-Instruct in 4-bit plus the published LoRA adapter
(AyushPatel28/PyExplain-qwen-coder-7b), the same way test_model.py does: the
base is loaded ONCE and the adapter is toggled off/on with disable_adapter().

For each of the 20 held-out snippets in eval_snippets.py it generates an
explanation with the adapter OFF (base) and ON (fine-tuned) using greedy
decoding and the system prompt from config.py, then scores every answer:

  * Flesch Reading Ease (textstat) — higher = easier to read
  * jargon count — occurrences of terms from a fixed list (JARGON_TERMS)
  * word count

Outputs:
  results/eval_results.csv   every answer and its scores
  results/eval_summary.md    averages, base vs fine-tuned

Needs a CUDA GPU (a free Colab T4 is enough) — see eval_colab.ipynb.

Run:
    python eval.py                  # full evaluation
    python eval.py --limit 2        # quick smoke test on the first 2 snippets
    python eval.py --summarize-only # rebuild the summary from an existing CSV
"""

import argparse
import csv
import os
import re
import time

import config  # imported first so HF_HOME is set before HuggingFace imports
from eval_snippets import EVAL_SNIPPETS, check_no_overlap

ADAPTER_ID = os.environ.get("ADAPTER_ID", "AyushPatel28/PyExplain-qwen-coder-7b")

RESULTS_DIR = os.path.join(config.PROJECT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "eval_results.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "eval_summary.md")

# Generation settings. Greedy decoding (do_sample=False) so results are
# deterministic; repetition_penalty matches test_model.py. max_new_tokens is a
# bit larger than test_model.py so answers are rarely cut off mid-sentence
# (which would distort the word count) — the CSV records when they are.
MAX_NEW_TOKENS = 400
REPETITION_PENALTY = 1.15

MODELS = ("base", "fine_tuned")

CSV_FIELDS = [
    "snippet_id",
    "topic",
    "model",
    "flesch_reading_ease",
    "jargon_count",
    "jargon_per_100_words",
    "jargon_terms",
    "word_count",
    "new_tokens",
    "hit_token_limit",
    "code",
    "answer",
]

# --------------------------------------------------------------------------
# Jargon list
# --------------------------------------------------------------------------
# A FIXED list of programming terms a complete beginner would not know. Each
# entry maps a term to a regex covering its common word forms. Matching is
# case-insensitive, whole-word only, and done on the prose only (code blocks
# and `inline code` are removed first, so code like `float(x)` doesn't count).
#
# Deliberately NOT included: everyday-sounding words that the system prompt
# itself asks the model to use and define (function, loop, variable, list,
# return value). Note the system prompt also asks the model to use proper terms
# and define them, so a higher jargon count is not automatically worse — read
# it together with the readability score.
JARGON_TERMS = {
    "iterate": r"iterat(?:e|es|ed|ing|ion|ions)",
    "iterable": r"iterables?",
    "iterator": r"iterators?",
    "parameter": r"parameters?",
    "argument": r"arguments?",
    "boolean": r"booleans?",
    "instantiate": r"instantiat(?:e|es|ed|ing|ion|ions)",
    "instance": r"instances?",
    "recursion": r"recursion|recursions|recursive|recursively",
    "exception": r"exceptions?",
    "constructor": r"constructors?",
    "initialize": r"initiali[sz](?:e|es|ed|ing|ation|er|ers)",
    "attribute": r"attributes?",
    "method": r"methods?",
    "object": r"objects?",
    "integer": r"integers?",
    "float": r"floats?|floating[- ]point",
    "string": r"strings?",
    "tuple": r"tuples?",
    "syntax": r"syntax",
    "operator": r"operators?",
    "expression": r"expressions?",
    "conditional": r"conditionals?",
    "comprehension": r"comprehensions?",
    "index": r"index|indexes|indexed|indexing|indices",
    "invoke": r"invok(?:e|es|ed|ing)|invocations?",
    "concatenate": r"concatenat(?:e|es|ed|ing|ion)",
    "increment": r"increment(?:s|ed|ing)?",
    "decrement": r"decrement(?:s|ed|ing)?",
    "scope": r"scopes?",
    "lambda": r"lambdas?",
    "mutable": r"mutable|immutable",
    "callable": r"callables?",
    "inheritance": r"inherit(?:s|ed|ing|ance)?",
    "subclass": r"subclass(?:es)?",
    "interpreter": r"interpreters?",
    "data type": r"data[- ]?types?",
}
_JARGON_RE = {
    term: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)
    for term, pattern in JARGON_TERMS.items()
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def prose_only(answer: str) -> str:
    """Strip ```fenced code blocks``` and `inline code` so only the English
    explanation is scored."""
    text = re.sub(r"```.*?(```|$)", " ", answer, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jargon_hits(text: str) -> dict:
    """Return {term: number_of_occurrences} for every jargon term found."""
    hits = {}
    for term, rx in _JARGON_RE.items():
        n = len(rx.findall(text))
        if n:
            hits[term] = n
    return hits


def score_answer(answer: str) -> dict:
    import textstat

    text = prose_only(answer)
    words = len(re.findall(r"\b[\w'-]+\b", text))
    hits = jargon_hits(text)
    jargon = sum(hits.values())
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2) if words else 0.0,
        "jargon_count": jargon,
        "jargon_per_100_words": round(100 * jargon / words, 2) if words else 0.0,
        "jargon_terms": "; ".join(f"{t}({n})" for t, n in sorted(hits.items())),
        "word_count": words,
    }


# --------------------------------------------------------------------------
# Model loading + generation (mirrors test_model.py)
# --------------------------------------------------------------------------
def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "eval.py needs a CUDA GPU to run the 7B model in 4-bit. "
            "Use eval_colab.ipynb on a free Colab T4."
        )

    print(f"[eval] Loading tokenizer: {config.MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"[eval] Loading base model in 4-bit: {config.MODEL_NAME}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=torch.float16,
        quantization_config=bnb,
        device_map={"": 0},
    )

    print(f"[eval] Attaching adapter: {ADAPTER_ID}", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER_ID)
    model.eval()
    # Clear sampling defaults from the model's generation config so greedy
    # decoding doesn't emit warnings about unused temperature/top_p/top_k.
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    return model, tokenizer


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
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, len(new_tokens)


# --------------------------------------------------------------------------
# Results files
# --------------------------------------------------------------------------
def write_csv(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def build_summary(rows) -> str:
    by_model = {m: [r for r in rows if r["model"] == m] for m in MODELS}
    n = len(by_model["fine_tuned"])

    metrics = [
        ("Flesch Reading Ease", "flesch_reading_ease", "higher = easier to read"),
        ("Jargon count", "jargon_count", "terms from the fixed list"),
        ("Jargon per 100 words", "jargon_per_100_words", "length-adjusted jargon"),
        ("Word count", "word_count", "length of the explanation"),
    ]

    lines = [
        "# PyExplain evaluation: base vs fine-tuned",
        "",
        f"- Base model: `{config.MODEL_NAME}` (4-bit NF4)",
        f"- Fine-tuned: base + LoRA adapter `{ADAPTER_ID}`",
        f"- Snippets: {n} held-out beginner snippets (`eval_snippets.py`), none in the training data",
        f"- Decoding: greedy, max_new_tokens={MAX_NEW_TOKENS}, "
        f"repetition_penalty={REPETITION_PENALTY}, same system prompt (`config.SYSTEM_PROMPT`)",
        "",
        "## Averages",
        "",
        "| Metric | Base | Fine-tuned | Change | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for label, key, note in metrics:
        b = _mean(r[key] for r in by_model["base"])
        f = _mean(r[key] for r in by_model["fine_tuned"])
        lines.append(f"| {label} | {b:.2f} | {f:.2f} | {f - b:+.2f} | {note} |")

    # Per-snippet head-to-head on readability and jargon.
    base_by_id = {r["snippet_id"]: r for r in by_model["base"]}
    pairs = [(base_by_id[r["snippet_id"]], r) for r in by_model["fine_tuned"]
             if r["snippet_id"] in base_by_id]
    easier = sum(float(f["flesch_reading_ease"]) > float(b["flesch_reading_ease"]) for b, f in pairs)
    less_jargon = sum(int(f["jargon_count"]) < int(b["jargon_count"]) for b, f in pairs)
    truncated = {m: sum(r["hit_token_limit"] in ("True", True) for r in by_model[m]) for m in MODELS}

    lines += [
        "",
        f"- Fine-tuned answer was easier to read (higher Flesch) on **{easier}/{len(pairs)}** snippets.",
        f"- Fine-tuned answer used less jargon on **{less_jargon}/{len(pairs)}** snippets.",
        f"- Answers cut off at the token limit: base {truncated['base']}, fine-tuned {truncated['fine_tuned']}.",
        "",
        "## Per snippet",
        "",
        "| Snippet | Flesch (base -> FT) | Jargon (base -> FT) | Words (base -> FT) |",
        "|---|---:|---:|---:|",
    ]
    for b, f in pairs:
        lines.append(
            f"| {b['snippet_id']} "
            f"| {float(b['flesch_reading_ease']):.1f} -> {float(f['flesch_reading_ease']):.1f} "
            f"| {b['jargon_count']} -> {f['jargon_count']} "
            f"| {b['word_count']} -> {f['word_count']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Scores are computed on the prose only: fenced code blocks and `inline code` are removed first.",
        "- Flesch Reading Ease: 60-70 is plain English, 30-50 is college level. Markdown bullet lists",
        "  without full stops can read as long sentences and lower the score for both models alike.",
        "- The system prompt asks the model to use proper terms *and* define them, so jargon count is",
        "  a signal to read alongside readability, not a pure \"lower is better\" score.",
        "- These are automatic proxies for beginner-friendliness, not a measure of correctness.",
        "  See `eval_results.csv` for every answer.",
        "",
    ]
    return "\n".join(lines)


def write_summary(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = build_summary(rows)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned PyExplain.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N snippets (smoke test)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="rebuild eval_summary.md from an existing eval_results.csv")
    args = parser.parse_args()

    if args.summarize_only:
        print(write_summary(read_csv()))
        return

    # Refuse to run if any eval snippet leaked into the training data.
    check_no_overlap()
    snippets = EVAL_SNIPPETS[: args.limit] if args.limit else EVAL_SNIPPETS

    model, tokenizer = load_model()
    rows = []
    start = time.time()
    for i, snip in enumerate(snippets, 1):
        prompt = build_prompt(tokenizer, snip["code"])
        for which in MODELS:
            if which == "base":
                with model.disable_adapter():  # adapter OFF = original model
                    answer, n_tok = generate(model, tokenizer, prompt)
            else:                              # adapter ON = fine-tuned
                answer, n_tok = generate(model, tokenizer, prompt)
            rows.append({
                "snippet_id": snip["id"],
                "topic": snip["topic"],
                "model": which,
                **score_answer(answer),
                "new_tokens": n_tok,
                "hit_token_limit": n_tok >= MAX_NEW_TOKENS,
                "code": snip["code"],
                "answer": answer,
            })
        # Save after every snippet so a disconnect doesn't lose everything.
        write_csv(rows)
        b, f = rows[-2], rows[-1]
        print(
            f"[eval] {i:2d}/{len(snippets)} {snip['id']:<22} "
            f"Flesch {b['flesch_reading_ease']:6.1f} -> {f['flesch_reading_ease']:6.1f} | "
            f"jargon {b['jargon_count']:2d} -> {f['jargon_count']:2d} | "
            f"words {b['word_count']:3d} -> {f['word_count']:3d} "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )

    summary = write_summary(rows)
    print(f"\n[eval] Wrote {CSV_PATH}")
    print(f"[eval] Wrote {SUMMARY_PATH}\n")
    print(summary)


if __name__ == "__main__":
    main()
