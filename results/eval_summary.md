# PyExplain evaluation: base vs fine-tuned

- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit NF4)
- Fine-tuned: base + LoRA adapter `AyushPatel28/PyExplain-qwen-coder-7b`
- Snippets: 20 held-out beginner snippets (`eval_snippets.py`), none in the training data
- Decoding: greedy, max_new_tokens=400, repetition_penalty=1.15, same system prompt (`config.SYSTEM_PROMPT`)

## Averages

| Metric | Base | Fine-tuned | Change | Note |
|---|---:|---:|---:|---|
| Flesch Reading Ease | 76.83 | 88.27 | +11.44 | higher = easier to read |
| Jargon count | 5.40 | 0.20 | -5.20 | terms from the fixed list |
| Jargon per 100 words | 3.55 | 0.42 | -3.13 | length-adjusted jargon |
| Word count | 156.75 | 57.70 | -99.05 | length of the explanation |

- Fine-tuned answer was easier to read (higher Flesch) on **19/20** snippets.
- Fine-tuned answer used less jargon on **19/20** snippets.
- Answers cut off at the token limit: base 3, fine-tuned 0.

## Per snippet

| Snippet | Flesch (base -> FT) | Jargon (base -> FT) | Words (base -> FT) |
|---|---:|---:|---:|
| for_squares | 87.5 -> 109.0 | 1 -> 0 | 65 -> 37 |
| while_countdown | 88.8 -> 92.8 | 0 -> 0 | 199 -> 63 |
| nested_loops | 73.2 -> 83.2 | 2 -> 0 | 101 -> 104 |
| listcomp_upper | 81.7 -> 78.8 | 5 -> 0 | 78 -> 54 |
| listcomp_filter | 67.1 -> 81.0 | 4 -> 1 | 124 -> 57 |
| dict_update_loop | 76.8 -> 91.7 | 1 -> 0 | 160 -> 98 |
| dict_letter_count | 76.3 -> 97.4 | 1 -> 0 | 198 -> 41 |
| class_bank_account | 71.5 -> 81.2 | 12 -> 0 | 138 -> 54 |
| class_str | 69.9 -> 98.4 | 27 -> 0 | 209 -> 109 |
| try_zero_division | 73.9 -> 76.8 | 3 -> 0 | 178 -> 68 |
| try_missing_file | 80.8 -> 83.1 | 4 -> 1 | 176 -> 63 |
| file_count_lines | 84.1 -> 100.2 | 3 -> 0 | 157 -> 54 |
| file_sum_numbers | 78.8 -> 87.8 | 7 -> 0 | 212 -> 38 |
| recursion_factorial | 65.2 -> 68.4 | 4 -> 1 | 155 -> 34 |
| recursion_sum_list | 82.1 -> 88.5 | 1 -> 0 | 225 -> 47 |
| str_replace_title | 79.2 -> 85.9 | 10 -> 0 | 122 -> 25 |
| str_count_vowels | 67.3 -> 87.2 | 9 -> 0 | 205 -> 37 |
| str_endswith | 86.0 -> 99.6 | 5 -> 1 | 98 -> 49 |
| default_argument | 67.9 -> 83.6 | 6 -> 0 | 202 -> 72 |
| list_methods | 78.6 -> 90.9 | 3 -> 0 | 133 -> 50 |

## Notes

- Scores are computed on the prose only: fenced code blocks and `inline code` are removed first.
- Flesch Reading Ease: 60-70 is plain English, 30-50 is college level. Markdown bullet lists
  without full stops can read as long sentences and lower the score for both models alike.
- The system prompt asks the model to use proper terms *and* define them, so jargon count is
  a signal to read alongside readability, not a pure "lower is better" score.
- These are automatic proxies for beginner-friendliness, not a measure of correctness.
  See `eval_results.csv` for every answer.
