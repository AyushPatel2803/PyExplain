"""
eval_snippets.py
================
A held-out evaluation set of 20 beginner-level Python snippets.

None of these snippets appear in the training data (data/code_explanations.json).
`check_no_overlap()` enforces that: it compares every snippet to every training
example with ALL whitespace removed and raises an error on any match.

Run directly to verify:
    python eval_snippets.py
"""

import json
import os

_TRAINING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "code_explanations.json"
)

# Each entry: a short id, the topic it covers, and the code.
EVAL_SNIPPETS = [
    {
        "id": "for_squares",
        "topic": "for loop",
        "code": "for n in range(1, 6):\n    print(n, 'squared is', n * n)",
    },
    {
        "id": "while_countdown",
        "topic": "while loop",
        "code": "steps = 10\nwhile steps > 0:\n    print('Steps left:', steps)\n    steps -= 2\nprint('Done!')",
    },
    {
        "id": "nested_loops",
        "topic": "nested loops",
        "code": "for row in range(1, 4):\n    for col in range(1, 4):\n        print(row * col, end=' ')\n    print()",
    },
    {
        "id": "listcomp_upper",
        "topic": "list comprehension",
        "code": "cities = ['paris', 'tokyo', 'lima']\nloud = [city.upper() for city in cities]",
    },
    {
        "id": "listcomp_filter",
        "topic": "list comprehension with condition",
        "code": "temps = [12, 25, 31, 8, 19]\nwarm_days = [t for t in temps if t > 20]",
    },
    {
        "id": "dict_update_loop",
        "topic": "dictionaries",
        "code": "stock = {'apples': 5, 'pears': 0}\nstock['pears'] = 12\nfor fruit, amount in stock.items():\n    print(fruit, amount)",
    },
    {
        "id": "dict_letter_count",
        "topic": "dictionary counting",
        "code": "word = 'banana'\ncounts = {}\nfor letter in word:\n    counts[letter] = counts.get(letter, 0) + 1\nprint(counts)",
    },
    {
        "id": "class_bank_account",
        "topic": "classes and methods",
        "code": (
            "class BankAccount:\n"
            "    def __init__(self, owner):\n"
            "        self.owner = owner\n"
            "        self.balance = 0\n\n"
            "    def deposit(self, amount):\n"
            "        self.balance += amount\n"
            "        return self.balance\n\n"
            "account = BankAccount('Maya')\n"
            "account.deposit(50)"
        ),
    },
    {
        "id": "class_str",
        "topic": "class with __str__",
        "code": (
            "class Book:\n"
            "    def __init__(self, title, pages):\n"
            "        self.title = title\n"
            "        self.pages = pages\n\n"
            "    def __str__(self):\n"
            "        return f'{self.title} ({self.pages} pages)'\n\n"
            "print(Book('Dune', 412))"
        ),
    },
    {
        "id": "try_zero_division",
        "topic": "try/except",
        "code": (
            "def split_bill(total, people):\n"
            "    try:\n"
            "        return total / people\n"
            "    except ZeroDivisionError:\n"
            "        return 'Nobody to split with'"
        ),
    },
    {
        "id": "try_missing_file",
        "topic": "try/except with files",
        "code": (
            "try:\n"
            "    with open('notes.txt') as f:\n"
            "        print(f.read())\n"
            "except FileNotFoundError:\n"
            "    print('The file does not exist yet.')"
        ),
    },
    {
        "id": "file_count_lines",
        "topic": "file reading",
        "code": "line_count = 0\nwith open('diary.txt') as f:\n    for line in f:\n        line_count += 1\nprint(line_count)",
    },
    {
        "id": "file_sum_numbers",
        "topic": "file reading with conversion",
        "code": (
            "total = 0\n"
            "with open('numbers.txt') as f:\n"
            "    for line in f:\n"
            "        total += float(line.strip())\n"
            "print('Total:', total)"
        ),
    },
    {
        "id": "recursion_factorial",
        "topic": "recursion",
        "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    },
    {
        "id": "recursion_sum_list",
        "topic": "recursion on a list",
        "code": "def total(items):\n    if not items:\n        return 0\n    return items[0] + total(items[1:])",
    },
    {
        "id": "str_replace_title",
        "topic": "string methods",
        "code": "headline = 'the quick brown fox'\nheadline = headline.replace('quick', 'slow').title()\nprint(headline)",
    },
    {
        "id": "str_count_vowels",
        "topic": "strings and loops",
        "code": "def count_vowels(text):\n    count = 0\n    for ch in text.lower():\n        if ch in 'aeiou':\n            count += 1\n    return count",
    },
    {
        "id": "str_endswith",
        "topic": "string methods with filtering",
        "code": "files = ['report.pdf', 'photo.jpg', 'resume.pdf']\npdfs = [name for name in files if name.endswith('.pdf')]",
    },
    {
        "id": "default_argument",
        "topic": "function with default argument",
        "code": "def welcome(name='friend', times=1):\n    for _ in range(times):\n        print('Welcome,', name)\n\nwelcome()\nwelcome('Lena', 2)",
    },
    {
        "id": "list_methods",
        "topic": "list methods",
        "code": (
            "shopping = ['milk', 'eggs']\n"
            "shopping.append('bread')\n"
            "shopping.remove('eggs')\n"
            "last = shopping.pop()\n"
            "print(shopping, last)"
        ),
    },
]


def _normalize(code: str) -> str:
    """Remove ALL whitespace so formatting differences can't hide a duplicate."""
    return "".join(code.split())


def get_snippet(snippet_id: str) -> str:
    """Return the code for one eval snippet by its id."""
    for s in EVAL_SNIPPETS:
        if s["id"] == snippet_id:
            return s["code"]
    raise KeyError(f"No eval snippet with id {snippet_id!r}")


def check_no_overlap(training_path: str = _TRAINING_PATH) -> None:
    """Raise ValueError if any eval snippet matches a training example
    (ignoring whitespace), or if the eval set contains duplicates."""
    with open(training_path, encoding="utf-8") as f:
        training = json.load(f)
    train_index = {_normalize(ex["code"]): i for i, ex in enumerate(training)}

    problems = []
    seen = {}
    for s in EVAL_SNIPPETS:
        norm = _normalize(s["code"])
        if norm in train_index:
            problems.append(
                f"eval snippet {s['id']!r} matches training example #{train_index[norm]}"
            )
        if norm in seen:
            problems.append(f"eval snippet {s['id']!r} duplicates {seen[norm]!r}")
        seen[norm] = s["id"]

    if len(EVAL_SNIPPETS) != 20:
        problems.append(f"expected 20 eval snippets, found {len(EVAL_SNIPPETS)}")

    if problems:
        raise ValueError(
            "Eval set overlaps training data:\n  - " + "\n  - ".join(problems)
        )


if __name__ == "__main__":
    check_no_overlap()
    print(f"OK: {len(EVAL_SNIPPETS)} eval snippets, none match the training data.")
