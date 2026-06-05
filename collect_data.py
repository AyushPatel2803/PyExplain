"""
collect_data.py
===============
Step 1 of the pipeline.

Builds a curated dataset of real Python *code -> explanation* pairs and writes
it to data/code_explanations.json.

Every explanation is written in a deliberately SIMPLE style: short sentences,
one idea at a time, plain everyday words, technical terms explained the moment
they appear, and a concrete "so you get X" at the end. The goal is that someone
with zero Python knowledge can understand it, while it stays fully accurate.

The set covers easy basics (functions, loops, if/else), common patterns
(comprehensions, generators, dictionaries) and genuinely hard topics
(decorators with arguments, context managers, closures, async, recursion) so
the fine-tuned model learns to explain advanced code simply too.

Run:
    python collect_data.py
"""

import json
import os

import config


# Each pair: a self-contained Python snippet and a simple, accurate explanation.
CODE_EXPLANATION_PAIRS = [
    # ---------------- Easy basics ----------------
    {
        "code": "def add(a, b):\n    return a + b",
        "explanation": (
            "This is a function. A function is like a little machine: you put "
            "things in and it gives something back. This machine is called "
            "`add`. You give it two numbers. It adds them together and hands "
            "back the answer. So `add(2, 3)` gives you 5."
        ),
    },
    {
        "code": "for fruit in ['apple', 'banana', 'cherry']:\n    print(fruit)",
        "explanation": (
            "This is a loop. A loop repeats the same action for each item in a "
            "list. Here the list has three fruits. The loop takes one fruit at "
            "a time and shows it on the screen. So you see apple, then banana, "
            "then cherry — one per line."
        ),
    },
    {
        "code": "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n"
                "    grade = 'B'\nelse:\n    grade = 'C'",
        "explanation": (
            "This picks a letter grade from a number score. It checks things "
            "in order. First: is the score 90 or more? If yes, the grade is "
            "'A'. If not, it checks the next one: 80 or more gives 'B'. If none "
            "of those are true, the last part runs and the grade is 'C'. Only "
            "the first true check is used; the rest are skipped."
        ),
    },
    {
        "code": "count = 0\nwhile count < 5:\n    print(count)\n    count += 1",
        "explanation": (
            "This is a 'while' loop — it keeps repeating as long as something "
            "stays true. It starts a counter at 0. While the counter is below "
            "5, it shows the counter and then adds 1 to it. So it prints 0, 1, "
            "2, 3, 4, and stops once the counter reaches 5. Without the 'add 1' "
            "step, it would repeat forever."
        ),
    },
    {
        "code": "person = {'name': 'Ada', 'age': 36}\nprint(person['name'])",
        "explanation": (
            "This makes a dictionary and looks something up in it. A dictionary "
            "stores information as labels and values. Here the label 'name' has "
            "the value 'Ada', and 'age' has 36. Writing `person['name']` asks "
            "for the value stored under the label 'name'. So it prints 'Ada'."
        ),
    },
    {
        "code": "results = []\nfor i in range(3):\n    results.append(i * 10)",
        "explanation": (
            "This builds a list step by step. It starts with an empty list. "
            "The loop runs three times. Each time, it adds one new item to the "
            "end of the list — the current number times 10. So the list grows "
            "to [0, 10, 20]. `append` just means 'add to the end'."
        ),
    },
    {
        "code": "name = 'Sam'\nage = 30\nmessage = f'{name} is {age} years old'",
        "explanation": (
            "This builds a sentence by dropping variables into text. The `f` "
            "before the quotes turns on this feature. Anything inside curly "
            "braces { } gets swapped for that variable's value. So {name} "
            "becomes 'Sam' and {age} becomes 30, making the sentence 'Sam is "
            "30 years old'."
        ),
    },
    {
        "code": "letters = ['a', 'b', 'c', 'd', 'e']\nmiddle = letters[1:4]",
        "explanation": (
            "This takes a piece out of a list. The list has five items, counted "
            "starting at 0 (so 'a' is 0, 'b' is 1, and so on). `letters[1:4]` "
            "grabs the items from position 1 up to — but not including — "
            "position 4. So you get ['b', 'c', 'd']. The start is included and "
            "the end is left out."
        ),
    },
    {
        "code": "unique = list(set(names))",
        "explanation": (
            "This removes duplicates from a list. A 'set' is a collection that "
            "can't hold the same item twice. Turning the list into a set drops "
            "any repeats, and turning it back into a list gives you the clean "
            "version. So ['a', 'b', 'a', 'c'] becomes ['a', 'b', 'c'] (the "
            "order may change)."
        ),
    },
    {
        "code": "status = 'pass' if score >= 50 else 'fail'",
        "explanation": (
            "This picks one of two values in a single line. It checks: is the "
            "score 50 or more? If yes, `status` becomes 'pass'. If no, it "
            "becomes 'fail'. It's a short way to write a simple either/or choice "
            "on one line."
        ),
    },
    {
        "code": "clean = text.strip().lower()",
        "explanation": (
            "This cleans up a piece of text. `.strip()` removes any spaces at "
            "the start and end. `.lower()` turns all letters into small "
            "(lowercase) letters. They run one after the other on the same "
            "text. So '  Hello  ' becomes 'hello'."
        ),
    },
    {
        "code": "sentence = ' '.join(words)",
        "explanation": (
            "This glues a list of words into one piece of text. The ' ' (a "
            "space) is the glue placed between each word. So ['I', 'love', "
            "'Python'] becomes 'I love Python'. `join` is the normal way to "
            "turn a list of pieces into one joined string."
        ),
    },
    {
        "code": 'parts = "a,b,c".split(",")',
        "explanation": (
            "This breaks a piece of text into pieces. It cuts the text wherever "
            "it finds a comma. So 'a,b,c' becomes the list ['a', 'b', 'c']. "
            "`split` is handy for pulling apart text that has a separator like "
            "commas or spaces."
        ),
    },
    {
        "code": "avg = sum(scores) / len(scores)",
        "explanation": (
            "This finds the average of a list of numbers. `sum(scores)` adds "
            "them all up. `len(scores)` counts how many there are. Dividing the "
            "total by the count gives the average. So [2, 4, 6] gives "
            "(2 + 4 + 6) / 3 = 4."
        ),
    },
    {
        "code": "color = settings.get('color', 'blue')",
        "explanation": (
            "This safely looks up a value in a dictionary (a collection of "
            "labels and values). It asks for the value under the label 'color'. "
            "If 'color' isn't there, instead of crashing it just gives back "
            "'blue' as a backup. So you always get an answer, even when the "
            "label is missing."
        ),
    },
    {
        "code": "if isinstance(x, int):\n    print('it is a whole number')",
        "explanation": (
            "This checks what kind of value something is. `isinstance(x, int)` "
            "asks 'is x a whole number (an int)?' and gives back True or False. "
            "It's the safe way to check a value's type before using it, so you "
            "don't accidentally do the wrong thing with the wrong kind of data."
        ),
    },
    {
        "code": "longest = max(words, key=len)",
        "explanation": (
            "This finds the longest word in a list. `max` normally finds the "
            "biggest value, but `key=len` tells it to compare words by their "
            "length (how many letters each has). So from ['hi', 'hello', "
            "'hey'], it gives back 'hello' because that word has the most "
            "letters."
        ),
    },
    {
        "code": "with open('out.txt', 'w') as f:\n    f.write('hello')",
        "explanation": (
            "This creates a text file and writes into it. The 'w' means 'write "
            "mode', which makes a brand-new file (or replaces one that already "
            "exists). The word `with` closes the file for you when you're done. "
            "`f.write('hello')` puts the text 'hello' inside. So you end up with "
            "a file called out.txt containing the word hello."
        ),
    },

    # ---------------- Common patterns ----------------
    {
        "code": "squares = [x * x for x in range(10) if x % 2 == 0]",
        "explanation": (
            "This makes a list of numbers in one line. It looks at the numbers "
            "0 through 9. It keeps only the even ones (numbers that split into "
            "2 with nothing left over). For each kept number, it multiplies the "
            "number by itself. So you get [0, 4, 16, 36, 64] — the even numbers "
            "0, 2, 4, 6, 8, each times itself."
        ),
    },
    {
        "code": "result = {k: v for k, v in zip(keys, values)}",
        "explanation": (
            "This builds a dictionary by pairing two lists together. A "
            "dictionary stores things as label-and-value pairs. It takes the "
            "first item from `keys` and the first from `values` and pairs them, "
            "then the second of each, and so on (`zip` does this pairing). So "
            "each key ends up linked to its matching value."
        ),
    },
    {
        "code": "with open('data.txt', 'r', encoding='utf-8') as f:\n"
                "    lines = f.readlines()",
        "explanation": (
            "This opens a text file so the program can read it. The word `with` "
            "is helpful: it closes the file for you when you're done, even if "
            "something goes wrong. `readlines()` reads the whole file and gives "
            "you back a list, where each line of the file becomes one item in "
            "that list. So you end up with all the file's lines, ready to use."
        ),
    },
    {
        "code": "matrix = [[1, 2], [3, 4], [5, 6]]\n"
                "transposed = list(zip(*matrix))",
        "explanation": (
            "This flips a grid on its side. The grid is rows of numbers: "
            "[1, 2], [3, 4], [5, 6]. The code turns the rows into columns: it "
            "takes the first number of every row together, then the second of "
            "every row. So you get [(1, 3, 5), (2, 4, 6)]. What used to be the "
            "columns are now the rows."
        ),
    },
    {
        "code": "words = ['apple', 'banana', 'cherry']\n"
                "for i, word in enumerate(words, start=1):\n"
                "    print(f'{i}. {word}')",
        "explanation": (
            "This prints a list of words as a numbered list, like '1. apple', "
            "'2. banana'. `enumerate` is the helper: it gives you both the word "
            "and its position number at the same time. `start=1` means the "
            "counting begins at 1 instead of the usual 0. So you don't have to "
            "keep track of the number yourself."
        ),
    },
    {
        "code": "from collections import defaultdict\n\n"
                "counts = defaultdict(int)\n"
                "for ch in 'mississippi':\n"
                "    counts[ch] += 1",
        "explanation": (
            "This counts how many times each letter shows up in the word "
            "'mississippi'. It uses a special dictionary that automatically "
            "starts any brand-new letter at 0. So you can just add 1 each time "
            "you see a letter, without first checking whether it's there. After "
            "it runs you get m:1, i:4, s:4, p:2."
        ),
    },
    {
        "code": "even, odd = [], []\nfor n in range(10):\n"
                "    (even if n % 2 == 0 else odd).append(n)",
        "explanation": (
            "This splits the numbers 0 to 9 into two lists: evens and odds. For "
            "each number it picks a list — the `even` list if the number splits "
            "into 2 with nothing left over, otherwise the `odd` list — and then "
            "adds the number to that list. So `even` ends up [0, 2, 4, 6, 8] "
            "and `odd` ends up [1, 3, 5, 7, 9]."
        ),
    },
    {
        "code": "import json\n\n"
                "with open('config.json') as f:\n"
                "    cfg = json.load(f)\n"
                "name = cfg.get('name', 'default')",
        "explanation": (
            "This reads a settings file and looks something up in it safely. "
            "The file is written in JSON, a common text format for storing "
            "data. `json.load` reads it and turns it into a dictionary "
            "(label-and-value pairs). Then `.get('name', 'default')` looks up "
            "'name' — but if 'name' isn't there, it gives back 'default' instead "
            "of crashing."
        ),
    },
    {
        "code": "total = sum(int(x) for x in input().split())",
        "explanation": (
            "This adds up a list of numbers you type on one line. It reads your "
            "line, splits it into pieces wherever there's a space, turns each "
            "piece into a whole number, and adds them all together. So if you "
            "type '3 5 7', it gives 15."
        ),
    },
    {
        "code": "flat = [x for row in grid for x in row]",
        "explanation": (
            "This flattens a list of lists into one flat list. The `grid` is "
            "rows of items. The code goes through each row, then through each "
            "item inside that row, and collects them all into a single list. So "
            "[[1, 2], [3, 4]] becomes [1, 2, 3, 4]."
        ),
    },

    # ---------------- Functions, generators, classes ----------------
    {
        "code": "def countdown(n):\n    while n > 0:\n        yield n\n        n -= 1",
        "explanation": (
            "This makes numbers count down, one at a time. It's a 'generator', "
            "which means it hands you one value, pauses, and waits until you "
            "ask for the next one. It gives you `n`, then `n` minus 1, and so "
            "on, down to 1. Because it makes numbers one by one instead of all "
            "at once, it barely uses any memory. So countdown(3) gives you 3, "
            "then 2, then 1."
        ),
    },
    {
        "code": "def chunks(lst, size):\n"
                "    for i in range(0, len(lst), size):\n"
                "        yield lst[i:i + size]",
        "explanation": (
            "This cuts a long list into smaller pieces of a size you choose. "
            "It's a 'generator', so it hands you one piece at a time instead of "
            "all at once. It walks through the list in steps and slices off "
            "that many items each time. If the list doesn't divide evenly, the "
            "last piece is just whatever is left over. So a list of 10 cut by 3 "
            "gives pieces of 3, 3, 3, and 1."
        ),
    },
    {
        "code": "from dataclasses import dataclass\n\n"
                "@dataclass\nclass Point:\n    x: int\n    y: int = 0",
        "explanation": (
            "This makes a simple container for two pieces of information: `x` "
            "and `y`. A 'class' is a blueprint for making objects that hold "
            "data. The `@dataclass` line on top is a helper that writes the "
            "boring setup code for you automatically. `x` is required, but `y` "
            "is optional and becomes 0 if you don't give it. So Point(1) and "
            "Point(1, 2) both work."
        ),
    },
    {
        "code": "class Dog:\n    def __init__(self, name):\n"
                "        self.name = name\n\n    def bark(self):\n"
                "        return f'{self.name} says woof!'",
        "explanation": (
            "This makes a blueprint for dogs. A 'class' is a blueprint for "
            "building objects that hold data and can do actions. Each dog "
            "stores a name. The setup part (`__init__`) saves the name when you "
            "make a new dog. The `bark` part is an action the dog can do — it "
            "gives back a sentence. So Dog('Rex').bark() gives 'Rex says woof!'."
        ),
    },
    {
        "code": "def greet(name: str, *, greeting: str = 'Hello') -> str:\n"
                "    return f'{greeting}, {name}!'",
        "explanation": (
            "This makes a greeting sentence. You give it a `name`. The greeting "
            "word is optional and becomes 'Hello' if you don't choose one. The "
            "`*` is a rule: if you want a different greeting, you must name it, "
            "like greeting='Hi'. So greet('Ada') gives 'Hello, Ada!' and "
            "greet('Ada', greeting='Hi') gives 'Hi, Ada!'."
        ),
    },
    {
        "code": "nums = [3, 1, 4, 1, 5, 9, 2, 6]\nnums.sort(reverse=True)",
        "explanation": (
            "This puts a list of numbers in order from biggest to smallest. "
            "`reverse=True` is what makes it go big-to-small instead of the "
            "usual small-to-big. Important: it changes the original list "
            "itself, it doesn't make a new one. So the list becomes [9, 6, 5, "
            "4, 3, 2, 1, 1]. To keep the original safe, you'd use `sorted()` "
            "instead."
        ),
    },
    {
        "code": "def safe_divide(a, b):\n    return a / b if b else float('inf')",
        "explanation": (
            "This divides one number by another without crashing on zero. "
            "Normally, dividing by zero stops the program with an error. Here "
            "it checks first: if the bottom number is zero, it gives back "
            "infinity instead of crashing. Otherwise it does normal division. "
            "So it's a safe divide that never breaks."
        ),
    },

    # ---------------- Hard / advanced ----------------
    {
        "code": "def memoize(func):\n"
                "    cache = {}\n"
                "    def wrapper(*args):\n"
                "        if args not in cache:\n"
                "            cache[args] = func(*args)\n"
                "        return cache[args]\n"
                "    return wrapper",
        "explanation": (
            "This adds a memory to a function so it doesn't repeat work. A "
            "'decorator' is extra code you stick on top of a function to give "
            "it a new power. The new power here is remembering. The first time "
            "you call the function with some input, it does the work and saves "
            "the answer in a hidden note (the `cache`). Next time you ask the "
            "same thing, it just reads the saved answer instead of doing the "
            "work again. So repeated calls become very fast."
        ),
    },
    {
        "code": "import functools\n\n"
                "@functools.lru_cache(maxsize=None)\n"
                "def fib(n):\n"
                "    return n if n < 2 else fib(n - 1) + fib(n - 2)",
        "explanation": (
            "This finds Fibonacci numbers — a list of numbers where each one is "
            "the two before it added together (0, 1, 1, 2, 3, 5, 8, and so on). "
            "The function calls itself to add up the two earlier numbers; a "
            "function calling itself is called 'recursion'. Normally this is "
            "slow because it redoes the same work again and again. The "
            "`@lru_cache` line fixes that by remembering answers it already "
            "found, so each one is worked out only once."
        ),
    },
    {
        "code": "def repeat(times):\n"
                "    def decorator(func):\n"
                "        def wrapper(*args, **kwargs):\n"
                "            for _ in range(times):\n"
                "                result = func(*args, **kwargs)\n"
                "            return result\n"
                "        return wrapper\n"
                "    return decorator",
        "explanation": (
            "This is a 'decorator' you can set up with a number. A decorator is "
            "extra code you attach to a function to change how it behaves. This "
            "one lets you say how many times to run the function. You write "
            "@repeat(3) on top of a function, and now every time you call that "
            "function it actually runs three times in a row. The last result is "
            "what you get back."
        ),
    },
    {
        "code": "def log(*args, **kwargs):\n    print(args)\n    print(kwargs)",
        "explanation": (
            "This shows two ways a function can take in any amount of input. "
            "`*args` collects any number of plain values into one group. "
            "`**kwargs` collects any number of named values (like color='red') "
            "into a dictionary (label-and-value pairs). Together they let one "
            "function accept almost anything. So log(1, 2, color='red') would "
            "show (1, 2) and then {'color': 'red'}."
        ),
    },
    {
        "code": "import time\n\nclass Timer:\n"
                "    def __enter__(self):\n"
                "        self.start = time.time()\n"
                "        return self\n"
                "    def __exit__(self, *exc):\n"
                "        print(time.time() - self.start)",
        "explanation": (
            "This makes your own tool for Python's `with` block. A `with` block "
            "runs some setup when you enter it and some cleanup when you leave "
            "it. The `__enter__` part runs at the start — here it notes the "
            "current time. The `__exit__` part runs automatically at the end, no "
            "matter what — here it prints how long the block took. So you can "
            "wrap any code to measure how long it runs."
        ),
    },
    {
        "code": "class Circle:\n    def __init__(self, r):\n        self._r = r\n\n"
                "    @property\n    def area(self):\n"
                "        return 3.14159 * self._r ** 2",
        "explanation": (
            "This makes a value that's worked out on the spot but used like a "
            "stored fact. The `Circle` keeps a radius. The `@property` line "
            "lets you write circle.area (with no parentheses), even though it's "
            "secretly doing a calculation each time (radius times radius times "
            "pi). So the area is always correct for the current radius, and you "
            "don't have to call it like a function."
        ),
    },
    {
        "code": "class Animal:\n    def __init__(self, name):\n"
                "        self.name = name\n\n"
                "class Cat(Animal):\n    def __init__(self, name, color):\n"
                "        super().__init__(name)\n        self.color = color",
        "explanation": (
            "This builds one blueprint on top of another so it can reuse code. "
            "`Animal` is the basic blueprint and stores a name. `Cat` is a kind "
            "of `Animal`, so it gets everything an animal has. In the cat's "
            "setup, super().__init__(name) calls the animal's setup to handle "
            "the name (so we don't rewrite it), and then the cat adds its own "
            "extra piece of data: a color."
        ),
    },
    {
        "code": "class TooBigError(Exception):\n    pass\n\n"
                "def check(n):\n    if n > 100:\n"
                "        raise TooBigError('too big')",
        "explanation": (
            "This makes your own kind of error and triggers it on purpose. The "
            "first part defines a new error type called TooBigError. In the "
            "function, if the number is over 100, `raise` fires that error with "
            "a message, which stops the program and signals that something went "
            "wrong. Custom errors make it clear exactly what the problem was."
        ),
    },
    {
        "code": "def make_counter():\n"
                "    count = 0\n"
                "    def increment():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "        return count\n"
                "    return increment",
        "explanation": (
            "This makes a counter that remembers its number. First you call "
            "make_counter() to set one up; it starts at 0. After that, each "
            "time you use it, it adds 1 and gives you the new number: first 1, "
            "then 2, then 3. The number is kept hidden inside — nothing else in "
            "your program can touch or change it. (This trick is called a "
            "'closure'.)"
        ),
    },
    {
        "code": "async def fetch(url):\n"
                "    async with session.get(url) as resp:\n"
                "        return await resp.json()",
        "explanation": (
            "This is an 'async' function, built for slow jobs like grabbing "
            "data from the internet, without freezing the rest of the program. "
            "The word `await` marks the spots where it waits for something "
            "slow; while it waits, the program is free to do other things "
            "instead of sitting idle. Here it sends a web request, waits for "
            "the reply, and gives back the reply's data."
        ),
    },
    {
        "code": "try:\n    value = int(user_input)\nexcept ValueError:\n"
                "    value = 0\nfinally:\n    print('done')",
        "explanation": (
            "This safely tries to turn what the user typed into a whole number. "
            "The `try` part attempts it. If they typed something that isn't a "
            "number, it would normally crash — but the `except` part catches "
            "that and uses 0 instead. The `finally` part runs no matter what, "
            "so 'done' always prints. So bad input never crashes the program; "
            "it just becomes 0."
        ),
    },
    {
        "code": "people.sort(key=lambda p: p['age'])",
        "explanation": (
            "This sorts a list of people by their age. `sort` puts the list in "
            "order. The `key=` part says what to sort by. `lambda p: p['age']` "
            "is a tiny one-line function with no name (called a 'lambda') that "
            "pulls out a person's age. So the people get arranged from youngest "
            "to oldest."
        ),
    },
    {
        "code": "while (line := input()) != 'quit':\n    print(line)",
        "explanation": (
            "This uses the `:=` symbol, nicknamed the 'walrus', which grabs a "
            "value and saves it at the same time. Each time through the loop it "
            "reads a line you type, saves it as `line`, and checks whether it "
            "equals 'quit' — all in one step. If it's not 'quit', it prints the "
            "line and loops again. Type 'quit' and the loop stops."
        ),
    },
    {
        "code": "from functools import reduce\n\n"
                "total = reduce(lambda a, b: a + b, numbers)",
        "explanation": (
            "This squashes a whole list down to a single value by combining "
            "items two at a time. The little function `lambda a, b: a + b` says "
            "how to combine two values — here, add them. It adds the first two, "
            "then adds the next one to that running total, and keeps going. So "
            "[1, 2, 3, 4] becomes 1 + 2 + 3 + 4 = 10."
        ),
    },
    {
        "code": "result = {k: v for k, v in data.items() if v > 0}",
        "explanation": (
            "This makes a new dictionary by filtering an old one. It goes "
            "through every label-and-value pair in `data` (`.items()` gives you "
            "the pairs). It keeps only the pairs whose value is bigger than 0. "
            "Each kept pair goes into the new dictionary. So it's a quick way to "
            "make a copy with the zero or negative entries removed."
        ),
    },

    # ---------------- Multi-part scripts (explained part by part) ----------------
    {
        "code": "numbers = [1, 2, 3, 4]\ntotal = 0\nfor n in numbers:\n"
                "    total += n\nprint(total)",
        "explanation": (
            "Overall, this adds up a list of numbers and prints the total. Now "
            "the parts: `numbers` is a list holding 1, 2, 3, 4. `total = 0` "
            "makes a variable to hold the running sum, starting at 0. The `for` "
            "loop goes through each number one at a time, and `total += n` adds "
            "that number onto the running total. Finally, `print(total)` shows "
            "the result on the screen. So it prints 10."
        ),
    },
    {
        "code": "def square(x):\n    return x * x\n\n"
                "result = square(5)\nprint(f'The answer is {result}')",
        "explanation": (
            "Overall, this defines a function, uses it on the number 5, and "
            "prints the result. Now the parts: `def square(x)` makes a function "
            "called `square` that takes one number and gives back that number "
            "times itself. `result = square(5)` runs the function with 5, so "
            "`result` becomes 25. The last line prints a sentence with the "
            "answer dropped into it using an f-string (text with a variable "
            "inside the braces). So it prints 'The answer is 25'."
        ),
    },
    {
        "code": "names = ['Sam', 'Ada', 'Lee']\n"
                "for i, name in enumerate(names, start=1):\n"
                "    print(f'{i}: {name}')",
        "explanation": (
            "Overall, this prints a numbered list of names. Now the parts: "
            "`names` is a list of three names. The `for` loop goes through them, "
            "and `enumerate(names, start=1)` gives both the name and a position "
            "number that starts at 1. Inside the loop, the f-string prints the "
            "number and the name together. So it prints '1: Sam', '2: Ada', "
            "'3: Lee', one per line."
        ),
    },
    {
        "code": "text = input('Enter a word: ')\n"
                "if len(text) > 5:\n    print('long word')\n"
                "else:\n    print('short word')",
        "explanation": (
            "Overall, this asks the user for a word and says whether it is long "
            "or short. Now the parts: `input(...)` shows a message and waits for "
            "the user to type a word, which is saved in `text`. `len(text)` "
            "counts how many letters the word has. The `if` checks whether that "
            "count is more than 5: if yes it prints 'long word'; otherwise the "
            "`else` part prints 'short word'."
        ),
    },
    {
        "code": "class Counter:\n    def __init__(self):\n"
                "        self.count = 0\n\n    def add(self):\n"
                "        self.count += 1\n\n"
                "c = Counter()\nc.add()\nc.add()\nprint(c.count)",
        "explanation": (
            "Overall, this makes a simple counter, adds to it twice, and prints "
            "the result. Now the parts: `class Counter` is a blueprint for a "
            "counter object. Its setup (`__init__`) starts `count` at 0. The "
            "`add` action increases `count` by 1 each time it runs. Then "
            "`c = Counter()` makes one counter, `c.add()` is called twice (so "
            "count goes from 0 to 1 to 2), and `print(c.count)` shows the final "
            "number. So it prints 2."
        ),
    },
    {
        "code": "prices = [10, 20, 30]\ntotal = sum(prices)\n"
                "average = total / len(prices)\n"
                "print(f'Total: {total}, Average: {average}')",
        "explanation": (
            "Overall, this works out the total and average of a list of prices "
            "and prints them. Now the parts: `prices` is a list of three "
            "numbers. `sum(prices)` adds them all into `total` (60). "
            "`len(prices)` counts how many prices there are (3), and dividing "
            "the total by that count gives the `average` (20). The last line "
            "prints both numbers in a sentence using an f-string. So it prints "
            "'Total: 60, Average: 20.0'."
        ),
    },
]


def main() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Basic validation: every pair must have non-empty code and explanation.
    for i, pair in enumerate(CODE_EXPLANATION_PAIRS):
        assert pair["code"].strip(), f"Empty code in pair {i}"
        assert pair["explanation"].strip(), f"Empty explanation in pair {i}"

    with open(config.RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(CODE_EXPLANATION_PAIRS, f, indent=2, ensure_ascii=False)

    print(f"[collect_data] Wrote {len(CODE_EXPLANATION_PAIRS)} "
          f"code-explanation pairs to {config.RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
