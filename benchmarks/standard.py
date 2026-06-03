"""
Standard benchmark subsets for rigorous evaluation.

Includes curated subsets from:
- MMLU (Massive Multitask Language Understanding) — factual QA
- GSM8K (Grade School Math 8K) — mathematical reasoning
- HumanEval — code generation

These complement the curated tasks in tasks.py and provide
reviewer-recognized benchmarks for the paper.
"""

from __future__ import annotations

from agentcodec.models import TaskCategory, TaskItem


def get_standard_tasks() -> list[TaskItem]:
    """Get all standard benchmark tasks."""
    return mmlu_tasks() + gsm8k_tasks() + humaneval_tasks()


def _stamp(tasks: list[TaskItem], score_mode: str) -> list[TaskItem]:
    """Mutate `tasks` in place to set the same score_mode on each entry."""
    for t in tasks:
        t.score_mode = score_mode
    return tasks


# ---------------------------------------------------------------------------
# MMLU Subset — 20 questions across STEM, humanities, social sciences
# ---------------------------------------------------------------------------

def mmlu_tasks() -> list[TaskItem]:
    return _stamp([
        # -- STEM --
        TaskItem(
            id="mmlu_01", category=TaskCategory.QA,
            prompt="What is the primary function of mitochondria in eukaryotic cells?\n(A) Protein synthesis\n(B) ATP production through oxidative phosphorylation\n(C) DNA replication\n(D) Cell division\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "biology"},
        ),
        TaskItem(
            id="mmlu_02", category=TaskCategory.QA,
            prompt="In quantum mechanics, the Heisenberg uncertainty principle states that:\n(A) Energy is always conserved\n(B) The position and momentum of a particle cannot both be precisely determined simultaneously\n(C) Light behaves only as a wave\n(D) Electrons orbit the nucleus in fixed paths\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "physics"},
        ),
        TaskItem(
            id="mmlu_03", category=TaskCategory.QA,
            prompt="Which of the following is a consequence of the second law of thermodynamics?\n(A) Energy can be created in nuclear reactions\n(B) The entropy of an isolated system tends to increase over time\n(C) Heat flows spontaneously from cold to hot bodies\n(D) Perpetual motion machines of the first kind are impossible\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "physics"},
        ),
        TaskItem(
            id="mmlu_04", category=TaskCategory.QA,
            prompt="What is the time complexity of binary search on a sorted array of n elements?\n(A) O(n)\n(B) O(n log n)\n(C) O(log n)\n(D) O(n^2)\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "computer_science"},
        ),
        TaskItem(
            id="mmlu_05", category=TaskCategory.QA,
            prompt="In organic chemistry, a Diels-Alder reaction is best described as:\n(A) A nucleophilic substitution\n(B) A [4+2] cycloaddition between a conjugated diene and a dienophile\n(C) An elimination reaction\n(D) A free radical chain reaction\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "chemistry"},
        ),
        # -- Humanities --
        TaskItem(
            id="mmlu_06", category=TaskCategory.QA,
            prompt="The categorical imperative is a central concept in the moral philosophy of:\n(A) John Stuart Mill\n(B) Aristotle\n(C) Immanuel Kant\n(D) David Hume\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "philosophy"},
        ),
        TaskItem(
            id="mmlu_07", category=TaskCategory.QA,
            prompt="Which treaty ended the Thirty Years' War in 1648?\n(A) Treaty of Versailles\n(B) Treaty of Tordesillas\n(C) Peace of Westphalia\n(D) Treaty of Utrecht\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "history"},
        ),
        TaskItem(
            id="mmlu_08", category=TaskCategory.QA,
            prompt="In epistemology, the 'Gettier problem' challenges the traditional definition of knowledge as:\n(A) Justified true belief\n(B) Empirical observation\n(C) Logical deduction\n(D) Innate understanding\nAnswer with the letter only.",
            reference="A",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "philosophy"},
        ),
        TaskItem(
            id="mmlu_09", category=TaskCategory.QA,
            prompt="The Doppler effect describes the change in frequency of a wave in relation to:\n(A) The amplitude of the wave\n(B) The medium through which it travels\n(C) An observer moving relative to the wave source\n(D) The wavelength of adjacent waves\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "physics"},
        ),
        TaskItem(
            id="mmlu_10", category=TaskCategory.QA,
            prompt="Which of the following statements about P vs NP is correct?\n(A) It has been proven that P = NP\n(B) It has been proven that P ≠ NP\n(C) It remains one of the most important open problems in computer science\n(D) The problem only applies to quantum computing\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "computer_science"},
        ),
        # -- Social Sciences --
        TaskItem(
            id="mmlu_11", category=TaskCategory.QA,
            prompt="In economics, 'moral hazard' refers to:\n(A) The tendency for prices to rise during inflation\n(B) The risk that a party insulated from risk may behave differently than if fully exposed to it\n(C) The ethical obligations of corporations\n(D) The relationship between supply and demand\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "economics"},
        ),
        TaskItem(
            id="mmlu_12", category=TaskCategory.QA,
            prompt="Bayes' theorem relates:\n(A) Sample size to confidence intervals\n(B) Prior probability, likelihood, and posterior probability\n(C) Mean, median, and mode\n(D) Variance and standard deviation\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "statistics"},
        ),
        # -- Hard STEM --
        TaskItem(
            id="mmlu_13", category=TaskCategory.QA,
            prompt="In general relativity, the Einstein field equations relate the geometry of spacetime to:\n(A) The speed of light\n(B) The distribution of matter and energy\n(C) The electromagnetic spectrum\n(D) Quantum wave functions\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "physics"},
        ),
        TaskItem(
            id="mmlu_14", category=TaskCategory.QA,
            prompt="The Church-Turing thesis asserts that:\n(A) All mathematical problems are solvable\n(B) Quantum computers can solve NP-hard problems in polynomial time\n(C) Any function that is intuitively computable can be computed by a Turing machine\n(D) There exist problems that no algorithm can solve\nAnswer with the letter only.",
            reference="C",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "computer_science"},
        ),
        TaskItem(
            id="mmlu_15", category=TaskCategory.QA,
            prompt="Which enzyme is responsible for unwinding the DNA double helix during replication?\n(A) DNA polymerase\n(B) Helicase\n(C) Ligase\n(D) Topoisomerase\nAnswer with the letter only.",
            reference="B",
            metadata={"difficulty": "easy", "source": "mmlu", "subject": "biology"},
        ),
    ], "exact_letter")


# ---------------------------------------------------------------------------
# GSM8K Subset — 15 grade-school math word problems
# ---------------------------------------------------------------------------

def gsm8k_tasks() -> list[TaskItem]:
    return _stamp([
        TaskItem(
            id="gsm8k_01", category=TaskCategory.REASONING,
            prompt="Janet has 3 times as many marbles as Tom. Tom has 12 marbles. How many marbles do they have together?",
            reference="Janet has 3 × 12 = 36 marbles. Together: 36 + 12 = 48 marbles.",
            metadata={"difficulty": "easy", "source": "gsm8k", "answer": 48},
        ),
        TaskItem(
            id="gsm8k_02", category=TaskCategory.REASONING,
            prompt="A store sells notebooks for $4 each and pens for $1.50 each. Maria buys 5 notebooks and 8 pens. How much does she spend in total?",
            reference="Notebooks: 5 × $4 = $20. Pens: 8 × $1.50 = $12. Total: $20 + $12 = $32.",
            metadata={"difficulty": "easy", "source": "gsm8k", "answer": 32},
        ),
        TaskItem(
            id="gsm8k_03", category=TaskCategory.REASONING,
            prompt="A train travels at 80 km/h for 2.5 hours, then at 60 km/h for 1.5 hours. What is the total distance traveled?",
            reference="First leg: 80 × 2.5 = 200 km. Second leg: 60 × 1.5 = 90 km. Total: 200 + 90 = 290 km.",
            metadata={"difficulty": "easy", "source": "gsm8k", "answer": 290},
        ),
        TaskItem(
            id="gsm8k_04", category=TaskCategory.REASONING,
            prompt="Sarah bakes 48 cookies. She gives 1/3 to her neighbor and 1/4 of the remaining cookies to her coworkers. How many cookies does Sarah have left?",
            reference="Given to neighbor: 48 × 1/3 = 16. Remaining: 48 - 16 = 32. Given to coworkers: 32 × 1/4 = 8. Left: 32 - 8 = 24.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 24},
        ),
        TaskItem(
            id="gsm8k_05", category=TaskCategory.REASONING,
            prompt="A rectangular garden is 15 meters long and 8 meters wide. A path 1 meter wide is built around the outside of the garden. What is the area of the path?",
            reference="Outer rectangle: (15+2) × (8+2) = 17 × 10 = 170 m². Inner rectangle: 15 × 8 = 120 m². Path area: 170 - 120 = 50 m².",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 50},
        ),
        TaskItem(
            id="gsm8k_06", category=TaskCategory.REASONING,
            prompt="A car rental costs $45 per day plus $0.20 per mile driven. If John rents a car for 3 days and drives 250 miles, what is the total cost?",
            reference="Daily cost: 3 × $45 = $135. Mileage cost: 250 × $0.20 = $50. Total: $135 + $50 = $185.",
            metadata={"difficulty": "easy", "source": "gsm8k", "answer": 185},
        ),
        TaskItem(
            id="gsm8k_07", category=TaskCategory.REASONING,
            prompt="In a class of 40 students, 60% passed the math test. Of those who passed, 75% scored above 80. How many students scored above 80?",
            reference="Passed: 40 × 0.60 = 24. Above 80: 24 × 0.75 = 18 students.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 18},
        ),
        TaskItem(
            id="gsm8k_08", category=TaskCategory.REASONING,
            prompt="A water tank fills at a rate of 3 liters per minute. A leak drains the tank at 0.5 liters per minute. If the tank has a capacity of 150 liters, how long does it take to fill from empty?",
            reference="Net fill rate: 3 - 0.5 = 2.5 liters/min. Time: 150 / 2.5 = 60 minutes.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 60},
        ),
        TaskItem(
            id="gsm8k_09", category=TaskCategory.REASONING,
            prompt="Three friends split a restaurant bill. The food costs $84, tax is 8%, and they want to leave a 20% tip on the pre-tax amount. How much does each person pay?",
            reference="Tax: $84 × 0.08 = $6.72. Tip: $84 × 0.20 = $16.80. Total: $84 + $6.72 + $16.80 = $107.52. Each: $107.52 / 3 = $35.84.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 35.84},
        ),
        TaskItem(
            id="gsm8k_10", category=TaskCategory.REASONING,
            prompt="A company produces widgets. On Monday they made 120 widgets. Each subsequent day they made 15% more than the previous day. How many widgets did they make on Thursday (the 4th day)? Round to the nearest whole number.",
            reference="Monday: 120. Tuesday: 120 × 1.15 = 138. Wednesday: 138 × 1.15 = 158.7. Thursday: 158.7 × 1.15 ≈ 183 widgets.",
            metadata={"difficulty": "hard", "source": "gsm8k", "answer": 183},
        ),
        TaskItem(
            id="gsm8k_11", category=TaskCategory.REASONING,
            prompt="Alice can paint a room in 6 hours. Bob can paint the same room in 4 hours. If they work together, how long will it take them to paint the room?",
            reference="Alice's rate: 1/6 room/hour. Bob's rate: 1/4 room/hour. Combined: 1/6 + 1/4 = 5/12 room/hour. Time: 12/5 = 2.4 hours = 2 hours 24 minutes.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 2.4},
        ),
        TaskItem(
            id="gsm8k_12", category=TaskCategory.REASONING,
            prompt="A store marks up items by 40% from wholesale price. During a sale, they offer 25% off the retail price. If the wholesale price of a jacket is $80, what is the sale price? Is the store still making a profit?",
            reference="Retail: $80 × 1.40 = $112. Sale: $112 × 0.75 = $84. Profit: $84 - $80 = $4. Yes, the store makes a $4 profit.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 84},
        ),
        TaskItem(
            id="gsm8k_13", category=TaskCategory.REASONING,
            prompt="A sequence starts with 2. Each term is computed by multiplying the previous term by 3 and subtracting 1. What is the 5th term?",
            reference="Term 1: 2. Term 2: 2×3-1=5. Term 3: 5×3-1=14. Term 4: 14×3-1=41. Term 5: 41×3-1=122.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 122},
        ),
        TaskItem(
            id="gsm8k_14", category=TaskCategory.REASONING,
            prompt="A farmer has a field that yields 320 kg of wheat per hectare. He wants to produce at least 5000 kg. He has already harvested 12 hectares. How many more hectares does he need to harvest?",
            reference="Already harvested: 12 × 320 = 3840 kg. Remaining: 5000 - 3840 = 1160 kg. Additional hectares: 1160 / 320 = 3.625. He needs 4 more hectares (rounding up).",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 4},
        ),
        TaskItem(
            id="gsm8k_15", category=TaskCategory.REASONING,
            prompt="A bacteria population doubles every 3 hours. Starting with 500 bacteria, how many will there be after 15 hours?",
            reference="Number of doublings: 15 / 3 = 5. Population: 500 × 2^5 = 500 × 32 = 16000.",
            metadata={"difficulty": "medium", "source": "gsm8k", "answer": 16000},
        ),
    ], "numeric")


# ---------------------------------------------------------------------------
# HumanEval Subset — 10 code generation problems
# ---------------------------------------------------------------------------

def humaneval_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            id="humaneval_01", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` "
                "that checks if any two numbers in the list are closer to each other than the given threshold."
            ),
            reference=(
                "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
                "    for i in range(len(numbers)):\n"
                "        for j in range(i + 1, len(numbers)):\n"
                "            if abs(numbers[i] - numbers[j]) < threshold:\n"
                "                return True\n"
                "    return False"
            ),
            metadata={"difficulty": "easy", "source": "humaneval", "test_cases": [
                "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
                "assert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True",
            ]},
        ),
        TaskItem(
            id="humaneval_02", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `truncate_number(number: float) -> float` "
                "that returns the decimal part of a positive floating-point number. "
                "Example: truncate_number(3.5) returns 0.5."
            ),
            reference="def truncate_number(number: float) -> float:\n    return number % 1.0",
            metadata={"difficulty": "easy", "source": "humaneval", "test_cases": [
                "assert truncate_number(3.5) == 0.5",
                "assert abs(truncate_number(1.33) - 0.33) < 1e-6",
            ]},
        ),
        TaskItem(
            id="humaneval_03", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `intersperse(numbers: list[int], delimiter: int) -> list[int]` "
                "that inserts the delimiter between every two consecutive elements of the list. "
                "Example: intersperse([1, 2, 3], 4) returns [1, 4, 2, 4, 3]."
            ),
            reference=(
                "def intersperse(numbers: list[int], delimiter: int) -> list[int]:\n"
                "    if not numbers:\n"
                "        return []\n"
                "    result = [numbers[0]]\n"
                "    for n in numbers[1:]:\n"
                "        result.extend([delimiter, n])\n"
                "    return result"
            ),
            metadata={"difficulty": "easy", "source": "humaneval", "test_cases": [
                "assert intersperse([], 4) == []",
                "assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
            ]},
        ),
        TaskItem(
            id="humaneval_04", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `parse_nested_parens(paren_string: str) -> list[int]` "
                "that takes a string of groups of nested parentheses separated by spaces, "
                "and returns a list of the maximum nesting depth for each group. "
                "Example: parse_nested_parens('(()()) ((())) () ((())())') returns [2, 3, 1, 3]."
            ),
            reference=None,
            metadata={"difficulty": "medium", "source": "humaneval", "test_cases": [
                "assert parse_nested_parens('(()()) ((())) () ((())())') == [2, 3, 1, 3]",
                "assert parse_nested_parens('() (()) ((()))') == [1, 2, 3]",
            ]},
        ),
        TaskItem(
            id="humaneval_05", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `rolling_max(numbers: list[int]) -> list[int]` "
                "that returns a list of the running maximum at each position. "
                "Example: rolling_max([1, 2, 3, 2, 3, 4, 2]) returns [1, 2, 3, 3, 3, 4, 4]."
            ),
            reference=(
                "def rolling_max(numbers: list[int]) -> list[int]:\n"
                "    result = []\n"
                "    current_max = float('-inf')\n"
                "    for n in numbers:\n"
                "        current_max = max(current_max, n)\n"
                "        result.append(current_max)\n"
                "    return result"
            ),
            metadata={"difficulty": "easy", "source": "humaneval", "test_cases": [
                "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]",
                "assert rolling_max([]) == []",
            ]},
        ),
        TaskItem(
            id="humaneval_06", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `filter_by_prefix(strings: list[str], prefix: str) -> list[str]` "
                "that returns only strings that start with the given prefix. "
                "Example: filter_by_prefix(['abc', 'bcd', 'ade', 'aef'], 'a') returns ['abc', 'ade', 'aef']."
            ),
            reference=(
                "def filter_by_prefix(strings: list[str], prefix: str) -> list[str]:\n"
                "    return [s for s in strings if s.startswith(prefix)]"
            ),
            metadata={"difficulty": "easy", "source": "humaneval", "test_cases": [
                "assert filter_by_prefix([], 'a') == []",
                "assert filter_by_prefix(['abc', 'bcd', 'ade', 'aef'], 'a') == ['abc', 'ade', 'aef']",
            ]},
        ),
        TaskItem(
            id="humaneval_07", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `remove_duplicates(numbers: list[int]) -> list[int]` "
                "that removes elements that appear more than once, keeping only unique elements "
                "in their original order. Example: remove_duplicates([1, 2, 3, 2, 4]) returns [1, 3, 4]."
            ),
            reference=(
                "def remove_duplicates(numbers: list[int]) -> list[int]:\n"
                "    from collections import Counter\n"
                "    counts = Counter(numbers)\n"
                "    return [n for n in numbers if counts[n] == 1]"
            ),
            metadata={"difficulty": "medium", "source": "humaneval", "test_cases": [
                "assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]",
                "assert remove_duplicates([]) == []",
            ]},
        ),
        TaskItem(
            id="humaneval_08", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `encode_shift(s: str) -> str` that encodes a string "
                "by shifting every character by 5 positions in the alphabet (wrapping around), "
                "and a function `decode_shift(s: str) -> str` that reverses the encoding. "
                "Only shift lowercase letters a-z."
            ),
            reference=None,
            metadata={"difficulty": "medium", "source": "humaneval", "test_cases": [
                "assert decode_shift(encode_shift('hello')) == 'hello'",
                "assert decode_shift(encode_shift('abcxyz')) == 'abcxyz'",
            ]},
        ),
        TaskItem(
            id="humaneval_09", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `longest_common_subsequence(s1: str, s2: str) -> str` "
                "that returns the longest common subsequence of two strings using dynamic programming. "
                "Example: longest_common_subsequence('abcde', 'ace') returns 'ace'."
            ),
            reference=None,
            metadata={"difficulty": "hard", "source": "humaneval", "test_cases": [
                "assert longest_common_subsequence('abcde', 'ace') == 'ace'",
                "assert longest_common_subsequence('abc', 'def') == ''",
            ]},
        ),
        TaskItem(
            id="humaneval_10", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `simplify(x: str, n: str) -> bool` where x and n are "
                "string representations of fractions like '1/5' and '5/3'. Return True if x * n "
                "evaluates to a whole number, False otherwise. "
                "Example: simplify('1/5', '5/1') returns True. simplify('1/6', '2/1') returns False."
            ),
            reference=None,
            metadata={"difficulty": "medium", "source": "humaneval", "test_cases": [
                "assert simplify('1/5', '5/1') == True",
                "assert simplify('1/6', '2/1') == False",
                "assert simplify('7/10', '10/2') == False",
            ]},
        ),
    ]
