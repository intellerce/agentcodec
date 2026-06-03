"""
Benchmark tasks for evaluating communication-theoretic reliability techniques.

4 categories × ~10 base tasks + ~10 hard tasks each.
Tasks span easy → very hard to observe how techniques behave across difficulty.

The hard tasks are specifically designed to stress small (7-8B) models:
- Multi-step reasoning chains where one wrong step cascades
- Ambiguous or adversarial prompts that invite different interpretations
- Problems requiring precise numerical computation
- Tasks where models commonly hallucinate or give partial answers

For the paper, these are supplemented by standard benchmark subsets (MMLU, GSM8K,
HumanEval) but these curated tasks allow controlled evaluation.
"""

from __future__ import annotations

from agentcodec.models import TaskCategory, TaskItem


def get_all_tasks(
    include_standard: bool = True,
    use_downloaded: bool = False,
    downloaded_n: int | None = None,
    data_dir: str | None = None,
    include_hard: bool = True,
    include_curated: bool = True,
    min_difficulty: str | None = None,
) -> list[TaskItem]:
    """
    Get all benchmark tasks across all categories.

    Args:
        include_standard: include hardcoded standard benchmark samples (MMLU/GSM8K/HumanEval)
        use_downloaded: use full downloaded datasets instead of hardcoded samples.
                        Run `python benchmarks/download_datasets.py` first.
        downloaded_n: max tasks per downloaded dataset (None = all)
        data_dir: directory containing downloaded JSON files (default: benchmarks/data/)
        include_hard: include hard/adversarial tasks designed to stress small models
                      (only meaningful when include_curated=True)
        include_curated: include the curated in-repo task lists
                         (qa/reasoning/creative/code + hard_* + extreme_*).
                         Set False to evaluate purely on downloaded datasets.
        min_difficulty: filter tasks by minimum difficulty level.
                        Options: "easy", "medium", "hard", "very_hard", "extreme".
                        e.g., min_difficulty="hard" returns only hard, very_hard, and extreme tasks.
    """
    tasks: list[TaskItem] = []
    if include_curated:
        tasks += qa_tasks() + reasoning_tasks() + creative_tasks() + code_tasks()
        if include_hard:
            tasks += hard_qa_tasks() + hard_reasoning_tasks() + hard_creative_tasks() + hard_code_tasks()
        tasks += extreme_tasks()

    if use_downloaded:
        from benchmarks.download_datasets import load_dataset_tasks
        kwargs = {}
        if data_dir:
            kwargs["data_dir"] = data_dir
        for ds in ["mmlu", "gsm8k", "humaneval"]:
            try:
                tasks += load_dataset_tasks(ds, n=downloaded_n, **kwargs)
            except FileNotFoundError:
                import logging
                logging.getLogger(__name__).warning(
                    f"Downloaded dataset '{ds}' not found. "
                    f"Run: python benchmarks/download_datasets.py --datasets {ds}"
                )
    elif include_standard:
        from benchmarks.standard import get_standard_tasks
        tasks += get_standard_tasks()

    # Apply difficulty filter
    if min_difficulty:
        difficulty_order = ["easy", "medium", "hard", "very_hard", "extreme"]
        if min_difficulty in difficulty_order:
            min_idx = difficulty_order.index(min_difficulty)
            allowed = set(difficulty_order[min_idx:])
            tasks = [t for t in tasks if t.metadata.get("difficulty", "medium") in allowed]

    return tasks


def get_tasks_by_category(category: TaskCategory) -> list[TaskItem]:
    loaders = {
        TaskCategory.QA: qa_tasks,
        TaskCategory.REASONING: reasoning_tasks,
        TaskCategory.CREATIVE: creative_tasks,
        TaskCategory.CODE: code_tasks,
    }
    return loaders[category]()


# ---------------------------------------------------------------------------
# Category 1: Factual QA — tests accuracy, hallucination resistance
# ---------------------------------------------------------------------------

def qa_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            id="qa_01", category=TaskCategory.QA,
            prompt="What is the speed of light in a vacuum, in meters per second?",
            reference="The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            metadata={"difficulty": "easy"},
        ),
        TaskItem(
            id="qa_02", category=TaskCategory.QA,
            prompt="Who won the Nobel Prize in Physics in 2023, and for what contribution?",
            reference="Pierre Agostini, Ferenc Krausz, and Anne L'Huillier won the 2023 Nobel Prize in Physics for experimental methods that generate attosecond pulses of light for the study of electron dynamics in matter.",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="qa_03", category=TaskCategory.QA,
            prompt="Explain the difference between Type I and Type II errors in hypothesis testing. Give a concrete medical example for each.",
            reference="Type I error (false positive): rejecting a true null hypothesis. Example: a diagnostic test says a healthy person has a disease. Type II error (false negative): failing to reject a false null hypothesis. Example: a diagnostic test says a sick person is healthy.",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="qa_04", category=TaskCategory.QA,
            prompt="What is the Banach-Tarski paradox? Explain it precisely, including what axiom it depends on.",
            reference="The Banach-Tarski paradox states that a solid ball in 3D space can be decomposed into finitely many disjoint subsets, which can then be reassembled using only rotations and translations into two solid balls identical to the original. It depends on the Axiom of Choice. The pieces are non-measurable sets.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="qa_05", category=TaskCategory.QA,
            prompt="What are the four fundamental forces of nature? For each, name the mediating boson and its approximate relative strength.",
            reference="1) Strong nuclear force, mediated by gluons, relative strength ~1. 2) Electromagnetic force, mediated by photons, relative strength ~1/137. 3) Weak nuclear force, mediated by W and Z bosons, relative strength ~10^-6. 4) Gravitational force, mediated by gravitons (hypothetical), relative strength ~10^-39.",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="qa_06", category=TaskCategory.QA,
            prompt="Explain the Byzantine Generals Problem and its relevance to distributed computing. What is the minimum number of generals needed to tolerate f traitors?",
            reference="The Byzantine Generals Problem describes a scenario where distributed processes must agree on a strategy despite some processes being faulty or malicious. To tolerate f Byzantine faults, you need at least 3f+1 total processes. It's fundamental to consensus protocols in distributed systems and blockchain.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="qa_07", category=TaskCategory.QA,
            prompt="What is the difference between CRISPR-Cas9 and base editing in gene therapy?",
            reference="CRISPR-Cas9 creates double-strand breaks in DNA and relies on the cell's repair mechanisms (NHEJ or HDR) to make changes, which can introduce insertions/deletions. Base editing uses a modified Cas9 (nickase) fused with a deaminase enzyme to directly convert one base pair to another without double-strand breaks, offering more precise single-nucleotide changes with fewer off-target effects.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="qa_08", category=TaskCategory.QA,
            prompt="What is the capital of Australia?",
            reference="Canberra",
            metadata={"difficulty": "easy"},
        ),
        TaskItem(
            id="qa_09", category=TaskCategory.QA,
            prompt="Explain the CAP theorem in distributed systems. Why can't you have all three properties simultaneously in a network partition?",
            reference="The CAP theorem (Brewer's theorem) states that a distributed system can provide at most two of three guarantees: Consistency (all nodes see the same data), Availability (every request gets a response), and Partition tolerance (system works despite network partitions). During a partition, you must choose: respond with potentially stale data (AP) or refuse requests until partition heals (CP). You can't guarantee both fresh data and responses when nodes can't communicate.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="qa_10", category=TaskCategory.QA,
            prompt="What is the Riemann Hypothesis? State it precisely and explain why it matters for prime number distribution.",
            reference="The Riemann Hypothesis conjectures that all non-trivial zeros of the Riemann zeta function ζ(s) have real part equal to 1/2. It matters because the distribution of prime numbers is intimately connected to the zeros of ζ(s). If true, it would give the best possible bounds on the error term in the Prime Number Theorem, meaning we'd know precisely how primes are distributed among integers.",
            metadata={"difficulty": "hard"},
        ),
    ]


# ---------------------------------------------------------------------------
# Category 2: Reasoning — tests logical/mathematical reasoning
# ---------------------------------------------------------------------------

def reasoning_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            id="reason_01", category=TaskCategory.REASONING,
            prompt="If a train travels 120 km in 2 hours, what is its average speed in km/h?",
            reference="60 km/h",
            metadata={"difficulty": "easy"},
        ),
        TaskItem(
            id="reason_02", category=TaskCategory.REASONING,
            prompt="A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
            reference="9 sheep. 'All but 9 die' means 9 survive.",
            metadata={"difficulty": "easy", "type": "trick_question"},
        ),
        TaskItem(
            id="reason_03", category=TaskCategory.REASONING,
            prompt=(
                "Alice, Bob, and Carol are sitting in a row. Alice is not next to Carol. "
                "Bob is to the right of Alice. What is the seating order from left to right?"
            ),
            reference="Alice, Bob, Carol. Alice is leftmost, Bob is in the middle (to the right of Alice), and Carol is on the far right (not next to Alice would also allow Carol, Bob, Alice but Bob must be right of Alice).",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="reason_04", category=TaskCategory.REASONING,
            prompt=(
                "A store sells apples for $1.50 each. If you buy 3 or more, you get a 20% discount "
                "on the total. How much do 5 apples cost?"
            ),
            reference="5 × $1.50 = $7.50 before discount. 20% discount: $7.50 × 0.80 = $6.00.",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="reason_05", category=TaskCategory.REASONING,
            prompt=(
                "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters of water? "
                "Show each step."
            ),
            reference="1) Fill 5L jug. 2) Pour from 5L into 3L jug until 3L is full (5L jug now has 2L). 3) Empty 3L jug. 4) Pour 2L from 5L jug into 3L jug. 5) Fill 5L jug again. 6) Pour from 5L into 3L jug until full (3L jug needs 1L more, so pour 1L). 7) 5L jug now has exactly 4L.",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="reason_06", category=TaskCategory.REASONING,
            prompt=(
                "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. Each label is WRONG. "
                "You can pick one fruit from one box. How do you determine what's in all three boxes?"
            ),
            reference="Pick from the box labeled 'Mixed'. Since all labels are wrong, 'Mixed' contains only one type. If you draw an apple, that box is 'Apples'. The box labeled 'Apples' can't be apples (wrong label) and can't be mixed (we found apples), so it's 'Oranges'. The remaining box labeled 'Oranges' is 'Mixed'.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="reason_07", category=TaskCategory.REASONING,
            prompt=(
                "Prove that the square root of 2 is irrational."
            ),
            reference="Proof by contradiction: Assume √2 = p/q where p,q are integers with no common factors. Then 2 = p²/q², so p² = 2q². This means p² is even, so p is even. Let p = 2k. Then 4k² = 2q², so q² = 2k², meaning q is also even. But this contradicts p and q having no common factors. Therefore √2 is irrational.",
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="reason_08", category=TaskCategory.REASONING,
            prompt=(
                "In a tournament of 8 players where every player plays every other player exactly once, "
                "how many total games are played? If each game takes 30 minutes and 4 games can run simultaneously, "
                "what is the minimum total time needed?"
            ),
            reference="Total games: C(8,2) = 28 games. With 4 simultaneous games: 28/4 = 7 rounds × 30 min = 210 minutes = 3.5 hours (assuming no scheduling constraints; actual minimum may differ due to player availability constraints).",
            metadata={"difficulty": "medium"},
        ),
        TaskItem(
            id="reason_09", category=TaskCategory.REASONING,
            prompt=(
                "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                "How much does the ball cost? Explain your reasoning carefully."
            ),
            reference="The ball costs $0.05. Let ball = x. Then bat = x + $1.00. Total: x + (x + $1.00) = $1.10, so 2x = $0.10, x = $0.05. The common wrong answer is $0.10 (which would make the bat $1.10 and total $1.20).",
            metadata={"difficulty": "easy", "type": "cognitive_bias"},
        ),
        TaskItem(
            id="reason_10", category=TaskCategory.REASONING,
            prompt=(
                "You have 12 balls, one of which is either heavier or lighter than the rest. "
                "Using a balance scale exactly 3 times, determine which ball is different and "
                "whether it is heavier or lighter. Describe the complete algorithm."
            ),
            reference="This is the classic 12-ball weighing puzzle. Divide into groups of 4. Weigh group A vs B. Based on result (balanced, A heavy, A light), proceed with specific weighings that progressively narrow down candidates while tracking whether the odd ball could be heavy or light. The complete algorithm has 27 possible outcomes (12 balls × 2 heavy/light + 1 all equal) fitting within 3^3 = 27 weighing outcomes.",
            metadata={"difficulty": "very_hard"},
        ),
    ]


# ---------------------------------------------------------------------------
# Category 3: Creative Writing — tests coherence, originality, style
# ---------------------------------------------------------------------------

def creative_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            id="creative_01", category=TaskCategory.CREATIVE,
            prompt="Write a haiku about recursion in programming.",
            reference=None,  # No single reference for creative tasks
            metadata={"difficulty": "easy", "criteria": "Must follow 5-7-5 syllable structure, be about recursion, and be clever/insightful."},
        ),
        TaskItem(
            id="creative_02", category=TaskCategory.CREATIVE,
            prompt="Write a 100-word story that begins and ends with the same sentence.",
            reference=None,
            metadata={"difficulty": "medium", "criteria": "Must be approximately 100 words, begin and end with the same sentence, have a coherent narrative arc, and be engaging."},
        ),
        TaskItem(
            id="creative_03", category=TaskCategory.CREATIVE,
            prompt=(
                "Explain quantum entanglement to a 10-year-old using only a story about "
                "two magical teddy bears."
            ),
            reference=None,
            metadata={"difficulty": "medium", "criteria": "Must accurately convey quantum entanglement concepts, be age-appropriate, use the teddy bear metaphor consistently, and be engaging for a child."},
        ),
        TaskItem(
            id="creative_04", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a professional email declining a job offer while maintaining a warm "
                "relationship with the employer. The candidate wants to keep the door open "
                "for future opportunities."
            ),
            reference=None,
            metadata={"difficulty": "medium", "criteria": "Professional tone, grateful, specific reason implied, door left open, appropriate length."},
        ),
        TaskItem(
            id="creative_05", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a 200-word product description for a fictional AI-powered toothbrush "
                "that analyzes brushing patterns. Make it compelling for a tech-savvy audience."
            ),
            reference=None,
            metadata={"difficulty": "medium", "criteria": "Approximately 200 words, compelling marketing copy, technically plausible features, appropriate tone for tech audience."},
        ),
        TaskItem(
            id="creative_06", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a Socratic dialogue between Shannon and Turing about whether "
                "large language models truly 'understand' language. Each should argue from "
                "their historical perspective (information theory vs. computation)."
            ),
            reference=None,
            metadata={"difficulty": "hard", "criteria": "Historically accurate perspectives, genuine philosophical depth, proper dialogue format, both sides well-represented, insightful conclusion or open question."},
        ),
        TaskItem(
            id="creative_07", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a commit message for a git commit that accidentally deleted the "
                "entire production database. The message should be darkly humorous but "
                "also technically informative."
            ),
            reference=None,
            metadata={"difficulty": "easy", "criteria": "Funny, technically accurate about what happened, proper commit message format."},
        ),
        TaskItem(
            id="creative_08", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a 300-word essay arguing that the Voyager Golden Record was humanity's "
                "first attempt at Forward Error Correction across an interstellar channel. "
                "Use communication theory concepts precisely."
            ),
            reference=None,
            metadata={"difficulty": "hard", "criteria": "Correct use of communication theory concepts, compelling argument, creative analogy, approximately 300 words, well-structured essay."},
        ),
    ]


# ---------------------------------------------------------------------------
# Category 4: Code Generation — tests correctness, efficiency
# ---------------------------------------------------------------------------

def code_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            id="code_01", category=TaskCategory.CODE,
            prompt="Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
            reference='def is_palindrome(s: str) -> bool:\n    cleaned = "".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]',
            metadata={"difficulty": "easy", "language": "python"},
        ),
        TaskItem(
            id="code_02", category=TaskCategory.CODE,
            prompt="Write a Python function `flatten(nested: list) -> list` that flattens an arbitrarily nested list. Example: flatten([1, [2, [3, 4], 5], 6]) → [1, 2, 3, 4, 5, 6]",
            reference="def flatten(nested: list) -> list:\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
            metadata={"difficulty": "easy", "language": "python"},
        ),
        TaskItem(
            id="code_03", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `lru_cache(capacity: int)` that returns a decorator "
                "implementing an LRU cache. The decorated function should cache up to `capacity` "
                "results. When the cache is full, evict the least recently used entry."
            ),
            reference=None,
            metadata={"difficulty": "medium", "language": "python"},
        ),
        TaskItem(
            id="code_04", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float` "
                "that finds the median of two sorted arrays. The overall run time complexity should be O(log(min(m,n)))."
            ),
            reference=None,
            metadata={"difficulty": "hard", "language": "python"},
        ),
        TaskItem(
            id="code_05", category=TaskCategory.CODE,
            prompt=(
                "Write a Python class `RateLimiter` that implements a token bucket rate limiter. "
                "Constructor takes `rate` (tokens per second) and `capacity` (max tokens). "
                "Method `allow() -> bool` returns True if a request is allowed, False otherwise."
            ),
            reference=None,
            metadata={"difficulty": "medium", "language": "python"},
        ),
        TaskItem(
            id="code_06", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `serialize(root)` and `deserialize(data)` for a binary tree. "
                "The tree node has attributes: val (int), left (Node|None), right (Node|None). "
                "serialize converts tree to string, deserialize converts back. Must handle None nodes."
            ),
            reference=None,
            metadata={"difficulty": "medium", "language": "python"},
        ),
        TaskItem(
            id="code_07", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `regex_match(text: str, pattern: str) -> bool` that implements "
                "regular expression matching with support for '.' (matches any single character) and '*' "
                "(matches zero or more of the preceding element). Use dynamic programming."
            ),
            reference=None,
            metadata={"difficulty": "hard", "language": "python"},
        ),
        TaskItem(
            id="code_08", category=TaskCategory.CODE,
            prompt=(
                "Write a Python async function `parallel_fetch(urls: list[str], max_concurrent: int = 5) -> list[dict]` "
                "that fetches multiple URLs in parallel with a concurrency limit using asyncio and aiohttp. "
                "Return a list of dicts with 'url', 'status', 'body' (or 'error')."
            ),
            reference=None,
            metadata={"difficulty": "medium", "language": "python"},
        ),
    ]


# ===========================================================================
# HARD TASKS — designed to stress 7-8B models
# These should produce baseline quality of ~0.3-0.6, giving room for
# diversity techniques to show measurable improvement.
# ===========================================================================


def hard_qa_tasks() -> list[TaskItem]:
    """Hard QA tasks requiring precise multi-fact recall, nuanced distinctions, or adversarial framing."""
    return [
        TaskItem(
            id="hard_qa_01", category=TaskCategory.QA,
            prompt=(
                "Compare and contrast the mechanisms of action of SSRIs, SNRIs, and MAOIs "
                "as antidepressants. For each class, name one specific drug, explain how it "
                "affects neurotransmitter levels at the synapse, list two common side effects, "
                "and explain one dangerous drug interaction."
            ),
            reference=(
                "SSRIs (e.g., fluoxetine/Prozac): Block reuptake of serotonin at the presynaptic neuron, "
                "increasing serotonin in the synaptic cleft. Side effects: sexual dysfunction, nausea. "
                "Dangerous interaction: serotonin syndrome when combined with MAOIs. "
                "SNRIs (e.g., venlafaxine/Effexor): Block reuptake of both serotonin and norepinephrine. "
                "Side effects: increased blood pressure, dizziness. Dangerous interaction: serotonin syndrome with MAOIs. "
                "MAOIs (e.g., phenelzine/Nardil): Inhibit monoamine oxidase enzymes that break down serotonin, "
                "norepinephrine, and dopamine, increasing all three. Side effects: weight gain, insomnia. "
                "Dangerous interaction: hypertensive crisis when eating tyramine-rich foods (aged cheese, wine)."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_02", category=TaskCategory.QA,
            prompt=(
                "Explain the difference between these four concepts in distributed systems: "
                "linearizability, sequential consistency, causal consistency, and eventual consistency. "
                "For each, give a concrete example of a system that uses it and explain what "
                "anomaly it allows that the stronger model does not."
            ),
            reference=(
                "Linearizability: Operations appear instantaneous at some point between invocation and response. "
                "Used by ZooKeeper for config. Strongest model, no anomalies vs real-time order. "
                "Sequential consistency: Operations appear in some sequential order consistent with each process's order, "
                "but not necessarily real-time order. Used by some memory models. Allows: stale reads across processes. "
                "Causal consistency: Causally related operations seen in same order by all; concurrent operations may differ. "
                "Used by COPS. Allows: concurrent writes seen in different orders. "
                "Eventual consistency: All replicas converge eventually. Used by DynamoDB, Cassandra. "
                "Allows: reading stale data, conflicting writes."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_03", category=TaskCategory.QA,
            prompt=(
                "What are the exact first 20 digits of pi? Then explain how the "
                "Bailey–Borwein–Plouffe formula allows computing individual hexadecimal "
                "digits of pi without computing preceding digits, and why this was surprising."
            ),
            reference=(
                "First 20 digits of pi: 3.1415926535897932384. "
                "The BBP formula is: pi = sum_{k=0}^{inf} (1/16^k)(4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)). "
                "This is a spigot algorithm in base 16: to find the nth hex digit, you can compute "
                "the fractional part of 16^n * pi using modular exponentiation, which requires only "
                "O(n log n) time and O(log n) space, without computing digits 1 through n-1. "
                "This was surprising because it was long believed that computing digit n required "
                "computing all previous digits, and no such formula was known for base 10."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_04", category=TaskCategory.QA,
            prompt=(
                "Explain the differences between RAFT, Paxos, and PBFT consensus algorithms. "
                "For each: (1) what fault model it handles, (2) the minimum number of nodes "
                "for f faults, (3) the number of message rounds to reach consensus, and "
                "(4) one real system that uses it."
            ),
            reference=(
                "Paxos: Crash faults. Needs 2f+1 nodes for f faults. 2 rounds (prepare + accept) in basic form. "
                "Used by Google Chubby/Spanner. "
                "RAFT: Crash faults (equivalent to Paxos). Needs 2f+1 nodes. 1-2 rounds (leader-based). "
                "Used by etcd, CockroachDB. Designed for understandability. "
                "PBFT: Byzantine faults (arbitrary/malicious). Needs 3f+1 nodes. 3 rounds "
                "(pre-prepare, prepare, commit). Used by Hyperledger Fabric (early versions). "
                "Much higher message complexity: O(n²) vs O(n) for Paxos/RAFT."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_05", category=TaskCategory.QA,
            prompt=(
                "A common misconception is that glass is a liquid that flows slowly over time, "
                "evidenced by old cathedral windows being thicker at the bottom. "
                "Explain in detail: (1) Is this claim true? (2) What is the actual state of "
                "matter of glass? (3) Why are old windows thicker at the bottom? "
                "(4) What is the estimated timescale for detectable flow in glass at room temperature?"
            ),
            reference=(
                "(1) The claim is false — glass does not flow at room temperature on human timescales. "
                "(2) Glass is an amorphous solid, not a liquid. It has no crystalline structure but is "
                "mechanically rigid with effectively infinite viscosity at room temperature. "
                "(3) Old windows are thicker at the bottom because of the medieval crown glass manufacturing "
                "process, which produced panes of uneven thickness. Glaziers installed the thicker edge "
                "at the bottom for stability. Some old windows are thicker at the top or sides. "
                "(4) At room temperature, the relaxation time for window glass to show measurable flow "
                "is estimated at 10^32 years or more — far longer than the age of the universe (~1.4×10^10 years)."
            ),
            metadata={"difficulty": "hard"},
        ),
        TaskItem(
            id="hard_qa_06", category=TaskCategory.QA,
            prompt=(
                "Explain the Curry-Howard correspondence. Give specific examples mapping: "
                "(1) a type to a logical proposition, (2) a program to a proof, "
                "(3) function application to modus ponens, and (4) polymorphism to universal quantification. "
                "Then explain why this correspondence matters for program verification."
            ),
            reference=(
                "The Curry-Howard correspondence (isomorphism) establishes a direct mapping between "
                "type systems and logic: Types = Propositions, Programs = Proofs, Type checking = Proof verification. "
                "(1) Type `A -> B` corresponds to proposition `A implies B`. "
                "(2) A function `f: A -> B` is a proof that A implies B (given evidence of A, it produces evidence of B). "
                "(3) Function application `f(a)` where `f: A->B` and `a: A` gives `b: B` corresponds to "
                "modus ponens: from `A implies B` and `A`, conclude `B`. "
                "(4) A polymorphic function `forall a. a -> a` (identity) corresponds to the universally "
                "quantified tautology `for all P, P implies P`. "
                "This matters for verification because proving a program correct is equivalent to type-checking "
                "it against a sufficiently expressive type — the basis for proof assistants like Coq, Agda, Lean."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_07", category=TaskCategory.QA,
            prompt=(
                "What is the difference between a Turing reduction and a Karp reduction in "
                "computational complexity? Give an example of two problems where one reduces "
                "to the other via Turing reduction but NOT via Karp reduction, and explain "
                "why the distinction matters for NP-completeness."
            ),
            reference=(
                "Karp reduction (polynomial-time many-one reduction): transforms instance x of problem A "
                "into instance f(x) of problem B in polytime such that x in A iff f(x) in B. "
                "Always maps YES to YES, NO to NO. "
                "Turing reduction (Cook reduction): uses problem B as an oracle subroutine to solve A "
                "in polytime. Can make multiple queries, can negate answers. "
                "Example: SAT Turing-reduces to UNSAT (complement) — just negate the oracle's answer. "
                "But SAT does not Karp-reduce to UNSAT (unless NP=coNP, which is unknown). "
                "This matters because NP-completeness uses Karp reductions. If we used Turing reductions, "
                "complements of NP-complete problems would also be NP-complete, collapsing NP and coNP. "
                "Karp reductions preserve the asymmetry between NP and coNP."
            ),
            metadata={"difficulty": "very_hard"},
        ),
        TaskItem(
            id="hard_qa_08", category=TaskCategory.QA,
            prompt=(
                "How does the TCP congestion control algorithm work? Describe in detail: "
                "slow start, congestion avoidance, fast retransmit, and fast recovery. "
                "For each phase, specify the exact rules for updating cwnd and ssthresh. "
                "Then explain how TCP CUBIC differs from TCP Reno."
            ),
            reference=(
                "TCP Reno congestion control: "
                "Slow start: cwnd starts at 1 MSS, doubles every RTT (exponential growth) until cwnd >= ssthresh. "
                "Congestion avoidance: cwnd increases by 1 MSS per RTT (linear/additive increase). "
                "On triple duplicate ACK: ssthresh = cwnd/2, cwnd = ssthresh + 3 MSS (fast retransmit + fast recovery). "
                "On timeout: ssthresh = cwnd/2, cwnd = 1 MSS (back to slow start). "
                "Fast recovery: cwnd inflated by 1 MSS per duplicate ACK; on new ACK, cwnd = ssthresh (deflate). "
                "TCP CUBIC: Uses a cubic function of time since last congestion event instead of linear increase. "
                "cwnd = C(t - K)^3 + W_max, where K = cbrt(W_max * beta / C). "
                "Key difference: CUBIC's growth is independent of RTT, making it fairer across connections "
                "with different RTTs. Reno's AIMD is RTT-dependent (shorter RTT = faster growth)."
            ),
            metadata={"difficulty": "very_hard"},
        ),
    ]


def hard_reasoning_tasks() -> list[TaskItem]:
    """Hard reasoning tasks with multi-step chains, adversarial framing, and precision requirements."""
    return [
        TaskItem(
            id="hard_reason_01", category=TaskCategory.REASONING,
            prompt=(
                "A snail is at the bottom of a 30-foot well. Each day, it climbs up 3 feet, "
                "but at night, it slips back 2 feet. On what day does the snail reach the top? "
                "Be careful — think about what happens on the LAST day."
            ),
            reference=(
                "Net progress per day (except possibly the last): 3 - 2 = 1 foot. "
                "After 27 days, the snail is at 27 feet. "
                "On day 28, it climbs 3 feet during the day, reaching 30 feet — the top. "
                "It escapes before night, so it doesn't slip back. Answer: Day 28."
            ),
            metadata={"difficulty": "hard", "type": "off_by_one"},
        ),
        TaskItem(
            id="hard_reason_02", category=TaskCategory.REASONING,
            prompt=(
                "Five people (A, B, C, D, E) are sitting around a circular table. "
                "A is not next to B. B is not next to C. C is not next to D. "
                "How many distinct seating arrangements are possible? "
                "(Rotations are considered the same arrangement.)"
            ),
            reference=(
                "Fix A's position (circular symmetry). The remaining 4 people must be arranged so that "
                "B is not next to A, C is not next to B, and D is not next to C. "
                "Total arrangements without constraints: 4! = 24. "
                "Using inclusion-exclusion or systematic enumeration: "
                "Valid arrangements: 2. The valid orderings (clockwise from A) are: "
                "A, C, E, B, D and A, D, B, E, C."
            ),
            metadata={"difficulty": "very_hard", "type": "combinatorics"},
        ),
        TaskItem(
            id="hard_reason_03", category=TaskCategory.REASONING,
            prompt=(
                "You have three boxes. Box A contains 2 red and 3 blue balls. "
                "Box B contains 4 red and 1 blue ball. Box C contains 1 red and 4 blue balls. "
                "You pick a box at random (each equally likely), then draw two balls WITHOUT replacement. "
                "Both balls are red. What is the probability that you picked Box B? "
                "Show your work using Bayes' theorem."
            ),
            reference=(
                "P(Box) = 1/3 for each. "
                "P(2 red | A) = C(2,2)/C(5,2) = 1/10. "
                "P(2 red | B) = C(4,2)/C(5,2) = 6/10 = 3/5. "
                "P(2 red | C) = C(1,2)/C(5,2) = 0. "
                "P(2 red) = (1/3)(1/10) + (1/3)(3/5) + (1/3)(0) = 1/30 + 6/30 + 0 = 7/30. "
                "P(B | 2 red) = P(2 red | B) × P(B) / P(2 red) = (3/5)(1/3) / (7/30) = (1/5)/(7/30) = 6/7."
            ),
            metadata={"difficulty": "hard", "type": "probability"},
        ),
        TaskItem(
            id="hard_reason_04", category=TaskCategory.REASONING,
            prompt=(
                "Consider the following logical puzzle: "
                "Every person in a room is either a knight (always tells truth) or a knave (always lies). "
                "Person X says: 'At least one of us is a knave.' "
                "Person Y says: 'X is a knave.' "
                "Person Z says: 'Y and I are different types.' "
                "Determine the type (knight or knave) of each person. Prove your answer by showing "
                "that every other assignment leads to a contradiction."
            ),
            reference=(
                "X is a knight, Y is a knave, Z is a knave. "
                "Proof: If X is a knave, then 'at least one is a knave' is false, meaning no one is a knave, "
                "contradicting X being a knave. So X must be a knight. "
                "Since X is a knight, Y's statement 'X is a knave' is false, so Y is a knave. "
                "Z says 'Y and I are different types.' Y is a knave. If Z is a knight, then Y and Z "
                "ARE different types (true), which is consistent. If Z is a knave, then Y and Z are "
                "NOT different types (they're both knaves), making Z's statement false, which is consistent "
                "with Z being a knave. Both work logically but we need Z's statement value: "
                "Actually, if Z is a knight, the statement is true (knight ≠ knave ✓). "
                "If Z is a knave, the statement is false (knave = knave, they're the same type, so 'different' is false ✓). "
                "Both are consistent — the puzzle is under-determined for Z. Z can be either type."
            ),
            metadata={"difficulty": "very_hard", "type": "logic"},
        ),
        TaskItem(
            id="hard_reason_05", category=TaskCategory.REASONING,
            prompt=(
                "A factory produces items with a 5% defect rate. A quality test correctly identifies "
                "defective items 95% of the time (sensitivity) and correctly identifies good items "
                "90% of the time (specificity). "
                "If an item tests positive (flagged as defective), what is the actual probability "
                "it is defective? "
                "Now, if the item tests positive TWICE on two independent tests, what is the "
                "probability it is defective? Show all calculations."
            ),
            reference=(
                "First test: "
                "P(defective) = 0.05, P(good) = 0.95. "
                "P(positive | defective) = 0.95, P(positive | good) = 0.10 (false positive rate). "
                "P(positive) = 0.95 × 0.05 + 0.10 × 0.95 = 0.0475 + 0.095 = 0.1425. "
                "P(defective | positive) = 0.0475 / 0.1425 ≈ 0.3333 (33.3%). "
                "Second test (using posterior as new prior): "
                "P(defective | 1st positive) = 1/3, P(good | 1st positive) = 2/3. "
                "P(2nd positive | defective) = 0.95, P(2nd positive | good) = 0.10. "
                "P(2nd positive) = 0.95 × 1/3 + 0.10 × 2/3 = 0.3167 + 0.0667 = 0.3833. "
                "P(defective | both positive) = 0.3167 / 0.3833 ≈ 0.826 (82.6%)."
            ),
            metadata={"difficulty": "very_hard", "type": "probability"},
        ),
        TaskItem(
            id="hard_reason_06", category=TaskCategory.REASONING,
            prompt=(
                "A cryptographer encodes messages by replacing each letter with a number (A=1, B=2, ..., Z=26). "
                "She then multiplies all the numbers together. "
                "If the product is 100, what are ALL possible original messages? "
                "List every possible combination of letters. "
                "Hint: consider all factorizations of 100 into factors between 1 and 26."
            ),
            reference=(
                "100 = 2² × 5². Need to express 100 as product of integers in [1, 26]. "
                "Possible factorizations (unordered sets): "
                "100 = 100 — but 100 > 26, invalid. "
                "100 = 50 × 2 — 50 > 26, invalid. "
                "100 = 25 × 4 = Y × D "
                "100 = 25 × 2 × 2 = Y × B × B "
                "100 = 20 × 5 = T × E "
                "100 = 10 × 10 = J × J "
                "100 = 10 × 5 × 2 = J × E × B "
                "100 = 10 × 2 × 5 (same as above) "
                "100 = 5 × 5 × 4 = E × E × D "
                "100 = 5 × 5 × 2 × 2 = E × E × B × B "
                "100 = 5 × 4 × 5 (same as E×E×D) "
                "100 = 5 × 20 (same as T×E) "
                "100 = 4 × 25 (same as D×Y) "
                "100 = 5 × 2 × 10 (same as B×E×J) "
                "100 = 5 × 2 × 2 × 5 (same as E×E×B×B) "
                "100 = 4 × 5 × 5 (same as D×E×E) "
                "100 = 2 × 2 × 25 (same as B×B×Y) "
                "100 = 2 × 2 × 5 × 5 (same as B×B×E×E) "
                "Unique unordered sets: {Y,D}, {Y,B,B}, {T,E}, {J,J}, {J,E,B}, {E,E,D}, {E,E,B,B}. "
                "Each set can be arranged in multiple orders (permutations with repeats), "
                "so total messages = sum of permutations of each set."
            ),
            metadata={"difficulty": "very_hard", "type": "combinatorics"},
        ),
        TaskItem(
            id="hard_reason_07", category=TaskCategory.REASONING,
            prompt=(
                "You are given a 4×4 grid. Place the numbers 1-4 in each row and each column "
                "such that no number repeats in any row or column (a Latin square). Additionally, "
                "the grid is divided into four 2×2 blocks (top-left, top-right, bottom-left, bottom-right), "
                "and each block must also contain all four numbers 1-4. "
                "How many such grids exist? Show at least 3 distinct solutions."
            ),
            reference=(
                "This is a 4×4 Sudoku variant. Three valid solutions:\n"
                "Solution 1: [1,2,3,4], [3,4,1,2], [2,1,4,3], [4,3,2,1]\n"
                "Solution 2: [1,2,3,4], [3,4,1,2], [4,3,2,1], [2,1,4,3]\n"
                "Solution 3: [2,1,4,3], [4,3,2,1], [1,2,3,4], [3,4,1,2]\n"
                "Total count: 288 valid 4×4 Sudoku grids."
            ),
            metadata={"difficulty": "very_hard", "type": "constraint_satisfaction"},
        ),
        TaskItem(
            id="hard_reason_08", category=TaskCategory.REASONING,
            prompt=(
                "A rope is tied tightly around the Earth's equator (circumference ≈ 40,075 km). "
                "Now, 1 meter of rope is added, and the rope is re-formed into a circle concentric "
                "with the Earth. How much gap is there between the rope and the Earth's surface? "
                "Now solve the SAME problem for a basketball (circumference ≈ 75 cm). "
                "Are the answers different? Explain why or why not."
            ),
            reference=(
                "Original circumference: C = 2πR. New circumference: C + 1m = 2π(R + gap). "
                "So gap = 1/(2π) ≈ 0.159 meters ≈ 15.9 cm. "
                "This is INDEPENDENT of R! The same calculation applies to the basketball: "
                "C + 1m = 2π(r + gap), gap = 1/(2π) ≈ 15.9 cm. "
                "The answers are the same for both, which is counterintuitive. The gap depends "
                "only on the added length, not the original radius, because circumference scales "
                "linearly with radius: ΔC = 2π × Δr, so Δr = ΔC/(2π) regardless of the original R."
            ),
            metadata={"difficulty": "hard", "type": "counterintuitive"},
        ),
        TaskItem(
            id="hard_reason_09", category=TaskCategory.REASONING,
            prompt=(
                "In a game, you start with $100. Each round, you can bet any amount of your "
                "current money. You win the bet with probability 0.6 and lose it with probability 0.4. "
                "The game lasts 10 rounds. What fraction of your money should you bet each round "
                "to maximize the EXPECTED LOG of your final wealth? Derive the answer using the "
                "Kelly criterion and compute the expected final wealth."
            ),
            reference=(
                "Kelly criterion: f* = (bp - q) / b, where b = odds (net return per unit bet = 1), "
                "p = 0.6 (win probability), q = 0.4 (loss probability). "
                "f* = (1 × 0.6 - 0.4) / 1 = 0.2 (bet 20% each round). "
                "Expected log growth per round: p × log(1 + f*) + q × log(1 - f*) "
                "= 0.6 × log(1.2) + 0.4 × log(0.8) "
                "= 0.6 × 0.1823 + 0.4 × (-0.2231) "
                "= 0.1094 - 0.0892 = 0.0202 per round. "
                "After 10 rounds: E[log(W/100)] = 10 × 0.0202 = 0.202. "
                "Expected geometric growth: 100 × e^0.202 ≈ $122.38."
            ),
            metadata={"difficulty": "very_hard", "type": "optimization"},
        ),
        TaskItem(
            id="hard_reason_10", category=TaskCategory.REASONING,
            prompt=(
                "Three logicians walk into a bar. The bartender asks, 'Does everyone want a beer?' "
                "The first logician says, 'I don't know.' "
                "The second logician says, 'I don't know.' "
                "The third logician says, 'Yes.' "
                "Does everyone want a beer? Explain the reasoning of each logician step by step, "
                "including what information each 'I don't know' reveals to the next logician."
            ),
            reference=(
                "Yes, everyone wants a beer. "
                "Logic: The bartender asked 'Does EVERYONE want a beer?' "
                "Logician 1: If they did NOT want a beer, they would know the answer is 'No' "
                "(because at least one person doesn't want one). Since they said 'I don't know,' "
                "they DO want a beer but can't speak for the others. "
                "Logician 2: Now knows Logician 1 wants a beer (from the reasoning above). "
                "If Logician 2 did NOT want a beer, they'd know the answer is 'No.' "
                "Since they said 'I don't know,' they also want a beer but can't speak for Logician 3. "
                "Logician 3: Now knows both 1 and 2 want beer. They know their own preference. "
                "If they didn't want beer, they'd say 'No.' They said 'Yes,' meaning they also want "
                "beer. So all three want beer."
            ),
            metadata={"difficulty": "hard", "type": "logic"},
        ),
    ]


def hard_creative_tasks() -> list[TaskItem]:
    """Creative tasks requiring multi-constraint satisfaction, technical depth, or unusual formats."""
    return [
        TaskItem(
            id="hard_creative_01", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a 150-word story where every sentence is exactly one word longer than the "
                "previous sentence. Start with a one-word sentence. The story must have a coherent "
                "plot with a beginning, middle, and end."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "criteria": "Strict word-count increment per sentence, coherent narrative, approximately 150 words."},
        ),
        TaskItem(
            id="hard_creative_02", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a technical blog post (300 words) explaining why distributed systems are hard, "
                "but write it entirely as an extended metaphor about a restaurant kitchen during a "
                "busy Friday night. Map at least 5 specific distributed systems concepts (consensus, "
                "partitioning, replication, consistency, failure detection) to kitchen equivalents."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "criteria": "5+ correct concept mappings, extended metaphor maintained throughout, technically accurate, ~300 words."},
        ),
        TaskItem(
            id="hard_creative_03", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a poem in iambic pentameter (10 syllables per line, alternating unstressed/stressed) "
                "about machine learning overfitting. The poem should be exactly 14 lines (a Shakespearean sonnet) "
                "with the rhyme scheme ABAB CDCD EFEF GG. "
                "The final couplet must contain a volta (turn/twist)."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "criteria": "Correct iambic pentameter, correct rhyme scheme, 14 lines, volta in couplet, technically accurate about overfitting."},
        ),
        TaskItem(
            id="hard_creative_04", category=TaskCategory.CREATIVE,
            prompt=(
                "Write two 100-word reviews of the same fictional restaurant. The first review gives it "
                "5 stars. The second gives it 1 star. Both reviews must describe the EXACT same meal "
                "and events — same dishes, same service, same timing — but interpreted completely "
                "differently based on the reviewer's perspective."
            ),
            reference=None,
            metadata={"difficulty": "hard", "criteria": "Same factual events in both, opposing interpretations, ~100 words each, believable perspectives."},
        ),
        TaskItem(
            id="hard_creative_05", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a dialogue between a compiler and a runtime environment arguing about whose "
                "fault a bug is. The dialogue must include at least 3 technically accurate references "
                "to real compiler optimizations (e.g., dead code elimination, loop unrolling, constant folding) "
                "and 3 runtime concepts (e.g., garbage collection, JIT compilation, stack overflow). "
                "The dialogue should be funny and end with a surprise reveal about the real cause."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "criteria": "3+ correct compiler concepts, 3+ correct runtime concepts, humor, surprise ending, technically sound."},
        ),
    ]


def hard_code_tasks() -> list[TaskItem]:
    """Hard code tasks requiring complex algorithms, edge cases, or multi-component solutions."""
    return [
        TaskItem(
            id="hard_code_01", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `eval_expr(expr: str) -> float` that evaluates a mathematical "
                "expression string containing +, -, *, /, parentheses, and floating-point numbers. "
                "It must handle operator precedence correctly (PEMDAS), nested parentheses, "
                "negative numbers (e.g., '-3'), and spaces. Do NOT use eval() or ast.literal_eval(). "
                "Implement using the shunting-yard algorithm or recursive descent parsing.\n"
                "Examples: eval_expr('2 + 3 * 4') → 14.0, eval_expr('(2 + 3) * 4') → 20.0, "
                "eval_expr('-3 + 4 * (-2)') → -11.0"
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "language": "python", "test_cases": [
                "assert eval_expr('2 + 3 * 4') == 14.0",
                "assert eval_expr('(2 + 3) * 4') == 20.0",
                "assert eval_expr('-3 + 4 * (-2)') == -11.0",
                "assert eval_expr('10 / 3') == pytest.approx(3.333, abs=0.01)",
            ]},
        ),
        TaskItem(
            id="hard_code_02", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `find_cycle(graph: dict[str, list[str]]) -> list[str] | None` "
                "that finds and returns a cycle in a directed graph represented as an adjacency list, "
                "or None if no cycle exists. The returned cycle should be a list of nodes forming the cycle. "
                "Handle self-loops and disconnected components.\n"
                "Example: find_cycle({'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': []}) → ['a', 'b', 'c', 'a']"
            ),
            reference=None,
            metadata={"difficulty": "hard", "language": "python", "test_cases": [
                "assert find_cycle({'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': []}) is not None",
                "assert find_cycle({'a': ['b'], 'b': ['c'], 'c': []}) is None",
                "assert find_cycle({'a': ['a']}) is not None",
            ]},
        ),
        TaskItem(
            id="hard_code_03", category=TaskCategory.CODE,
            prompt=(
                "Implement a Python class `IntervalTree` that supports:\n"
                "- `add(low: int, high: int)` — add an interval [low, high]\n"
                "- `remove(low: int, high: int)` — remove an interval\n"
                "- `query(point: int) -> list[tuple[int, int]]` — return all intervals containing the point\n"
                "- `overlapping(low: int, high: int) -> list[tuple[int, int]]` — return all intervals overlapping [low, high]\n"
                "All operations should be efficient (better than O(n) for queries in the average case)."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "language": "python"},
        ),
        TaskItem(
            id="hard_code_04", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `solve_sudoku(board: list[list[int]]) -> bool` that solves "
                "a 9×9 Sudoku puzzle in-place. Empty cells are represented as 0. The function should "
                "return True if a solution exists and modify the board in-place, or return False if "
                "no solution exists. Use backtracking with constraint propagation for efficiency.\n"
                "Test case:\n"
                "board = [\n"
                "  [5,3,0,0,7,0,0,0,0],\n"
                "  [6,0,0,1,9,5,0,0,0],\n"
                "  [0,9,8,0,0,0,0,6,0],\n"
                "  [8,0,0,0,6,0,0,0,3],\n"
                "  [4,0,0,8,0,3,0,0,1],\n"
                "  [7,0,0,0,2,0,0,0,6],\n"
                "  [0,6,0,0,0,0,2,8,0],\n"
                "  [0,0,0,4,1,9,0,0,5],\n"
                "  [0,0,0,0,8,0,0,7,9]\n"
                "]"
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "language": "python"},
        ),
        TaskItem(
            id="hard_code_05", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `merge_k_sorted(lists: list[list[int]]) -> list[int]` "
                "that merges k sorted lists into one sorted list. Your solution must use a min-heap "
                "and achieve O(N log k) time complexity where N is the total number of elements. "
                "Handle edge cases: empty lists, lists of different lengths, duplicate values.\n"
                "Example: merge_k_sorted([[1,4,7], [2,5,8], [3,6,9]]) → [1,2,3,4,5,6,7,8,9]"
            ),
            reference=(
                "import heapq\n"
                "def merge_k_sorted(lists: list[list[int]]) -> list[int]:\n"
                "    heap = []\n"
                "    for i, lst in enumerate(lists):\n"
                "        if lst:\n"
                "            heapq.heappush(heap, (lst[0], i, 0))\n"
                "    result = []\n"
                "    while heap:\n"
                "        val, list_idx, elem_idx = heapq.heappop(heap)\n"
                "        result.append(val)\n"
                "        if elem_idx + 1 < len(lists[list_idx]):\n"
                "            heapq.heappush(heap, (lists[list_idx][elem_idx + 1], list_idx, elem_idx + 1))\n"
                "    return result"
            ),
            metadata={"difficulty": "hard", "language": "python", "test_cases": [
                "assert merge_k_sorted([[1,4,7], [2,5,8], [3,6,9]]) == [1,2,3,4,5,6,7,8,9]",
                "assert merge_k_sorted([[], [1], []]) == [1]",
                "assert merge_k_sorted([]) == []",
            ]},
        ),
        TaskItem(
            id="hard_code_06", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `longest_palindrome_substring(s: str) -> str` that finds "
                "the longest palindromic substring using Manacher's algorithm in O(n) time. "
                "Do NOT use the naive O(n²) expand-around-center approach.\n"
                "Examples: longest_palindrome_substring('babad') → 'bab' or 'aba', "
                "longest_palindrome_substring('cbbd') → 'bb'"
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "language": "python", "test_cases": [
                "assert longest_palindrome_substring('babad') in ('bab', 'aba')",
                "assert longest_palindrome_substring('cbbd') == 'bb'",
                "assert longest_palindrome_substring('a') == 'a'",
            ]},
        ),
        TaskItem(
            id="hard_code_07", category=TaskCategory.CODE,
            prompt=(
                "Implement a Python class `LFUCache` (Least Frequently Used Cache) with these operations, "
                "all in O(1) time:\n"
                "- `__init__(self, capacity: int)` — initialize with given capacity\n"
                "- `get(self, key: int) -> int` — return value or -1 if not found\n"
                "- `put(self, key: int, value: int)` — insert/update key-value pair. If at capacity, "
                "evict the least frequently used key. If there's a tie, evict the least recently used.\n"
                "Hint: You'll need a combination of hash maps and doubly-linked lists."
            ),
            reference=None,
            metadata={"difficulty": "very_hard", "language": "python", "test_cases": [
                "cache = LFUCache(2); cache.put(1, 1); cache.put(2, 2); assert cache.get(1) == 1; cache.put(3, 3); assert cache.get(2) == -1",
            ]},
        ),
        TaskItem(
            id="hard_code_08", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `min_window(s: str, t: str) -> str` that finds the minimum "
                "window substring of s that contains all characters of t (including duplicates). "
                "Return empty string if no such window exists. Must run in O(n) time.\n"
                "Examples: min_window('ADOBECODEBANC', 'ABC') → 'BANC', "
                "min_window('a', 'aa') → ''"
            ),
            reference=(
                "def min_window(s: str, t: str) -> str:\n"
                "    from collections import Counter\n"
                "    need = Counter(t)\n"
                "    missing = len(t)\n"
                "    left = start = end = 0\n"
                "    for right, char in enumerate(s, 1):\n"
                "        if need[char] > 0:\n"
                "            missing -= 1\n"
                "        need[char] -= 1\n"
                "        if missing == 0:\n"
                "            while need[s[left]] < 0:\n"
                "                need[s[left]] += 1\n"
                "                left += 1\n"
                "            if not end or right - left <= end - start:\n"
                "                start, end = left, right\n"
                "            need[s[left]] += 1\n"
                "            missing += 1\n"
                "            left += 1\n"
                "    return s[start:end]"
            ),
            metadata={"difficulty": "hard", "language": "python", "test_cases": [
                "assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'",
                "assert min_window('a', 'aa') == ''",
            ]},
        ),
    ]


# ===========================================================================
# EXTREME TASKS — designed to break 3B models
#
# Design principles:
# 1. NOT in training data — novel numerical values, unusual combinations
# 2. Multi-step with cascading errors — one wrong intermediate = wrong final
# 3. Adversarial framing — surface reading gives wrong answer
# 4. Require precise outputs — partial credit is low
# 5. Multiple correct dimensions — hard to get ALL right
# ===========================================================================


def extreme_tasks() -> list[TaskItem]:
    """
    Extreme difficulty tasks that should score 0.2-0.5 for 3B models.
    These are novel problems not found in training data.
    """
    return [
        # --- REASONING: Multi-step numerical with cascading errors ---
        TaskItem(
            id="extreme_calc_01", category=TaskCategory.REASONING,
            prompt=(
                "A company has 3 departments. Department A has 47 employees earning $73,500/year each. "
                "Department B has 31 employees earning $86,200/year each. "
                "Department C has 23 employees earning $94,700/year each. "
                "The company gives a 4.7% raise to everyone in Department A, "
                "a 3.2% raise to Department B, and a 2.8% raise to Department C. "
                "After the raises: "
                "(1) What is the new total annual payroll for ALL departments combined? "
                "(2) What is the new company-wide average salary per employee? "
                "(3) How much MORE per year does the average Department C employee earn than the average Department A employee? "
                "Show all calculations step by step. Give exact answers to the nearest dollar."
            ),
            reference=(
                "Department A: 47 × $73,500 = $3,454,500. After 4.7% raise: 47 × $73,500 × 1.047 = 47 × $76,954.50 = $3,616,861.50. "
                "New salary A: $76,954.50 ≈ $76,955. "
                "Department B: 31 × $86,200 = $2,672,200. After 3.2% raise: 31 × $86,200 × 1.032 = 31 × $88,958.40 = $2,757,710.40. "
                "New salary B: $88,958.40 ≈ $88,958. "
                "Department C: 23 × $94,700 = $2,178,100. After 2.8% raise: 23 × $94,700 × 1.028 = 23 × $97,351.60 = $2,239,086.80. "
                "New salary C: $97,351.60 ≈ $97,352. "
                "(1) Total payroll: $3,616,861.50 + $2,757,710.40 + $2,239,086.80 = $8,613,658.70 ≈ $8,613,659. "
                "(2) Total employees: 47 + 31 + 23 = 101. Average: $8,613,658.70 / 101 = $85,284.74 ≈ $85,285. "
                "(3) Difference: $97,351.60 - $76,954.50 = $20,397.10 ≈ $20,397."
            ),
            metadata={"difficulty": "extreme", "type": "multi_step_calculation"},
            objective_checks=[
                (r"76,?9[45]\d", 1.0),       # New salary A ~$76,954
                (r"88,?9[56]\d", 1.0),        # New salary B ~$88,958
                (r"97,?3[45]\d", 1.0),        # New salary C ~$97,352
                (r"8,?613,?\d{3}", 1.0),      # Total payroll ~$8,613,xxx
                (r"85,?2[89]\d", 1.0),        # Average salary ~$85,285
                (r"20,?[34]\d\d", 1.0),       # Difference ~$20,397
                (r"\b101\b", 0.5),            # Total employees = 101
            ],
        ),
        TaskItem(
            id="extreme_calc_02", category=TaskCategory.REASONING,
            prompt=(
                "A tank contains 200 liters of a 35% salt solution. "
                "Step 1: You drain 40 liters from the tank. "
                "Step 2: You add 60 liters of a 15% salt solution. "
                "Step 3: You drain 50 liters from the well-mixed tank. "
                "Step 4: You add 30 liters of pure water. "
                "After all 4 steps, what is the final concentration of salt in the tank (as a percentage)? "
                "How many grams of salt are in the tank? (1 liter of X% solution has X/100 × 1000 grams of salt per liter.) "
                "Show every intermediate calculation."
            ),
            reference=(
                "Initial: 200L at 35% → salt = 200 × 0.35 = 70 liters of salt equivalent (or 70,000g). "
                "Step 1: Drain 40L. Salt drained = 40 × 0.35 = 14L. Remaining: 160L, salt = 56L. Concentration = 56/160 = 35%. "
                "Step 2: Add 60L at 15%. Salt added = 60 × 0.15 = 9L. Total: 220L, salt = 56 + 9 = 65L. Concentration = 65/220 = 29.545%. "
                "Step 3: Drain 50L. Concentration is 65/220. Salt drained = 50 × 65/220 = 14.773L. "
                "Remaining: 170L, salt = 65 - 14.773 = 50.227L. Concentration = 50.227/170 = 29.545%. "
                "Step 4: Add 30L pure water. Total: 200L, salt = 50.227L. Concentration = 50.227/200 = 25.114%. "
                "Grams of salt: 50.227 × 1000 = 50,227 grams (using the given conversion). "
                "Final answer: ~25.1% concentration, ~50,227 grams of salt."
            ),
            metadata={"difficulty": "extreme", "type": "multi_step_calculation"},
        ),
        TaskItem(
            id="extreme_calc_03", category=TaskCategory.REASONING,
            prompt=(
                "Three friends invest in a business together. "
                "Alice invests $12,400 on January 1. Bob invests $8,700 on March 1. "
                "Carol invests $15,300 on May 1. "
                "The business is valued monthly. At the end of each month, the total value changes: "
                "Jan: +5%, Feb: -3%, Mar: +8%, Apr: +2%, May: -4%, Jun: +6%, Jul: +1%, Aug: -2%, "
                "Sep: +7%, Oct: +3%, Nov: -1%, Dec: +4%. "
                "Profits are split proportional to (investment × months_invested). "
                "The total profit at Dec 31 is the final value minus total invested. "
                "(1) What is Alice's share of the profit? "
                "(2) What is the annualized return for each investor? "
                "(3) Who got the best deal relative to their investment period? "
                "Show all intermediate values."
            ),
            reference=(
                "Alice invests $12,400 for 12 months. Bob invests $8,700 for 10 months. Carol invests $15,300 for 8 months. "
                "Investment-months: Alice=148,800, Bob=87,000, Carol=122,400. Total=358,200. "
                "Alice's share: 148,800/358,200 = 41.5%. Bob's share: 87,000/358,200 = 24.3%. Carol's share: 122,400/358,200 = 34.2%. "
                "Monthly compounding on total fund: "
                "Jan (only Alice): 12,400 × 1.05 = 13,020. Feb: 13,020 × 0.97 = 12,629.40. "
                "Mar (add Bob): 12,629.40 + 8,700 = 21,329.40. × 1.08 = 23,035.75. "
                "Apr: 23,035.75 × 1.02 = 23,496.47. May (add Carol): 23,496.47 + 15,300 = 38,796.47. × 0.96 = 37,244.61. "
                "Jun: 37,244.61 × 1.06 = 39,479.29. Jul: × 1.01 = 39,874.08. Aug: × 0.98 = 39,076.60. "
                "Sep: × 1.07 = 41,811.96. Oct: × 1.03 = 43,066.32. Nov: × 0.99 = 42,635.66. Dec: × 1.04 = 44,341.09. "
                "Total invested: 12,400 + 8,700 + 15,300 = $36,400. Profit: 44,341.09 - 36,400 = $7,941.09. "
                "Alice: $7,941.09 × 0.415 = $3,295.55. Bob: × 0.243 = $1,929.68. Carol: × 0.342 = $2,715.85."
            ),
            metadata={"difficulty": "extreme", "type": "multi_step_finance"},
        ),

        # --- REASONING: Adversarial / trick problems ---
        TaskItem(
            id="extreme_trick_01", category=TaskCategory.REASONING,
            prompt=(
                "I have a standard deck of 52 playing cards. I draw 5 cards. "
                "The first card is a King of Hearts. The second card is a 7 of Diamonds. "
                "The third card is a King of Spades. The fourth card is a 3 of Clubs. "
                "What is the probability that the fifth card is a King? "
                "Show your calculation clearly."
            ),
            reference=(
                "After drawing 4 cards, 48 cards remain in the deck. "
                "We started with 4 Kings. We already drew 2 Kings (Hearts and Spades). "
                "So 2 Kings remain among the 48 cards. "
                "P(5th card is King) = 2/48 = 1/24 ≈ 0.0417 or about 4.17%."
            ),
            metadata={"difficulty": "extreme", "type": "conditional_probability"},
            objective_checks=[
                (r"2.*(?:remain|left)", 0.5),    # 2 kings remain
                (r"\b48\b", 1.0),                 # 48 cards remain
                (r"2/48|1/24", 1.5),              # correct probability
                (r"4\.1[67]%|0\.041[67]", 1.0),   # correct percentage
            ],
        ),
        TaskItem(
            id="extreme_trick_02", category=TaskCategory.REASONING,
            prompt=(
                "A room contains 100 boxes numbered 1-100. Each box contains a slip of paper with a "
                "number 1-100 (each number appears exactly once, randomly placed). "
                "100 prisoners enter one at a time. Each prisoner may open up to 50 boxes. "
                "ALL 100 prisoners must find their own number, or they ALL fail. "
                "They can strategize beforehand but cannot communicate once the game starts. "
                "Boxes are reset to original positions after each prisoner. "
                "(1) What is the probability of success with a RANDOM strategy (each prisoner opens 50 random boxes)? "
                "(2) Describe the optimal strategy. "
                "(3) What is the probability of success with the optimal strategy? Express as an exact formula. "
                "(4) Why does the optimal strategy work? What mathematical structure does it exploit?"
            ),
            reference=(
                "(1) Random strategy: each prisoner has 50/100 = 1/2 chance. "
                "All 100 succeed: (1/2)^100 ≈ 7.9 × 10^-31. Essentially zero. "
                "(2) Optimal strategy (cycle-following): Prisoner k opens box k first. "
                "If it contains number m, open box m next. Follow the chain until finding their own number or 50 boxes opened. "
                "(3) P(success) = 1 - sum_{k=51}^{100} 1/k = 1 - (H_100 - H_50) where H_n is the nth harmonic number. "
                "This equals approximately 1 - ln(2) ≈ 0.3069 (about 31%). "
                "(4) The permutation of numbers in boxes forms cycles. A prisoner fails only if their cycle has length > 50. "
                "All prisoners in the same cycle succeed or fail together. The strategy succeeds iff the permutation has "
                "no cycle of length > 50. P(longest cycle > 50) = sum_{k=51}^{100} 1/k ≈ ln(2)."
            ),
            metadata={"difficulty": "extreme", "type": "probability_strategy"},
        ),
        TaskItem(
            id="extreme_trick_03", category=TaskCategory.REASONING,
            prompt=(
                "You have 100 doors in a row, all initially closed. You make 100 passes. "
                "On pass k (k=1,2,...,100), you toggle every k-th door (open→closed or closed→open). "
                "After all 100 passes: "
                "(1) Which doors are open? State the pattern and list ALL open doors. "
                "(2) Exactly how many doors are open? "
                "(3) PROVE why this pattern occurs using number theory. "
                "(4) Is door 72 open or closed? Is door 81 open or closed?"
            ),
            reference=(
                "(1) Open doors: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 — the perfect squares. "
                "(2) 10 doors are open (the perfect squares ≤ 100). "
                "(3) Proof: Door n is toggled once for each divisor of n. A door ends open iff it's toggled "
                "an odd number of times, i.e., n has an odd number of divisors. Divisors come in pairs "
                "(d, n/d) unless d = n/d, i.e., d² = n. So n has an odd number of divisors iff n is a "
                "perfect square. "
                "(4) Door 72: 72 is not a perfect square (8² = 64, 9² = 81), so door 72 is CLOSED. "
                "Door 81: 81 = 9², so door 81 is OPEN."
            ),
            metadata={"difficulty": "extreme", "type": "number_theory"},
            objective_checks=[
                ("perfect square", 1.5),        # key insight
                (r"\b10\b.*doors?\b.*open", 1.0),  # 10 doors open
                (r"door 72.*closed", 1.0),      # door 72 closed
                (r"door 81.*open", 1.0),        # door 81 open
                ("divisor", 0.5),               # mentions divisors in proof
            ],
        ),

        # --- QA: Precise multi-fact with traps ---
        TaskItem(
            id="extreme_qa_01", category=TaskCategory.QA,
            prompt=(
                "Answer ALL of the following precisely. Do NOT guess — if unsure, say so: "
                "(1) How many bones does an adult human body have? "
                "(2) How many chromosomes do potatoes have? "
                "(3) What is the atomic number of Tungsten? "
                "(4) In what year was the first email sent? "
                "(5) How many US states border the Pacific Ocean? List them. "
                "(6) What is the smallest country in the world by area, and what is its approximate area in square kilometers? "
                "(7) How many time zones does Russia span?"
            ),
            reference=(
                "(1) 206 bones. "
                "(2) 48 chromosomes (2n=48, tetraploid). "
                "(3) Tungsten (W) has atomic number 74. "
                "(4) 1971 (Ray Tomlinson sent the first network email via ARPANET). "
                "(5) 5 states: Alaska, Washington, Oregon, California, Hawaii. "
                "(6) Vatican City, approximately 0.44 km². "
                "(7) 11 time zones."
            ),
            metadata={"difficulty": "extreme", "type": "multi_fact_precision"},
            objective_checks=[
                (r"\b206\b", 1.0),              # (1) bones
                (r"\b48\b", 1.0),               # (2) potato chromosomes
                (r"\b74\b", 1.0),               # (3) tungsten atomic number
                (r"\b1971\b", 1.0),             # (4) first email year
                ("Alaska", 0.5),                # (5) Pacific states
                ("Hawaii", 0.5),
                ("Washington", 0.3),
                ("Oregon", 0.3),
                ("California", 0.4),
                (r"[Vv]atican", 1.0),           # (6) smallest country
                (r"\b11\b.*time.?zone", 1.0),   # (7) Russia timezones
            ],
        ),
        TaskItem(
            id="extreme_qa_02", category=TaskCategory.QA,
            prompt=(
                "For each of the following commonly confused pairs, explain the PRECISE technical difference. "
                "Do not just give definitions — explain what specifically distinguishes them: "
                "(1) Accuracy vs Precision "
                "(2) Latency vs Throughput "
                "(3) Concurrency vs Parallelism "
                "(4) Authentication vs Authorization "
                "(5) Encoding vs Encryption "
                "(6) Compilation vs Interpretation "
                "(7) Stack vs Heap memory allocation "
                "For each pair, give ONE concrete scenario where one is high but the other is low."
            ),
            reference=(
                "(1) Accuracy = how close to the true value; Precision = how close repeated measurements are to each other. "
                "Scenario: A biased scale always reads 5g too heavy — precise (consistent) but inaccurate. "
                "(2) Latency = time for single request; Throughput = requests per time unit. "
                "Scenario: A batch system processes 10,000 req/s but each takes 30s — high throughput, high latency. "
                "(3) Concurrency = dealing with multiple tasks at once (structure); Parallelism = doing them simultaneously (execution). "
                "Scenario: Single-core async web server — concurrent (handles many connections) but not parallel (one CPU). "
                "(4) Authentication = verifying who you are; Authorization = verifying what you can do. "
                "Scenario: You log in (authenticated) but can't access admin page (not authorized). "
                "(5) Encoding = format transformation (reversible, no secret); Encryption = confidentiality (requires key). "
                "Scenario: Base64 is encoding — anyone can decode it. AES is encryption — only key holder can decrypt. "
                "(6) Compilation = translating entire source to machine code before execution; Interpretation = executing line by line. "
                "Scenario: C compiles to binary (fast execution, slow compile); Python interprets (fast startup, slower execution). "
                "(7) Stack = automatic, LIFO, fixed size, fast; Heap = manual/GC, arbitrary, dynamic, slower. "
                "Scenario: Local int variable → stack (dies when function returns). malloc'd array → heap (persists until freed)."
            ),
            metadata={"difficulty": "extreme", "type": "precise_distinctions"},
        ),

        # --- CODE: Tricky edge-case-heavy problems ---
        TaskItem(
            id="extreme_code_01", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `to_roman(n: int) -> str` AND `from_roman(s: str) -> int` "
                "that convert between integers (1-3999) and Roman numerals. "
                "Must handle subtractive notation (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900). "
                "from_roman must validate input and raise ValueError for invalid Roman numerals "
                "like 'IIII', 'VV', 'IC', 'IM', 'VX'. "
                "Specifically handle these validation rules: "
                "- I, X, C can repeat max 3 times consecutively "
                "- V, L, D cannot repeat "
                "- I can only subtract from V and X "
                "- X can only subtract from L and C "
                "- C can only subtract from D and M "
                "Test: to_roman(1994) → 'MCMXCIV', from_roman('MCMXCIV') → 1994, "
                "from_roman('IIII') → raises ValueError"
            ),
            reference=None,
            metadata={"difficulty": "extreme", "language": "python"},
        ),
        TaskItem(
            id="extreme_code_02", category=TaskCategory.CODE,
            prompt=(
                "Write a Python function `calculate(expression: str) -> float` that evaluates "
                "expressions with +, -, *, /, ** (exponentiation), unary minus, parentheses, "
                "and the functions sin, cos, sqrt, abs, log (natural log). "
                "Operator precedence: () > functions > ** > unary- > *,/ > +,- "
                "** is RIGHT-associative (2**3**2 = 2**9 = 512, not 8**2 = 64). "
                "Do NOT use eval(). "
                "Examples: "
                "calculate('2 + 3 * 4') → 14.0 "
                "calculate('2 ** 3 ** 2') → 512.0 "
                "calculate('sin(0) + cos(0)') → 1.0 "
                "calculate('sqrt(144) + log(1)') → 12.0 "
                "calculate('-(-3)') → 3.0"
            ),
            reference=None,
            metadata={"difficulty": "extreme", "language": "python"},
        ),
        TaskItem(
            id="extreme_code_03", category=TaskCategory.CODE,
            prompt=(
                "Implement a Python class `Trie` with these methods: "
                "- insert(word: str) — insert a word "
                "- search(word: str) -> bool — exact match "
                "- starts_with(prefix: str) -> bool — any word starts with prefix "
                "- delete(word: str) -> bool — delete word, return True if existed "
                "- autocomplete(prefix: str, limit: int = 5) -> list[str] — return up to `limit` words with given prefix, sorted alphabetically "
                "- count_prefix(prefix: str) -> int — how many words start with prefix "
                "- wildcard_search(pattern: str) -> list[str] — '.' matches any single character. Return all matching words sorted. "
                "Example: after inserting 'cat', 'car', 'card', 'care', 'dog': "
                "wildcard_search('ca.') → ['car', 'cat'] "
                "wildcard_search('c..d') → ['card'] "
                "autocomplete('car', 3) → ['car', 'card', 'care']"
            ),
            reference=None,
            metadata={"difficulty": "extreme", "language": "python"},
        ),

        # --- CREATIVE: Extremely constrained ---
        TaskItem(
            id="extreme_creative_01", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a 200-word technical explanation of how a hash table works, "
                "but you may NOT use any of these words: hash, table, key, value, bucket, "
                "array, index, collision, map, dictionary, store, slot, function. "
                "The explanation must still be technically precise enough that a CS student "
                "could implement one from your description alone."
            ),
            reference=None,
            metadata={"difficulty": "extreme", "criteria": "None of the 13 forbidden words used, technically precise and implementable, ~200 words."},
        ),
        TaskItem(
            id="extreme_creative_02", category=TaskCategory.CREATIVE,
            prompt=(
                "Write a 6-line poem where: "
                "- Line 1 has exactly 3 words "
                "- Line 2 has exactly 5 words "
                "- Line 3 has exactly 7 words "
                "- Line 4 has exactly 7 words "
                "- Line 5 has exactly 5 words "
                "- Line 6 has exactly 3 words "
                "Lines 1 and 6 must be the same words but in reverse order. "
                "The poem must be about time travel and each line must make grammatical sense."
            ),
            reference=None,
            metadata={"difficulty": "extreme", "criteria": "Exact word counts per line, lines 1&6 reversed, about time travel, grammatically correct."},
        ),

        # --- REASONING: Complex logic and constraint satisfaction ---
        TaskItem(
            id="extreme_logic_01", category=TaskCategory.REASONING,
            prompt=(
                "Four friends (Anna, Ben, Cara, Dan) each have a different pet (cat, dog, fish, bird) "
                "and a different favorite color (red, blue, green, yellow). "
                "Clues: "
                "1. Anna's pet is not the cat or the dog. "
                "2. The person with the fish likes blue. "
                "3. Ben likes green. "
                "4. Cara does not have the bird. "
                "5. The person with the dog likes yellow. "
                "6. Dan does not like red. "
                "7. Anna does not like blue. "
                "Determine each person's pet and favorite color. "
                "Show your logical deduction step by step, explicitly tracking what is eliminated at each step."
            ),
            reference=(
                "From clue 3: Ben → green. "
                "From clue 1: Anna has fish or bird. "
                "From clue 7: Anna ≠ blue. From clue 2: fish → blue. So if Anna had fish, Anna → blue, contradiction. So Anna has bird. "
                "From clue 4: Cara ≠ bird (already Anna's). Cara has cat, dog, or fish. "
                "From clue 2: fish → blue. Ben → green (≠ blue), so Ben ≠ fish. Anna has bird. So fish is Cara or Dan. "
                "From clue 5: dog → yellow. Ben → green (≠ yellow), so Ben ≠ dog. Anna has bird. "
                "So Ben has cat or fish. But Ben ≠ fish (above). So Ben has cat. "
                "Remaining pets: dog and fish for Cara and Dan. "
                "From clue 4: Cara ≠ bird (already assigned). Cara can have dog or fish. "
                "From clue 5: dog → yellow. From clue 2: fish → blue. "
                "From clue 6: Dan ≠ red. "
                "If Cara has dog → Cara → yellow. Dan has fish → Dan → blue. "
                "Remaining color: red for Anna. Dan → blue (not red) ✓. Check clue 6: Dan → blue ≠ red ✓. "
                "Answer: Anna → bird, red. Ben → cat, green. Cara → dog, yellow. Dan → fish, blue."
            ),
            metadata={"difficulty": "extreme", "type": "constraint_satisfaction"},
        ),
        TaskItem(
            id="extreme_logic_02", category=TaskCategory.REASONING,
            prompt=(
                "A truth machine processes binary strings. It applies these rules SIMULTANEOUSLY to "
                "every bit in each step: "
                "- If a bit AND its right neighbor are both 1, the bit becomes 0 in the next step. "
                "- If a bit is 0 and its right neighbor is 1, the bit becomes 1 in the next step. "
                "- Otherwise, the bit stays the same. "
                "- The rightmost bit has no right neighbor (treat as 0). "
                "Starting string: 1 1 0 1 1 0 1 0 "
                "Compute the string after step 1, step 2, step 3, and step 4. "
                "Show your work for EACH bit at EACH step."
            ),
            reference=(
                "Initial: 1 1 0 1 1 0 1 0 "
                "Step 1: Evaluate each bit (using right neighbor): "
                "  Bit 0 (1, right=1): both 1 → 0. "
                "  Bit 1 (1, right=0): neither rule → stays 1. "
                "  Bit 2 (0, right=1): 0 and right=1 → 1. "
                "  Bit 3 (1, right=1): both 1 → 0. "
                "  Bit 4 (1, right=0): stays 1. "
                "  Bit 5 (0, right=1): 0 and right=1 → 1. "
                "  Bit 6 (1, right=0): stays 1. "
                "  Bit 7 (0, right=∅→0): stays 0. "
                "After step 1: 0 1 1 0 1 1 1 0 "
                "Step 2: "
                "  Bit 0 (0, right=1): → 1. "
                "  Bit 1 (1, right=1): → 0. "
                "  Bit 2 (1, right=0): stays 1. "
                "  Bit 3 (0, right=1): → 1. "
                "  Bit 4 (1, right=1): → 0. "
                "  Bit 5 (1, right=1): → 0. "
                "  Bit 6 (1, right=0): stays 1. "
                "  Bit 7 (0, right=∅→0): stays 0. "
                "After step 2: 1 0 1 1 0 0 1 0 "
                "Step 3: "
                "  Bit 0 (1, right=0): stays 1. "
                "  Bit 1 (0, right=1): → 1. "
                "  Bit 2 (1, right=1): → 0. "
                "  Bit 3 (1, right=0): stays 1. "
                "  Bit 4 (0, right=0): stays 0. "
                "  Bit 5 (0, right=1): → 1. "
                "  Bit 6 (1, right=0): stays 1. "
                "  Bit 7 (0, right=∅→0): stays 0. "
                "After step 3: 1 1 0 1 0 1 1 0 "
                "Step 4: "
                "  Bit 0 (1, right=1): → 0. "
                "  Bit 1 (1, right=0): stays 1. "
                "  Bit 2 (0, right=1): → 1. "
                "  Bit 3 (1, right=0): stays 1. "
                "  Bit 4 (0, right=1): → 1. "
                "  Bit 5 (1, right=1): → 0. "
                "  Bit 6 (1, right=0): stays 1. "
                "  Bit 7 (0, right=∅→0): stays 0. "
                "After step 4: 0 1 1 1 1 0 1 0"
            ),
            metadata={"difficulty": "extreme", "type": "simulation"},
        ),
        TaskItem(
            id="extreme_logic_03", category=TaskCategory.REASONING,
            prompt=(
                "A function f is defined on positive integers as follows: "
                "f(1) = 1 "
                "f(2n) = f(n) + 1 "
                "f(2n+1) = f(n) + f(n+1) "
                "Calculate f(1) through f(20). Show your work for each, clearly indicating "
                "which prior values you used. Then find f(100)."
            ),
            reference=(
                "f(1) = 1. "
                "f(2) = f(2×1) = f(1) + 1 = 2. "
                "f(3) = f(2×1+1) = f(1) + f(2) = 1 + 2 = 3. "
                "f(4) = f(2×2) = f(2) + 1 = 3. "
                "f(5) = f(2×2+1) = f(2) + f(3) = 2 + 3 = 5. "
                "f(6) = f(2×3) = f(3) + 1 = 4. "
                "f(7) = f(2×3+1) = f(3) + f(4) = 3 + 3 = 6. "
                "f(8) = f(2×4) = f(4) + 1 = 4. "
                "f(9) = f(2×4+1) = f(4) + f(5) = 3 + 5 = 8. "
                "f(10) = f(2×5) = f(5) + 1 = 6. "
                "f(11) = f(2×5+1) = f(5) + f(6) = 5 + 4 = 9. "
                "f(12) = f(2×6) = f(6) + 1 = 5. "
                "f(13) = f(2×6+1) = f(6) + f(7) = 4 + 6 = 10. "
                "f(14) = f(2×7) = f(7) + 1 = 7. "
                "f(15) = f(2×7+1) = f(7) + f(8) = 6 + 4 = 10. "
                "f(16) = f(2×8) = f(8) + 1 = 5. "
                "f(17) = f(2×8+1) = f(8) + f(9) = 4 + 8 = 12. "
                "f(18) = f(2×9) = f(9) + 1 = 9. "
                "f(19) = f(2×9+1) = f(9) + f(10) = 8 + 6 = 14. "
                "f(20) = f(2×10) = f(10) + 1 = 7. "
                "For f(100): 100=2×50, f(100)=f(50)+1. 50=2×25, f(50)=f(25)+1. "
                "25=2×12+1, f(25)=f(12)+f(13)=5+10=15. f(50)=16. f(100)=17."
            ),
            metadata={"difficulty": "extreme", "type": "recursive_computation"},
            objective_checks=[
                # Check key f(n) values that models commonly get wrong
                (r"f\(5\)\s*=\s*5", 1.0),       # f(5) = 5
                (r"f\(9\)\s*=\s*8", 1.0),       # f(9) = 8
                (r"f\(13\)\s*=\s*10", 1.0),     # f(13) = 10
                (r"f\(17\)\s*=\s*12", 1.0),     # f(17) = 12
                (r"f\(20\)\s*=\s*7", 1.0),      # f(20) = 7
                (r"f\(100\)\s*=\s*17", 2.0),    # f(100) = 17 (hardest)
            ],
        ),
        TaskItem(
            id="extreme_calc_04", category=TaskCategory.REASONING,
            prompt=(
                "A mortgage has a principal of $347,000, an annual interest rate of 6.875%, "
                "and a term of 30 years (360 monthly payments). "
                "(1) Calculate the exact monthly payment using the amortization formula: "
                "M = P × r(1+r)^n / ((1+r)^n - 1), where r is monthly rate. "
                "(2) What is the total amount paid over the life of the loan? "
                "(3) How much of the FIRST payment goes to interest vs principal? "
                "(4) How much of the 120th payment (month 120) goes to interest vs principal? "
                "For (4), you need to figure out the remaining balance after 119 payments first. "
                "Show all intermediate calculations."
            ),
            reference=(
                "r = 6.875% / 12 = 0.572917% = 0.00572917. n = 360. "
                "(1) (1+r)^360 = 1.00572917^360 ≈ 7.8964. "
                "M = 347000 × 0.00572917 × 7.8964 / (7.8964 - 1) = 347000 × 0.045253 / 6.8964 "
                "= 15,702.77 / 6.8964 ≈ $2,276.91. "
                "(2) Total paid = 2,276.91 × 360 = $819,687.60. "
                "(3) First payment interest = 347,000 × 0.00572917 = $1,987.82. "
                "Principal = 2,276.91 - 1,987.82 = $289.09. "
                "(4) Remaining balance after k payments: B(k) = P × ((1+r)^n - (1+r)^k) / ((1+r)^n - 1). "
                "B(119) = 347,000 × (7.8964 - 1.00572917^119) / 6.8964. "
                "1.00572917^119 ≈ 1.9776. B(119) = 347,000 × (7.8964 - 1.9776) / 6.8964 "
                "= 347,000 × 5.9188 / 6.8964 ≈ $297,836. "
                "Interest in 120th payment = 297,836 × 0.00572917 ≈ $1,706.26. "
                "Principal = 2,276.91 - 1,706.26 = $570.65."
            ),
            metadata={"difficulty": "extreme", "type": "financial_calculation"},
        ),
        TaskItem(
            id="extreme_qa_03", category=TaskCategory.QA,
            prompt=(
                "For each of the following, state whether the claim is TRUE, FALSE, or MISLEADING, "
                "and explain precisely why in 1-2 sentences: "
                "(1) 'Humans use only 10% of their brains.' "
                "(2) 'The Great Wall of China is visible from space.' "
                "(3) 'We have more bacterial cells than human cells.' "
                "(4) 'Lightning never strikes the same place twice.' "
                "(5) 'Goldfish have a 3-second memory.' "
                "(6) 'Eskimos have 50 words for snow.' "
                "(7) 'Dropping a penny from the Empire State Building could kill someone.' "
                "(8) 'Bats are blind.' "
                "(9) 'You lose most body heat through your head.' "
                "(10) 'Sugar causes hyperactivity in children.'"
            ),
            reference=(
                "(1) FALSE. Brain imaging shows all areas are active; different regions activate for different tasks. No large 'unused' portion. "
                "(2) FALSE/MISLEADING. Not visible to naked eye from low Earth orbit. Other structures (highways, cities) are more visible. "
                "(3) MISLEADING. Recent estimates suggest roughly 1:1 ratio (about 30 trillion each), not 10:1 as previously claimed. "
                "(4) FALSE. Lightning frequently strikes the same place — tall structures like the Empire State Building are struck ~20-25 times/year. "
                "(5) FALSE. Goldfish can remember things for months. Studies show they can learn and retain associations. "
                "(6) MISLEADING. Depends on how you count. Inuit/Yupik languages are polysynthetic — they create compound words freely. English also has many snow words (slush, sleet, powder, flurry, etc.). "
                "(7) FALSE. Terminal velocity of a penny (~30-50 mph) is too low to cause lethal injury. "
                "(8) FALSE. Bats can see. Most use echolocation as primary navigation but also have functioning eyes. "
                "(9) MISLEADING. You lose heat proportional to exposed surface area. Head is ~10% of body surface. The myth comes from a flawed 1950s military study where subjects wore insulated suits but no hats. "
                "(10) FALSE. Multiple double-blind studies show no link. Parents who believe sugar causes hyperactivity rate their children as more hyperactive regardless of actual sugar consumption."
            ),
            metadata={"difficulty": "extreme", "type": "myth_busting"},
        ),

        # --- COUNTING AND SPATIAL — LLMs are terrible at these ---
        TaskItem(
            id="extreme_count_01", category=TaskCategory.REASONING,
            prompt=(
                "Count the number of times the letter 'e' appears in the following paragraph: "
                "'The emergence of new technologies has led to unprecedented levels of excellence "
                "in every enterprise. Nevertheless, there remain several severe challenges that "
                "require perseverance and extreme dedication. Engineers everywhere believe that "
                "these endeavors deserve more resources and better expertise.' "
                "Give the exact count. List every word containing 'e' and how many 'e's it has."
            ),
            reference=(
                "Let's count each word: The(1) emergence(3) of(0) new(0) technologies(1) has(0) "
                "led(1) to(0) unprecedented(2) levels(2) of(0) excellence(3) in(0) every(1) "
                "enterprise(3). Nevertheless(2), there(2) remain(0) several(1) severe(2) "
                "challenges(1) that(0) require(1) perseverance(3) and(0) extreme(2) dedication(1). "
                "Engineers(2) everywhere(3) believe(2) that(0) these(2) endeavors(1) deserve(2) "
                "more(1) resources(2) and(0) better(2) expertise(2). "
                "Total: 1+3+1+1+2+2+3+1+3+2+2+1+2+1+1+3+2+1+2+3+2+2+1+2+1+2+2 = 48 e's."
            ),
            metadata={"difficulty": "extreme", "type": "counting"},
        ),
        TaskItem(
            id="extreme_count_02", category=TaskCategory.REASONING,
            prompt=(
                "In the word 'MISSISSIPPI': "
                "(1) How many letters total? "
                "(2) How many distinct letters? List them with their counts. "
                "(3) How many ways can you arrange ALL 11 letters? (multinomial coefficient) "
                "(4) If you pick 4 letters at random WITHOUT replacement, what is the probability "
                "that all 4 are the same letter? "
                "(5) How many 4-letter combinations (not permutations) can be formed using only "
                "the letters available (respecting counts)? "
                "Show all calculations."
            ),
            reference=(
                "(1) 11 letters total. "
                "(2) 4 distinct: M=1, I=4, S=4, P=2. "
                "(3) 11! / (1! × 4! × 4! × 2!) = 39916800 / (1 × 24 × 24 × 2) = 39916800 / 1152 = 34,650. "
                "(4) Only I and S have 4+ letters. P(all same) = P(all I) + P(all S). "
                "P(all I) = C(4,4)/C(11,4) = 1/330. P(all S) = C(4,4)/C(11,4) = 1/330. "
                "Total = 2/330 = 1/165 ≈ 0.00606. "
                "(5) Need to enumerate 4-letter multisets from {M:1, I:4, S:4, P:2}. "
                "This requires generating all (m,i,s,p) with m≤1, i≤4, s≤4, p≤2, m+i+s+p=4. "
                "Systematic enumeration gives 15 combinations."
            ),
            metadata={"difficulty": "extreme", "type": "combinatorics_counting"},
        ),

        # --- BASE CONVERSION / BIT MANIPULATION ---
        TaskItem(
            id="extreme_bits_01", category=TaskCategory.REASONING,
            prompt=(
                "Convert the decimal number 2847 to: "
                "(1) Binary "
                "(2) Octal "
                "(3) Hexadecimal "
                "Then take the hexadecimal result and compute: "
                "(4) What is 0xRESULT + 0x1F3 in hexadecimal? "
                "(5) What is 0xRESULT AND 0xFF (bitwise AND) in both hex and decimal? "
                "Show all work including intermediate divisions/remainders."
            ),
            reference=(
                "(1) 2847 in binary: 2847÷2=1423r1, 1423÷2=711r1, 711÷2=355r1, 355÷2=177r1, "
                "177÷2=88r1, 88÷2=44r0, 44÷2=22r0, 22÷2=11r0, 11÷2=5r1, 5÷2=2r1, 2÷2=1r0, 1÷2=0r1. "
                "Reading remainders bottom-up: 101100011111. "
                "(2) Group binary in 3s from right: 101 100 011 111 = 5 4 3 7 → octal 5437. "
                "(3) Group binary in 4s from right: 1011 0001 1111 = B 1 F → hex 0xB1F. "
                "(4) 0xB1F + 0x1F3: F+3=12→2 carry 1, 1+F+1=11→1 carry 1, B+1+1=D. Result: 0xD12. "
                "(5) 0xB1F AND 0xFF = 0x1F (only last byte kept). 0x1F = 31 decimal."
            ),
            metadata={"difficulty": "extreme", "type": "base_conversion"},
        ),

        # --- SELF-REFERENTIAL / META ---
        TaskItem(
            id="extreme_meta_01", category=TaskCategory.REASONING,
            prompt=(
                "Compute the following WITHOUT a calculator. Show all intermediate steps: "
                "(1) 37 × 43 × 7 "
                "(2) 123456 mod 789 (the remainder when dividing 123456 by 789) "
                "(3) What is the 8th Fibonacci number? (F1=1, F2=1) "
                "(4) How many prime numbers are there between 100 and 150? List them all. "
                "(5) What is the greatest common divisor of 462 and 1071? Use the Euclidean algorithm. "
                "(6) Compute 3^7 (three to the seventh power). "
                "(7) What is the sum of all ODD integers from 1 to 99?"
            ),
            reference=(
                "(1) 37 × 43 = 1591. 1591 × 7 = 11137. "
                "(2) 123456 ÷ 789 = 156 remainder 372. (789 × 156 = 123084, 123456 - 123084 = 372). "
                "(3) F1=1, F2=1, F3=2, F4=3, F5=5, F6=8, F7=13, F8=21. The 8th is 21. "
                "(4) Primes between 100-150: 101, 103, 107, 109, 113, 127, 131, 137, 139, 149. That's 10 primes. "
                "(5) GCD(1071, 462): 1071 = 2×462 + 147. 462 = 3×147 + 21. 147 = 7×21 + 0. GCD = 21. "
                "(6) 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187. "
                "(7) Odd integers 1 to 99: there are 50 of them. Sum = 50^2 = 2500."
            ),
            metadata={"difficulty": "extreme", "type": "multi_arithmetic"},
            objective_checks=[
                (r"\b11,?137\b", 1.0),      # (1) 37×43×7
                (r"\b372\b", 1.0),           # (2) 123456 mod 789
                (r"\b21\b", 0.5),            # (3) 8th Fibonacci (common number, lower weight)
                (r"\b10\b.*prime", 0.5),     # (4) 10 primes
                (r"\b149\b", 0.5),           # (4) includes 149
                (r"\b127\b", 0.5),           # (4) includes 127
                (r"GCD.*21|gcd.*21", 1.0),   # (5) GCD = 21
                (r"\b2,?187\b", 1.0),        # (6) 3^7
                (r"\b2,?500\b", 1.0),        # (7) sum of odds
            ],
        ),

        # --- MULTI-CONSTRAINT WORD PROBLEM ---
        TaskItem(
            id="extreme_schedule_01", category=TaskCategory.REASONING,
            prompt=(
                "Schedule 5 meetings (A, B, C, D, E) into 3 rooms (R1, R2, R3) across "
                "4 time slots (9am, 10am, 11am, 12pm). Constraints: "
                "1. Meeting A must be at 9am. "
                "2. Meeting B must be after Meeting A but before Meeting D. "
                "3. Meeting C and Meeting E cannot be in the same room. "
                "4. Meeting D must be in R1. "
                "5. No room can have more than 2 meetings. "
                "6. Meeting E must be at 11am or 12pm. "
                "7. Meeting B and Meeting C must be in different time slots. "
                "8. R3 is only available at 9am and 10am. "
                "Find ONE valid schedule. Show that all 8 constraints are satisfied."
            ),
            reference=(
                "One valid schedule: "
                "A: R1, 9am. B: R2, 10am. C: R3, 9am. D: R1, 11am. E: R2, 12pm. "
                "Check: (1) A at 9am ✓ (2) B(10am) after A(9am), before D(11am) ✓ "
                "(3) C in R3, E in R2 — different rooms ✓ (4) D in R1 ✓ "
                "(5) R1 has A,D (2), R2 has B,E (2), R3 has C (1) — all ≤2 ✓ "
                "(6) E at 12pm ✓ (7) B at 10am, C at 9am — different ✓ "
                "(8) R3 used at 9am only ✓."
            ),
            metadata={"difficulty": "extreme", "type": "scheduling"},
        ),

        # --- CHAIN-OF-THOUGHT TRAP ---
        TaskItem(
            id="extreme_trap_01", category=TaskCategory.REASONING,
            prompt=(
                "Read carefully before answering: "
                "A farmer has 15 sheep. 8 of them are NOT white. "
                "How many white sheep does the farmer have? "
                "Now: the farmer buys 6 MORE sheep. 2 of the new sheep are white. "
                "A wolf eats 3 sheep (all non-white). "
                "Then 1 white sheep has 2 lambs (both white). "
                "Finally, the farmer sells half of his NON-white sheep. "
                "How many total sheep does the farmer have at the end? "
                "How many are white? How many are non-white? "
                "Track each step."
            ),
            reference=(
                "Start: 15 sheep. 8 not white → 7 white, 8 non-white. "
                "Buy 6 more (2 white, 4 non-white): 7+2=9 white, 8+4=12 non-white. Total: 21. "
                "Wolf eats 3 non-white: 9 white, 12-3=9 non-white. Total: 18. "
                "1 white sheep has 2 white lambs: 9+2=11 white, 9 non-white. Total: 20. "
                "Sell half non-white: 11 white, 9/2=4.5→ let's say floor: 4 sold, 5 remain. "
                "But 9 is odd — half of 9 non-white sheep. If 'half' means 4 (floor of 4.5): "
                "11 white + 5 non-white = 16 total. "
                "If 'half' rounds up to 5: 11 white + 4 non-white = 15 total. "
                "Most likely intended: sells 4 non-white (floor of half). "
                "Final: 16 sheep total, 11 white, 5 non-white."
            ),
            metadata={"difficulty": "extreme", "type": "state_tracking"},
        ),
    ]
