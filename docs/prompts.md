# Prompt Formats

All three prompt variants produce a list of chat messages rendered by the Llama 3
tokenizer via `apply_chat_template`. Every variant instructs the model to annotate
arithmetic operations inline as `<<expr=result>>` (e.g. `3 × 4 = <<3*4=12>>12`),
making all three comparable under the same three metrics: binary correctness,
embedding similarity, and arithmetic step correctness.

The example problem used throughout:

> *Janet's ducks lay 16 eggs per day. She eats three for breakfast every
> morning and bakes muffins for her friends every day with four. She sells
> the remainder at the farmers' market daily for $2 per fresh duck egg. How
> much in dollars does she make every day at the farmers' market?*
>
> Expected answer: **18**

---

## Variant 1 — Zero-shot

No examples. The model must follow the `<<expr=result>>` format from the system
prompt instruction alone.

**Message list**

```python
[
    {"role": "system", "content": (
        "You are a careful mathematical reasoning assistant. "
        "Solve the problem step by step. "
        "For every arithmetic operation write the expression and its result as "
        "<<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. "
        "End your response with \"Final Answer: {answer}\"."
    )},
    {"role": "user", "content": "<problem>"},
]
```

**Rendered prompt (sent to model)**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful mathematical reasoning assistant. Solve the problem step by step. For every arithmetic operation write the expression and its result as <<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. End your response with "Final Answer: {answer}".
<|eot_id|><|start_header_id|>user<|end_header_id|>

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

---

## Variant 2 — Few-shot (CoT)

Three fixed hand-written examples are prepended as `user`/`assistant` turns.
The assistant turns demonstrate the `<<expr=result>>` format in natural prose.

**Message list**

```python
[
    {"role": "system", "content": (
        "You are a careful mathematical reasoning assistant. "
        "Solve the problem step by step. "
        "For every arithmetic operation write the expression and its result as "
        "<<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. "
        "End your response with \"Final Answer: {answer}\"."
    )},
    # --- example 1 ---
    {"role": "user", "content": "A store has 50 apples. They sell 23 in the morning and receive a delivery of 15 more in the afternoon. How many apples does the store have at the end of the day?"},
    {"role": "assistant", "content": (
        "Let me solve this step by step.\n"
        "The store starts with 50 apples.\n"
        "After selling 23 in the morning: 50 - 23 = <<50-23=27>>27 apples.\n"
        "After receiving a delivery of 15: 27 + 15 = <<27+15=42>>42 apples.\n\n"
        "Final Answer: 42"
    )},
    # --- example 2 ---
    {"role": "user", "content": "Sarah earns $12 per hour and works 8 hours a day, 5 days a week. How much does she earn in a week?"},
    {"role": "assistant", "content": (
        "Let me solve this step by step.\n"
        "Sarah earns $12 per hour.\n"
        "She works 8 hours per day: 12 × 8 = <<12*8=96>>96 per day.\n"
        "She works 5 days a week: 96 × 5 = <<96*5=480>>480 per week.\n\n"
        "Final Answer: 480"
    )},
    # --- example 3 ---
    {"role": "user", "content": "A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes?"},
    {"role": "assistant", "content": (
        "Let me solve this step by step.\n"
        "Convert time to hours: 2 hours 30 minutes = 2.5 hours.\n"
        "Distance = speed × time: 60 × 2.5 = <<60*2.5=150>>150 miles.\n\n"
        "Final Answer: 150"
    )},
    # --- test problem ---
    {"role": "user", "content": "<problem>"},
]
```

---

## Variant 3 — Few-shot + Step-by-step

Same three examples as Variant 2 but each assistant turn uses explicit `Step N:`
labels, enforcing a more structured decomposition. The system prompt also
reinforces numbered steps.

**Message list**

```python
[
    {"role": "system", "content": (
        "You are a careful mathematical reasoning assistant. "
        "Break your solution into explicit numbered steps, one per line. "
        "For every arithmetic operation write the expression and its result as "
        "<<expr=result>> — for example, 3 × 4 = <<3*4=12>>12. "
        "End your response with \"Final Answer: {answer}\"."
    )},
    # --- example 1 ---
    {"role": "user", "content": "A store has 50 apples. They sell 23 in the morning and receive a delivery of 15 more in the afternoon. How many apples does the store have at the end of the day?"},
    {"role": "assistant", "content": (
        "Step 1: The store starts with 50 apples.\n"
        "Step 2: After the morning sale: 50 - 23 = <<50-23=27>>27 apples.\n"
        "Step 3: After the afternoon delivery: 27 + 15 = <<27+15=42>>42 apples.\n\n"
        "Final Answer: 42"
    )},
    # --- example 2 ---
    {"role": "user", "content": "Sarah earns $12 per hour and works 8 hours a day, 5 days a week. How much does she earn in a week?"},
    {"role": "assistant", "content": (
        "Step 1: Sarah earns $12 per hour.\n"
        "Step 2: Daily earnings: 12 × 8 = <<12*8=96>>96 per day.\n"
        "Step 3: Weekly earnings: 96 × 5 = <<96*5=480>>480 per week.\n\n"
        "Final Answer: 480"
    )},
    # --- example 3 ---
    {"role": "user", "content": "A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes?"},
    {"role": "assistant", "content": (
        "Step 1: Convert time to hours: 2 hours 30 minutes = 2.5 hours.\n"
        "Step 2: Distance = speed × time: 60 × 2.5 = <<60*2.5=150>>150 miles.\n\n"
        "Final Answer: 150"
    )},
    # --- test problem ---
    {"role": "user", "content": "<problem>"},
]
```

---

## Key differences

| | Zero-shot | Few-shot (CoT) | Few-shot + Step-by-step |
|---|---|---|---|
| Few-shot examples | none | 3 fixed | 3 fixed |
| `<<expr=result>>` format | instruction only | instruction + examples | instruction + examples |
| Step labels (`Step N:`) | none | none | explicit |
| Prompt length | shortest | medium (~500 tokens) | medium (~500 tokens) |
| Expected benefit | baseline | format adherence | structured decomposition |

All three variants are evaluated under the same three metrics:
binary correctness, embedding similarity, and arithmetic step correctness.
