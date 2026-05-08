# Prompt Formats

All three prompt styles share the same system prompt used during training and
produce a list of chat messages that is rendered by the Llama 3 tokenizer via
`apply_chat_template`. The rendered text is what actually reaches the model.

The example problem used throughout:

> *Janet's ducks lay 16 eggs per day. She eats three for breakfast every
> morning and bakes muffins for her friends every day with four. She sells
> the remainder at the farmers' market daily for $2 per fresh duck egg. How
> much in dollars does she make every day at the farmers' market?*
>
> Expected answer: **18**

---

## Zero-shot

**Message list**

```python
[
    {"role": "system", "content": "You are a careful mathematical reasoning assistant. Solve the problem step by step."},
    {"role": "user",   "content": "<problem>"},
]
```

**Rendered prompt (sent to model)**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful mathematical reasoning assistant. Solve the problem step by step.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

The model generates from the trailing `assistant` header onwards.

---

## Chain-of-Thought (CoT)

Three fixed hand-written examples are prepended as `user`/`assistant` turns
before the test problem. The assistant turns follow the exact training format:
step-by-step reasoning ending with `Final Answer: {answer}`.

**Message list**

```python
[
    {"role": "system",    "content": "You are a careful mathematical reasoning assistant. Solve the problem step by step."},
    # --- example 1 ---
    {"role": "user",      "content": "A store has 50 apples. They sell 23 in the morning and receive a delivery of 15 more in the afternoon. How many apples does the store have at the end of the day?"},
    {"role": "assistant", "content": "Let me solve this step by step.\nThe store starts with 50 apples.\nAfter selling 23 in the morning: 50 - 23 = 27 apples.\nAfter receiving a delivery of 15: 27 + 15 = 42 apples.\n\nFinal Answer: 42"},
    # --- example 2 ---
    {"role": "user",      "content": "Sarah earns $12 per hour and works 8 hours a day, 5 days a week. How much does she earn in a week?"},
    {"role": "assistant", "content": "Let me solve this step by step.\nSarah earns $12 per hour.\nShe works 8 hours per day: 12 × 8 = $96 per day.\nShe works 5 days a week: 96 × 5 = $480 per week.\n\nFinal Answer: 480"},
    # --- example 3 ---
    {"role": "user",      "content": "A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes?"},
    {"role": "assistant", "content": "Let me solve this step by step.\nConvert time to hours: 2 hours 30 minutes = 2.5 hours.\nDistance = speed × time: 60 × 2.5 = 150 miles.\n\nFinal Answer: 150"},
    # --- test problem ---
    {"role": "user",      "content": "<problem>"},
]
```

**Rendered prompt (sent to model)**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful mathematical reasoning assistant. Solve the problem step by step.
<|eot_id|><|start_header_id|>user<|end_header_id|>

A store has 50 apples. They sell 23 in the morning and receive a delivery of 15 more in the afternoon. How many apples does the store have at the end of the day?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
The store starts with 50 apples.
After selling 23 in the morning: 50 - 23 = 27 apples.
After receiving a delivery of 15: 27 + 15 = 42 apples.

Final Answer: 42
<|eot_id|><|start_header_id|>user<|end_header_id|>

Sarah earns $12 per hour and works 8 hours a day, 5 days a week. How much does she earn in a week?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
Sarah earns $12 per hour.
She works 8 hours per day: 12 × 8 = $96 per day.
She works 5 days a week: 96 × 5 = $480 per week.

Final Answer: 480
<|eot_id|><|start_header_id|>user<|end_header_id|>

A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
Convert time to hours: 2 hours 30 minutes = 2.5 hours.
Distance = speed × time: 60 × 2.5 = 150 miles.

Final Answer: 150
<|eot_id|><|start_header_id|>user<|end_header_id|>

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

---

## Key differences

| | Zero-shot | CoT |
|---|---|---|
| Few-shot examples | none | 3 fixed |
| Example source | — | hand-written |
| Example selection | — | always the same |
| Prompt length | shortest | medium (~500 tokens) |
| Expected benefit | baseline | structured reasoning |

Both formats produce the same expected output structure from the model:
step-by-step reasoning ending with `Final Answer: {answer}`.
