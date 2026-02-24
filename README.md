# math-llm-infrastructure

LLaMA-style training infrastructure for large-scale mathematical instruction datasets, including preprocessing, token packing, and distributed benchmarking.

---

## Environment Setup

Follow the steps below to set up the development environment.

### Run the following command that calls the env script

```bash
bash scripts/setup_env.sh
```

## Inspecting the Dataset

Before preprocessing, you can inspect the dataset structure using the inspection script.

### Run the following command that calls the data inspection script 

```bash
python scripts/inspect_data.py --limit 3
```

It has available three arguments:

```markdown
### Available arguments

| Argument  | Description |
|----------|-------------|
| `--split` | Dataset split to load (default: `train`) |
| `--limit` | Number of examples to display |
| `--skip`  | Number of examples to skip before reading |
```