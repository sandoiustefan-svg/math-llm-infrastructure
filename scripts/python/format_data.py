from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.format_openmathinstruct2 import (
    FormatConfig,
    format_openmathinstruct2_example,
)


def main() -> None:
    """
    Read a raw OpenMathInstruct-2 JSONL file, format each example into
    Llama chat-style messages, and save the transformed dataset as JSONL.

    Input format:
        One JSON object per line containing raw OpenMathInstruct-2 fields
        such as:
            - problem
            - generated_solution
            - expected_answer

    Output format:
        One JSON object per line containing:
            - messages
            - prompt_messages
            - completion_text

    Command-line arguments:
        --input
            Path to the source JSONL file.

        --output
            Path where the formatted JSONL file will be written.

        --no-system-prompt
            Disable insertion of the system role message.

        --no-final-answer
            Disable appending the final answer section to the assistant reply.
    """
    parser = argparse.ArgumentParser(
        description="Format OpenMathInstruct-2 JSONL into Llama chat format."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/inspect/openmathinstruct2_train.jsonl"),
        help="Path to raw input JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inspect/openmathinstruct2_train_formatted.jsonl"),
        help="Path to save formatted JSONL file.",
    )

    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable the system prompt message.",
    )

    parser.add_argument(
        "--no-final-answer",
        action="store_true",
        help="Do not append final answer to assistant response.",
    )

    args = parser.parse_args()

    cfg = FormatConfig(
        include_system_prompt=not args.no_system_prompt,
        include_final_answer=not args.no_final_answer,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:

        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            try:
                raw_example = json.loads(line)

                formatted_example = format_openmathinstruct2_example(
                    raw_example,
                    cfg,
                )

                if not formatted_example["messages"][-1]["content"].strip():
                    skipped += 1
                    continue

                fout.write(
                    json.dumps(
                        formatted_example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                written += 1

            except Exception as exc:
                print(f"Skipping line {line_number}: {exc}")
                skipped += 1

    print(f"Done. Written: {written}, skipped: {skipped}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()