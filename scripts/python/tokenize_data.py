from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.tokenizer import process_dataset


def main() -> None:
    """
    CLI launcher.
    """
    parser = argparse.ArgumentParser(
        description="Render or tokenize OpenMathInstruct-2 data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=["text", "tokens"],
        required=True,
    )

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name_or_path=args.model_name_or_path,
        mode=args.mode,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()