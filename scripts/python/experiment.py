from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.experiments.registry import Experiment, ExperimentRegistry


def cmd_list(args, registry: ExperimentRegistry):
    registry.print_table()


def cmd_show(args, registry: ExperimentRegistry):
    registry.print_detail(args.id)


def cmd_register(args, registry: ExperimentRegistry):
    exp = Experiment(
        id="",
        name=args.name,
        type=args.type,
        description=args.description or "",
        notes=args.notes or "",
    )
    exp_id = registry.register(exp)
    print(f"Registered experiment: {exp_id}")


def cmd_note(args, registry: ExperimentRegistry):
    registry.add_note(args.id, args.text)
    print(f"Note added to {args.id}.")


def cmd_delete(args, registry: ExperimentRegistry):
    entry = registry._find(args.id)
    if entry is None:
        print(f"Experiment {args.id} not found.")
        sys.exit(1)
    registry._experiments.remove(entry)
    registry._save()
    print(f"Deleted experiment {args.id}.")


def main():
    ap = argparse.ArgumentParser(
        description="Manage the experiment registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/python/experiment.py list
  python scripts/python/experiment.py show exp_001
  python scripts/python/experiment.py register --type scratch_train --name "baseline-16L"
  python scripts/python/experiment.py note exp_001 "converged well, final loss 1.23"
  python scripts/python/experiment.py delete exp_001
""",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print all experiments in a table")

    p_show = sub.add_parser("show", help="Print full JSON detail for one experiment")
    p_show.add_argument("id", help="Experiment ID, e.g. exp_001")

    p_reg = sub.add_parser("register", help="Manually register a new experiment entry")
    p_reg.add_argument("--name", required=True, help="Short name for the experiment")
    p_reg.add_argument(
        "--type",
        required=True,
        choices=["scratch_train", "finetune", "uq_mc_dropout", "uq_ensemble"],
        help="Experiment type",
    )
    p_reg.add_argument("--description", default="", help="Longer description")
    p_reg.add_argument("--notes", default="", help="Initial notes")

    p_note = sub.add_parser("note", help="Append a timestamped note to an experiment")
    p_note.add_argument("id", help="Experiment ID")
    p_note.add_argument("text", help="Note text")

    p_del = sub.add_parser("delete", help="Remove an experiment from the registry")
    p_del.add_argument("id", help="Experiment ID to delete")

    args = ap.parse_args()
    registry = ExperimentRegistry()

    dispatch = {
        "list": cmd_list,
        "show": cmd_show,
        "register": cmd_register,
        "note": cmd_note,
        "delete": cmd_delete,
    }
    dispatch[args.command](args, registry)


if __name__ == "__main__":
    main()
