from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REGISTRY_DIR = Path(__file__).resolve().parents[2] / "experiments"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


@dataclass
class Experiment:
    """
    A single experiment entry in the registry.

    Fields are intentionally broad to cover training, fine-tuning, and UQ runs.
    Only `id`, `name`, and `type` are required; everything else is optional and
    filled in progressively as the run proceeds.
    """
    id: str
    name: str
    type: str                        # "scratch_train" | "finetune" | "uq_mc_dropout" | "uq_ensemble"
    description: str = ""
    status: str = "planned"          # "planned" | "running" | "completed" | "failed"
    created_at: str = ""
    completed_at: str = ""

    # Training config (populated for train/finetune runs)
    config: dict = None              # full TrainConfig as dict
    model_architecture: dict = None  # n_layers, hidden_size, n_heads, n_params
    dataset: dict = None             # tokenizer, data_dir, seq_len, total_steps

    # Results (populated on completion)
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    samples_consumed: Optional[int] = None
    checkpoint_path: str = ""

    # Linked files (relative to project root)
    metrics_json: str = ""
    training_plot: str = ""
    notes: str = ""

    # UQ results (populated for uq runs)
    uq_results: dict = None

    def __post_init__(self):
        if self.created_at == "":
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if self.config is None:
            self.config = {}
        if self.model_architecture is None:
            self.model_architecture = {}
        if self.dataset is None:
            self.dataset = {}
        if self.uq_results is None:
            self.uq_results = {}


class ExperimentRegistry:
    """
    Lightweight JSON-based experiment registry.

    All experiments are stored in experiments/registry.json at the project root.
    Plots and metrics files are referenced by path (not copied) to avoid duplication.

    Usage from trainer:
        registry = ExperimentRegistry()
        exp_id = registry.start_training(cfg, n_params)
        ...
        registry.complete_training(exp_id, final_loss, steps, samples, checkpoint_path)

    Usage from CLI (scripts/python/experiment.py):
        python scripts/python/experiment.py list
        python scripts/python/experiment.py show exp_001
        python scripts/python/experiment.py register --type scratch_train --name "baseline" ...
    """

    def __init__(self):
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self._experiments: list[dict] = []
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE) as f:
                self._experiments = json.load(f)

    def _save(self):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self._experiments, f, indent=2)

    def _next_id(self) -> str:
        n = len(self._experiments) + 1
        return f"exp_{n:03d}"

    def _find(self, exp_id: str) -> Optional[dict]:
        for e in self._experiments:
            if e["id"] == exp_id:
                return e
        return None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, exp: Experiment) -> str:
        """Add a new experiment. Returns its id."""
        if exp.id == "":
            exp.id = self._next_id()
        self._experiments.append(asdict(exp))
        self._save()
        return exp.id

    def update(self, exp_id: str, **fields: Any) -> None:
        """Update arbitrary fields on an existing experiment."""
        entry = self._find(exp_id)
        if entry is None:
            raise KeyError(f"Experiment {exp_id} not found")
        entry.update(fields)
        self._save()

    def start_training(self, cfg: Any, n_params: int) -> str:
        """
        Called at the start of a training run. Captures full config.
        Returns the new experiment id.
        """
        from dataclasses import asdict as dc_asdict
        cfg_dict = dc_asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)

        exp_type = "finetune" if cfg_dict.get("pretrained_model") else "scratch_train"
        name = cfg_dict.get("pretrained_model") or f"scratch_{cfg_dict.get('n_layers')}L_{cfg_dict.get('hidden_size')}H"

        exp = Experiment(
            id=self._next_id(),
            name=name,
            type=exp_type,
            status="running",
            config=cfg_dict,
            model_architecture={
                "n_layers": cfg_dict.get("n_layers"),
                "hidden_size": cfg_dict.get("hidden_size"),
                "n_heads": cfg_dict.get("n_heads"),
                "n_params": n_params,
                "pretrained_model": cfg_dict.get("pretrained_model", ""),
            },
            dataset={
                "tokenizer": cfg_dict.get("tokenizer"),
                "data_dir": cfg_dict.get("data_dir"),
                "seq_len": cfg_dict.get("seq_len"),
                "total_steps": cfg_dict.get("steps"),
                "batch_size": cfg_dict.get("batch_size"),
                "seed": cfg_dict.get("seed"),
            },
        )
        return self.register(exp)

    def complete_training(
        self,
        exp_id: str,
        final_loss: float,
        steps: int,
        samples_consumed: int,
        checkpoint_path: str,
        output_dir: str,
    ) -> None:
        """Called at the end of a successful training run."""
        self.update(
            exp_id,
            status="completed",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            final_loss=round(final_loss, 6),
            total_steps=steps,
            samples_consumed=samples_consumed,
            checkpoint_path=checkpoint_path,
            metrics_json=str(Path(output_dir) / "metrics.json"),
            training_plot=str(Path(output_dir) / "training_metrics.png"),
        )

    def add_uq_results(self, exp_id: str, summary: dict, results_dir: str) -> None:
        """Attach UQ evaluation results to an existing experiment."""
        self.update(
            exp_id,
            uq_results={
                "summary": summary,
                "results_json": str(Path(results_dir) / "results.json"),
                "reliability_diagram": str(Path(results_dir) / "reliability_diagram.png"),
            },
        )

    def add_note(self, exp_id: str, note: str) -> None:
        entry = self._find(exp_id)
        if entry is None:
            raise KeyError(f"Experiment {exp_id} not found")
        existing = entry.get("notes", "")
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry["notes"] = f"{existing}\n[{timestamp}] {note}".strip()
        self._save()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_all(self) -> list[dict]:
        return list(self._experiments)

    def get(self, exp_id: str) -> dict:
        entry = self._find(exp_id)
        if entry is None:
            raise KeyError(f"Experiment {exp_id} not found")
        return entry

    def print_table(self) -> None:
        if not self._experiments:
            print("No experiments registered yet.")
            return
        header = f"{'ID':<10} {'Name':<35} {'Type':<18} {'Status':<12} {'Loss':<8} {'Steps':<10} {'Created'}"
        print(header)
        print("-" * len(header))
        for e in self._experiments:
            loss = f"{e['final_loss']:.4f}" if e.get("final_loss") is not None else "-"
            steps = str(e.get("total_steps") or "-")
            print(
                f"{e['id']:<10} {e['name'][:34]:<35} {e['type']:<18} "
                f"{e['status']:<12} {loss:<8} {steps:<10} {e.get('created_at', '')[:19]}"
            )

    def print_detail(self, exp_id: str) -> None:
        entry = self.get(exp_id)
        print(json.dumps(entry, indent=2))
