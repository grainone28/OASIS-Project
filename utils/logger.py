import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(log_dir: str = "./logs", run_name: Optional[str] = None) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("vandal")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / f"{run_name}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class MetricLogger:

    def __init__(self, save_dir: str, run_name: str, use_wandb: bool = False, wandb_project: str = "vandal"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{run_name}_metrics.json"
        self.history: list[Dict[str, Any]] = []
        self.use_wandb = use_wandb

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=run_name)
                self._wandb = wandb
            except ImportError:
                print("[MetricLogger] wandb not installed. Disabling W&B logging.")
                self.use_wandb = False

    def log(self, epoch: int, metrics: Dict[str, float]):
        row = {"epoch": epoch, **metrics}
        self.history.append(row)

        # Persist
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

        if self.use_wandb:
            self._wandb.log(metrics, step=epoch)

    def finish(self):
        if self.use_wandb:
            self._wandb.finish()
