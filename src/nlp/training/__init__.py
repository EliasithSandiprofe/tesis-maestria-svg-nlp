"""
Training Package — Fase D: Entrenamiento del modelo Multi-Head DistilBERT.

Módulos
-------
training_config  : Configuración centralizada del entrenamiento.
training_utils   : Utilidades de semilla, checkpoints, métricas e I/O.
trainer          : Clase ``MultiTaskTrainer`` con loop de entrenamiento.
train            : Script principal (entrypoint para Colab y local).
"""

from .training_config import TrainingConfig
from .trainer import MultiTaskTrainer
from .training_utils import (
    set_seed,
    create_directories,
    save_checkpoint,
    load_checkpoint,
    save_metrics_csv,
    save_config_json,
    move_batch_to_device,
    format_time,
    print_epoch_summary,
    print_training_header,
)

__all__ = [
    "TrainingConfig",
    "MultiTaskTrainer",
    "set_seed",
    "create_directories",
    "save_checkpoint",
    "load_checkpoint",
    "save_metrics_csv",
    "save_config_json",
    "move_batch_to_device",
    "format_time",
    "print_epoch_summary",
    "print_training_header",
]
