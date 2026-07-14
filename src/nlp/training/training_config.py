"""
training_config.py
------------------
Configuración centralizada del pipeline de entrenamiento (Fase D).

Todos los hiperparámetros, rutas y parámetros del dispositivo se definen
en un único objeto ``TrainingConfig`` para garantizar trazabilidad completa
y reproducibilidad del experimento de tesis.

Clases
------
TrainingConfig
    Configuración del entrenamiento.  Resolución automática de rutas a
    partir de la raíz del proyecto y detección automática del dispositivo
    (CUDA si está disponible).
"""

from __future__ import annotations

from pathlib import Path

import torch


class TrainingConfig:
    """Configuración del entrenamiento Multi-Head DistilBERT SVG-NLP.

    Attributes
    ----------
    batch_size : int
        Tamaño del mini-batch (por defecto 16).
    epochs : int
        Número máximo de épocas de entrenamiento (por defecto 5).
    learning_rate : float
        Tasa de aprendizaje para AdamW (por defecto 2e-5).
    weight_decay : float
        Regularización L2 para AdamW (por defecto 0.01).
    max_grad_norm : float
        Umbral de gradient clipping (por defecto 1.0).
    patience : int
        Épocas sin mejora antes del early stopping (por defecto 2).
    random_seed : int
        Semilla para reproducibilidad (por defecto 42).
    warmup_ratio : float
        Fracción de pasos totales usados como warmup lineal (por defecto 0.1).
    max_length : int
        Longitud máxima de tokenización (por defecto 128, consistente con Fase B).
    device : torch.device
        Dispositivo de cómputo (CUDA si disponible, si no CPU).
    train_csv : Path
        Ruta al CSV de entrenamiento.
    val_csv : Path
        Ruta al CSV de validación.
    tokenizer_dir : Path
        Directorio del tokenizer serializado.
    encoders_dir : Path
        Directorio de los LabelEncoder serializados.
    checkpoint_dir : Path
        Directorio de salida para los checkpoints del modelo.
    reports_dir : Path
        Directorio de salida para los reportes de entrenamiento.
    best_checkpoint_path : Path
        Ruta exacta de ``best_model.pt``.
    last_checkpoint_path : Path
        Ruta exacta de ``last_model.pt``.
    history_csv_path : Path
        Ruta del CSV con el historial de métricas por época.
    config_json_path : Path
        Ruta del JSON con la configuración usada.
    summary_md_path : Path
        Ruta del Markdown con el resumen del entrenamiento.
    """

    def __init__(
        self,
        project_root: str | Path,
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        patience: int = 2,
        random_seed: int = 42,
        warmup_ratio: float = 0.1,
        max_length: int = 128,
    ) -> None:
        self.project_root = Path(project_root)

        # ── Hiperparámetros ───────────────────────────────────────────────────
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience      = patience
        self.random_seed   = random_seed
        self.warmup_ratio  = warmup_ratio
        self.max_length    = max_length

        # ── Dispositivo (detección automática) ────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Rutas de entrada ──────────────────────────────────────────────────
        _processed = self.project_root / "dataset" / "processed"
        self.train_csv     = _processed / "dataset_train.csv"
        self.val_csv       = _processed / "dataset_validation.csv"
        self.tokenizer_dir = self.project_root / "modelos" / "tokenizer"
        self.encoders_dir  = self.project_root / "modelos" / "label_encoders"

        # ── Rutas de salida ───────────────────────────────────────────────────
        self.checkpoint_dir = self.project_root / "modelos" / "checkpoints"
        self.reports_dir    = self.project_root / "dataset" / "reports" / "training"

        # Checkpoints
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last_model.pt"

        # Reportes
        self.history_csv_path = self.reports_dir / "training_history.csv"
        self.config_json_path = self.reports_dir / "training_config.json"
        self.summary_md_path  = self.reports_dir / "training_summary.md"

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convierte la configuración a un diccionario JSON-serializable.

        Todos los objetos ``Path`` y ``torch.device`` se convierten a cadenas.

        Returns
        -------
        dict
            Representación plana y serializable de la configuración.
        """
        return {
            "batch_size":    self.batch_size,
            "epochs":        self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay":  self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "patience":      self.patience,
            "random_seed":   self.random_seed,
            "warmup_ratio":  self.warmup_ratio,
            "max_length":    self.max_length,
            "device":        str(self.device),
            "train_csv":         str(self.train_csv),
            "val_csv":           str(self.val_csv),
            "tokenizer_dir":     str(self.tokenizer_dir),
            "encoders_dir":      str(self.encoders_dir),
            "checkpoint_dir":    str(self.checkpoint_dir),
            "reports_dir":       str(self.reports_dir),
            "best_checkpoint":   str(self.best_checkpoint_path),
            "last_checkpoint":   str(self.last_checkpoint_path),
        }

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  device={self.device},  batch_size={self.batch_size},\n"
            f"  epochs={self.epochs},  lr={self.learning_rate},  patience={self.patience},\n"
            f"  weight_decay={self.weight_decay},  max_grad_norm={self.max_grad_norm},\n"
            f"  warmup_ratio={self.warmup_ratio},  random_seed={self.random_seed}\n"
            f")"
        )
