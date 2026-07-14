"""
train.py
--------
Script principal de entrenamiento — Fase D del proyecto de tesis SVG-NLP.

Orquesta el pipeline completo:
    1. Carga de configuración y fijación de semilla.
    2. Creación de directorios de salida.
    3. Carga del tokenizer y los encoders.
    4. Construcción de los Dataset y DataLoader.
    5. Instanciación del modelo MultiTaskDistilBERT.
    6. Entrenamiento con early stopping.
    7. Guardado de checkpoints, CSV de historial, JSON de configuración
       y resumen Markdown.

Uso
---
Desde la raíz del proyecto::

    python src/nlp/training/train.py

Desde Google Colab (ajustar PROJECT_ROOT según el mount)::

    %run src/nlp/training/train.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

# ── Ajuste del PYTHONPATH para ejecución directa ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.nlp.preprocessing.dataset_builder import SVGPromptDataset
from src.nlp.preprocessing.tokenizer import tokenize_prompts
from src.nlp.modeling.model_config import SVGModelConfig
from src.nlp.modeling.model_utils import build_model, print_model_summary
from src.nlp.training.training_config import TrainingConfig
from src.nlp.training.trainer import MultiTaskTrainer
from src.nlp.training.training_utils import (
    create_directories,
    print_training_header,
    save_config_json,
    save_metrics_csv,
    set_seed,
)


def main() -> None:
    """Ejecuta el pipeline de entrenamiento completo (Fase D)."""
    t_total_start = time.perf_counter()

    print()
    print("╔" + "═" * 60 + "╗")
    print("║   FASE D — ENTRENAMIENTO MultiTaskDistilBERT SVG-NLP    ║")
    print("╚" + "═" * 60 + "╝")

    # ── 1. Configuración y semilla ────────────────────────────────────────────
    config = TrainingConfig(project_root=_PROJECT_ROOT)
    set_seed(config.random_seed)
    print(f"\n  Configuración cargada:")
    print(f"  {config}")

    # ── 2. Directorios de salida ──────────────────────────────────────────────
    create_directories(config.checkpoint_dir, config.reports_dir)

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    print("\n  Cargando tokenizer desde disco...", end=" ", flush=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.tokenizer_dir))
    print("OK")

    # ── 4. Datasets ───────────────────────────────────────────────────────────
    print("  Cargando datasets...")
    train_df = pd.read_csv(config.train_csv, encoding="utf-8")
    val_df   = pd.read_csv(config.val_csv,   encoding="utf-8")
    print(f"    Train : {len(train_df):,} registros")
    print(f"    Val   : {len(val_df):,} registros")

    print("  Tokenizando...", end=" ", flush=True)
    train_tokens = tokenize_prompts(
        train_df["prompt"], tokenizer, max_length=config.max_length
    )
    val_tokens = tokenize_prompts(
        val_df["prompt"], tokenizer, max_length=config.max_length
    )
    print("OK")

    train_ds = SVGPromptDataset(
        input_ids       = train_tokens["input_ids"],
        attention_mask  = train_tokens["attention_mask"],
        labels_color    = train_df["color_enc"].values,
        labels_estilo   = train_df["estilo_enc"].values,
        labels_elemento = train_df["elemento_enc"].values,
        labels_posicion = train_df["posicion_enc"].values,
    )
    val_ds = SVGPromptDataset(
        input_ids       = val_tokens["input_ids"],
        attention_mask  = val_tokens["attention_mask"],
        labels_color    = val_df["color_enc"].values,
        labels_estilo   = val_df["estilo_enc"].values,
        labels_elemento = val_df["elemento_enc"].values,
        labels_posicion = val_df["posicion_enc"].values,
    )

    # ── 5. DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,     # descartar último batch incompleto en train
        num_workers=0,
        pin_memory=str(config.device) == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,    # evaluar sobre todas las muestras
        num_workers=0,
        pin_memory=str(config.device) == "cuda",
    )
    print(
        f"  DataLoaders: "
        f"train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches"
    )

    # ── 6. Modelo ─────────────────────────────────────────────────────────────
    print("\n  Construyendo modelo...", end=" ", flush=True)
    model_config = SVGModelConfig.from_project(_PROJECT_ROOT)
    model = build_model(model_config).to(config.device)
    print("OK")
    print_model_summary(model, model_config)

    # ── 7. Entrenamiento ──────────────────────────────────────────────────────
    print_training_header(config, n_train=len(train_ds), n_val=len(val_ds))

    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    result = trainer.fit()

    # ── 8. Persistencia de artefactos ─────────────────────────────────────────
    save_metrics_csv(result["history"], config.history_csv_path)
    save_config_json(config.to_dict(),  config.config_json_path)
    _write_training_summary(result, config, model_config, len(train_ds), len(val_ds))

    # ── 9. Resumen final ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total_start
    best_m     = result["best_metrics"] or {}

    print()
    print("╔" + "═" * 60 + "╗")
    print("║              ENTRENAMIENTO COMPLETADO                   ║")
    print("╚" + "═" * 60 + "╝")
    print(f"  Tiempo total          : {_fmt(total_time)}")
    print(f"  Épocas entrenadas     : {result['total_epochs_trained']} / {config.epochs}")
    print(f"  Mejor época           : {result['best_epoch']}")
    print(f"  Mejor val_loss        : {result['best_val_loss']:.4f}")
    if best_m:
        print(f"  Accuracy (mejor época):")
        print(f"    color     = {best_m.get('accuracy_color',    0)*100:.1f}%")
        print(f"    estilo    = {best_m.get('accuracy_estilo',   0)*100:.1f}%")
        print(f"    elemento  = {best_m.get('accuracy_elemento', 0)*100:.1f}%")
        print(f"    posicion  = {best_m.get('accuracy_posicion', 0)*100:.1f}%")
        print(f"    mean      = {best_m.get('mean_accuracy',     0)*100:.1f}%")
    print()
    print("  Artefactos guardados:")
    print(f"    {config.best_checkpoint_path.relative_to(_PROJECT_ROOT)}")
    print(f"    {config.last_checkpoint_path.relative_to(_PROJECT_ROOT)}")
    print(f"    {config.history_csv_path.relative_to(_PROJECT_ROOT)}")
    print(f"    {config.config_json_path.relative_to(_PROJECT_ROOT)}")
    print(f"    {config.summary_md_path.relative_to(_PROJECT_ROOT)}")
    print()


# ── Generación del resumen Markdown ──────────────────────────────────────────

def _write_training_summary(
    result: dict,
    config: TrainingConfig,
    model_config: SVGModelConfig,
    n_train: int,
    n_val: int,
) -> None:
    """Genera ``training_summary.md`` con los resultados del entrenamiento.

    Parameters
    ----------
    result : dict
        Resultado devuelto por ``MultiTaskTrainer.fit()``.
    config : TrainingConfig
        Configuración del entrenamiento.
    model_config : SVGModelConfig
        Configuración del modelo (para ``num_classes``).
    n_train : int
        Número de muestras de entrenamiento.
    n_val : int
        Número de muestras de validación.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_m    = result["best_metrics"] or {}
    n_trained = result["total_epochs_trained"]
    stopped   = n_trained < config.epochs

    lines: list[str] = [
        "# Resumen del Entrenamiento — Fase D",
        "",
        f"> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  ",
        f"> **Generado:** {timestamp}  ",
        f"> **Fase:** D — Entrenamiento del modelo  ",
        "",
        "---",
        "",
        "## 1. Configuración del Entrenamiento",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Dispositivo | `{config.device}` |",
        f"| Modelo base | `distilbert-base-uncased` |",
        f"| Épocas planificadas | {config.epochs} |",
        f"| Épocas entrenadas | {n_trained}"
        + (" (early stopping)" if stopped else "") + " |",
        f"| Batch size | {config.batch_size} |",
        f"| Learning rate | {config.learning_rate} |",
        f"| Weight decay | {config.weight_decay} |",
        f"| Gradient clipping | {config.max_grad_norm} |",
        f"| Patience (early stopping) | {config.patience} |",
        f"| Warmup ratio | {config.warmup_ratio * 100:.0f}% |",
        f"| Random seed | {config.random_seed} |",
        f"| Max length (tokens) | {config.max_length} |",
        "",
        "---",
        "",
        "## 2. Dataset",
        "",
        "| Conjunto | Muestras |",
        "|---|---|",
        f"| Train | {n_train:,} |",
        f"| Validation | {n_val:,} |",
        "",
        "### Clases por tarea",
        "",
        "| Tarea | Clases |",
        "|---|---|",
    ]
    for task, n_cls in model_config.num_classes.items():
        lines.append(f"| `{task}` | {n_cls} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Mejor Época",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Mejor época | {result['best_epoch']} |",
        f"| Mejor val_loss | {result['best_val_loss']:.6f} |",
        f"| Accuracy `color` | {best_m.get('accuracy_color', 0) * 100:.2f}% |",
        f"| Accuracy `estilo` | {best_m.get('accuracy_estilo', 0) * 100:.2f}% |",
        f"| Accuracy `elemento` | {best_m.get('accuracy_elemento', 0) * 100:.2f}% |",
        f"| Accuracy `posicion` | {best_m.get('accuracy_posicion', 0) * 100:.2f}% |",
        f"| Mean Accuracy | {best_m.get('mean_accuracy', 0) * 100:.2f}% |",
        "",
        "---",
        "",
        "## 4. Historial de Métricas",
        "",
        "| Época | Train Loss | Val Loss | Acc Color | Acc Estilo | Acc Elem | Acc Pos | Mean Acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["history"]:
        lines.append(
            f"| {row['epoch']} "
            f"| {row['train_loss']:.4f} "
            f"| {row['val_loss']:.4f} "
            f"| {row['accuracy_color']*100:.1f}% "
            f"| {row['accuracy_estilo']*100:.1f}% "
            f"| {row['accuracy_elemento']*100:.1f}% "
            f"| {row['accuracy_posicion']*100:.1f}% "
            f"| {row['mean_accuracy']*100:.1f}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Artefactos Generados",
        "",
        "| Artefacto | Ruta |",
        "|---|---|",
        f"| Mejor checkpoint | `{config.best_checkpoint_path.relative_to(config.project_root)}` |",
        f"| Último checkpoint | `{config.last_checkpoint_path.relative_to(config.project_root)}` |",
        f"| Historial CSV | `{config.history_csv_path.relative_to(config.project_root)}` |",
        f"| Configuración JSON | `{config.config_json_path.relative_to(config.project_root)}` |",
        "",
        "---",
        "",
        "> *Generado automáticamente por `train.py` — Fase D del proyecto de tesis.*",
    ]

    config.summary_md_path.write_text("\n".join(lines), encoding="utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """Formatea segundos a HH:MM:SS."""
    total = int(seconds)
    h, r  = divmod(total, 3600)
    m, s  = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
