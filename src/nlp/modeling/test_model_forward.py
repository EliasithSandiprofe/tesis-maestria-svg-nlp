"""
test_model_forward.py
---------------------
Validación de la arquitectura MultiTaskDistilBERT mediante un forward pass
completo con un batch real extraído del pipeline de preprocesamiento.

OBJETIVO DE ESTA FASE
---------------------
Confirmar que:
1. El modelo se instancia correctamente a partir de los encoders y pesos
   preentrenados de DistilBERT.
2. Las cuatro cabezas producen logits con las dimensiones esperadas.
3. La pérdida total y las pérdidas individuales son escalares válidos.
4. El código queda listo para ser ejecutado en Google Colab (Fase D).

RESTRICCIONES (Fase C)
----------------------
- No se realiza entrenamiento (sin ``optimizer.step()`` ni ``backward()``).
- No se calculan métricas de evaluación.
- No se guardan pesos entrenados.

Uso::

    python src/nlp/modeling/test_model_forward.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

# ── Ajuste del PYTHONPATH ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.nlp.preprocessing.dataset_builder import SVGPromptDataset
from src.nlp.preprocessing.tokenizer import tokenize_prompts
from src.nlp.modeling.model_config import SVGModelConfig
from src.nlp.modeling.multitask_distilbert import MultiTaskDistilBERT
from src.nlp.modeling.model_utils import count_parameters, print_model_summary

# ── Rutas ─────────────────────────────────────────────────────────────────────
_TOKENIZER_DIR = _PROJECT_ROOT / "modelos" / "tokenizer"
_ENCODERS_DIR  = _PROJECT_ROOT / "modelos" / "label_encoders"
_TRAIN_CSV     = _PROJECT_ROOT / "dataset" / "processed" / "dataset_train.csv"

# ── Hiperparámetros del test ──────────────────────────────────────────────────
BATCH_SIZE = 4
MAX_LENGTH = 128


def run_forward_test() -> None:
    """Ejecuta el test completo del forward pass y muestra los resultados."""
    t0 = time.perf_counter()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  FASE C — VALIDACIÓN DEL FORWARD PASS                   ║")
    print("║  MultiTaskDistilBERT — SVG-NLP                          ║")
    print("╚" + "═" * 58 + "╝")

    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo      : {str(device).upper()}")

    # ── 1. Configuración del modelo ───────────────────────────────────────────
    _heading("1/5", "Cargando configuración del modelo")
    config = SVGModelConfig.from_project(_PROJECT_ROOT)
    print()
    print(config)

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    _heading("2/5", "Cargando tokenizer desde disco")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(_TOKENIZER_DIR))
    print(f"  Tokenizer cargado — vocabulario: {tokenizer.vocab_size:,} tokens")

    # ── 3. Dataset y DataLoader ───────────────────────────────────────────────
    _heading("3/5", "Construyendo Dataset PyTorch")
    train_df = pd.read_csv(_TRAIN_CSV, encoding="utf-8")
    print(f"  dataset_train.csv : {len(train_df):,} registros")

    print("  Tokenizando prompts...", end=" ", flush=True)
    tokens = tokenize_prompts(train_df["prompt"], tokenizer, max_length=MAX_LENGTH)
    print("OK")

    train_ds = SVGPromptDataset(
        input_ids       = tokens["input_ids"],
        attention_mask  = tokens["attention_mask"],
        labels_color    = train_df["color_enc"].values,
        labels_estilo   = train_df["estilo_enc"].values,
        labels_elemento = train_df["elemento_enc"].values,
        labels_posicion = train_df["posicion_enc"].values,
    )
    print(f"  SVGPromptDataset  : {len(train_ds):,} muestras")

    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  DataLoader        : batch_size={BATCH_SIZE}")

    # ── 4. Modelo ─────────────────────────────────────────────────────────────
    _heading("4/5", "Instanciando MultiTaskDistilBERT")
    print("  Cargando pesos de DistilBERT (caché local)...", end=" ", flush=True)
    model = MultiTaskDistilBERT(config).to(device)
    model.train()   # modo entrenamiento: activa dropout (representativo del flujo real)
    print("OK")

    # ── 5. Forward pass ───────────────────────────────────────────────────────
    _heading("5/5", "Ejecutando Forward Pass")

    batch = next(iter(dataloader))

    input_ids       = batch["input_ids"].to(device)
    attention_mask  = batch["attention_mask"].to(device)
    labels_color    = batch["labels_color"].to(device)
    labels_estilo   = batch["labels_estilo"].to(device)
    labels_elemento = batch["labels_elemento"].to(device)
    labels_posicion = batch["labels_posicion"].to(device)

    print(f"  Batch obtenido: {input_ids.shape[0]} muestras × {input_ids.shape[1]} tokens")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels_color=labels_color,
        labels_estilo=labels_estilo,
        labels_elemento=labels_elemento,
        labels_posicion=labels_posicion,
    )

    # ── Verificación de dimensiones ───────────────────────────────────────────
    sep = "=" * 60
    print()
    print(sep)
    print("  RESULTADOS DEL FORWARD PASS")
    print(sep)

    print(f"\n  Dispositivo          : {str(device).upper()}")
    print(f"  Tamaño del batch     : {BATCH_SIZE}")
    print()

    print("  Clases por tarea:")
    for task, n in config.num_classes.items():
        print(f"    • {task:<12} {n} clases")

    expected_shapes: dict[str, tuple[int, int]] = {
        "logits_color":    (BATCH_SIZE, config.num_classes["color"]),
        "logits_estilo":   (BATCH_SIZE, config.num_classes["estilo"]),
        "logits_elemento": (BATCH_SIZE, config.num_classes["elemento"]),
        "logits_posicion": (BATCH_SIZE, config.num_classes["posicion"]),
    }

    all_shapes_ok = True
    print()
    print(f"  {'Salida':<22} {'Forma real':<18} {'Forma esperada':<18} {'Estado'}")
    print(f"  {'-' * 64}")
    for key, exp in expected_shapes.items():
        actual = tuple(outputs[key].shape)
        ok = actual == exp
        if not ok:
            all_shapes_ok = False
        status = "[OK]" if ok else "[ERROR]"
        print(f"  {key:<22} {str(actual):<18} {str(exp):<18} {status}")

    # ── Pérdidas ──────────────────────────────────────────────────────────────
    print()
    print(f"  {'Pérdida':<22} {'Valor':>12} {'Escalar':>10}")
    print(f"  {'-' * 48}")
    all_losses_ok = True
    for loss_key in ["loss_color", "loss_estilo", "loss_elemento", "loss_posicion", "loss"]:
        val = outputs[loss_key]
        is_scalar = val.ndim == 0
        if not is_scalar:
            all_losses_ok = False
        prefix = "► " if loss_key == "loss" else "  "
        scalar_str = "[OK]" if is_scalar else "[ERROR]"
        print(f"  {prefix}{loss_key:<20} {val.item():>12.6f} {scalar_str:>10}")

    # ── Resumen del modelo ────────────────────────────────────────────────────
    print()
    print_model_summary(model, config)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    if all_shapes_ok and all_losses_ok:
        verdict = "✓  FASE C COMPLETADA — Arquitectura validada."
    else:
        verdict = "✗  ERRORES DETECTADOS — Revisar dimensiones."

    print("╔" + "═" * 58 + "╗")
    print(f"║  {verdict:<57}║")
    print(f"║  Tiempo de validación: {elapsed:.2f} s{' ' * (33 - len(f'{elapsed:.2f}'))}║")
    print("╚" + "═" * 58 + "╝")
    print()


# ── Helper ────────────────────────────────────────────────────────────────────

def _heading(step: str, description: str) -> None:
    print()
    print(f"  ── [{step}] {description}")


if __name__ == "__main__":
    run_forward_test()
