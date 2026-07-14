"""
evaluate.py
-----------
Script principal de evaluación final — Fase E del proyecto de tesis SVG-NLP.

Carga el mejor checkpoint entrenado (``best_model.pt``) y evalúa el modelo
**exclusivamente** sobre ``dataset_test.csv``.

NO usa ``dataset_train.csv`` ni ``dataset_validation.csv``.
NO realiza ningún paso de entrenamiento.

Artefactos generados en ``dataset/reports/evaluation/``:
    - evaluation_metrics.json
    - evaluation_report.md
    - classification_report_<tarea>.md  (×4)
    - confusion_matrix_<tarea>.png       (×4)

Uso
---
Desde la raíz del proyecto::

    python src/nlp/evaluation/evaluate.py

Desde Google Colab::

    %run src/nlp/evaluation/evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
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
from src.nlp.evaluation.evaluator import MultiTaskEvaluator
from src.nlp.evaluation.evaluation_utils import (
    generate_evaluation_report,
    load_label_encoders_for_eval,
    load_model_from_checkpoint,
    plot_confusion_matrix,
    save_classification_report_md,
    save_metrics_json,
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_TOKENIZER_DIR   = _PROJECT_ROOT / "modelos" / "tokenizer"
_ENCODERS_DIR    = _PROJECT_ROOT / "modelos" / "label_encoders"
_CHECKPOINT_PATH = _PROJECT_ROOT / "modelos" / "checkpoints" / "best_model.pt"
_TEST_CSV        = _PROJECT_ROOT / "dataset" / "processed" / "dataset_test.csv"
_OUTPUT_DIR      = _PROJECT_ROOT / "dataset" / "reports" / "evaluation"

# Parámetros de evaluación
_BATCH_SIZE = 16
_MAX_LENGTH = 128
_TASKS      = ["color", "estilo", "elemento", "posicion"]


def main() -> None:
    """Ejecuta el pipeline completo de evaluación final (Fase E)."""
    t0 = time.perf_counter()

    print()
    print("╔" + "═" * 60 + "╗")
    print("║   FASE E — EVALUACIÓN FINAL MultiTaskDistilBERT SVG-NLP ║")
    print("╚" + "═" * 60 + "╝")

    # ── Preparación ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo : {str(device).upper()}")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Tokenizer y encoders ───────────────────────────────────────────────
    _step("1/6", "Cargando tokenizer y LabelEncoders")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(_TOKENIZER_DIR))
    print(f"  Tokenizer  : vocabulario {tokenizer.vocab_size:,} tokens")

    label_encoders = load_label_encoders_for_eval(_ENCODERS_DIR)
    for task, le in label_encoders.items():
        print(f"  Encoder    : {task:<12} ({len(le.classes_)} clases)")

    # ── 2. Dataset test ───────────────────────────────────────────────────────
    _step("2/6", "Cargando dataset_test.csv")
    test_df = pd.read_csv(_TEST_CSV, encoding="utf-8")
    print(f"  Registros test : {len(test_df):,}")

    # ── 3. SVGPromptDataset y DataLoader ─────────────────────────────────────
    _step("3/6", "Construyendo Dataset y DataLoader")
    print("  Tokenizando prompts...", end=" ", flush=True)
    test_tokens = tokenize_prompts(
        test_df["prompt"], tokenizer, max_length=_MAX_LENGTH
    )
    print("OK")

    test_ds = SVGPromptDataset(
        input_ids       = test_tokens["input_ids"],
        attention_mask  = test_tokens["attention_mask"],
        labels_color    = test_df["color_enc"].values,
        labels_estilo   = test_df["estilo_enc"].values,
        labels_elemento = test_df["elemento_enc"].values,
        labels_posicion = test_df["posicion_enc"].values,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"  Dataset    : {len(test_ds):,} muestras  |  {len(test_loader)} batches")

    # ── 4. Cargar modelo ──────────────────────────────────────────────────────
    _step("4/6", "Cargando modelo desde best_model.pt")
    model, ckpt_info = load_model_from_checkpoint(
        checkpoint_path=_CHECKPOINT_PATH,
        project_root=_PROJECT_ROOT,
        device=device,
    )

    # ── 5. Evaluación ─────────────────────────────────────────────────────────
    _step("5/6", "Ejecutando evaluación sobre el conjunto test")
    evaluator = MultiTaskEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        label_encoders=label_encoders,
    )
    results = evaluator.run()

    # ── 6. Generación de artefactos ───────────────────────────────────────────
    _step("6/6", "Guardando artefactos")

    # 6a. Métricas JSON
    json_path = _OUTPUT_DIR / "evaluation_metrics.json"
    save_metrics_json(
        results,
        path=json_path,
        checkpoint_path=_CHECKPOINT_PATH,
        test_csv_path=_TEST_CSV,
    )
    print(f"  [OK] {json_path.name}")

    # 6b. Classification reports (Markdown) + matrices de confusión (PNG)
    for task in _TASKS:
        # Classification report
        cr_path = _OUTPUT_DIR / f"classification_report_{task}.md"
        save_classification_report_md(
            results[task]["classification_report"], task, cr_path
        )
        print(f"  [OK] {cr_path.name}")

        # Matriz de confusión
        cm_path = _OUTPUT_DIR / f"confusion_matrix_{task}.png"
        plot_confusion_matrix(
            cm=np.array(results[task]["confusion_matrix"]),
            class_names=results[task]["class_names"],
            task=task,
            path=cm_path,
        )
        print(f"  [OK] {cm_path.name}")

    # 6c. Reporte Markdown final
    report_path = _OUTPUT_DIR / "evaluation_report.md"
    generate_evaluation_report(
        results=results,
        output_path=report_path,
        checkpoint_path=_CHECKPOINT_PATH,
        test_csv_path=_TEST_CSV,
        checkpoint_info=ckpt_info,
        figures_rel_dir=".",
    )
    print(f"  [OK] {report_path.name}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    glob    = results["global"]

    print()
    print("╔" + "═" * 60 + "╗")
    print("║              EVALUACIÓN COMPLETADA                      ║")
    print("╚" + "═" * 60 + "╝")
    print(f"  Tiempo total    : {_fmt(elapsed)}")
    print(f"  Mean Accuracy   : {glob['mean_accuracy']*100:.2f}%")
    print(f"  Mean F1 Macro   : {glob['mean_f1_macro']*100:.2f}%")
    print()
    print(f"  {'Tarea':<14} {'Accuracy':>9} {'F1 Macro':>9}")
    print(f"  {'─' * 35}")
    for task in _TASKS:
        m = results[task]
        print(f"  {task:<14} {m['accuracy']*100:>8.1f}%  {m['f1_macro']*100:>8.1f}%")
    print()
    print(f"  Artefactos en: {_OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")
    files = list(_OUTPUT_DIR.glob("*"))
    for f in sorted(files):
        print(f"    {f.name}")
    print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step(number: str, description: str) -> None:
    print()
    print(f"  ── [{number}] {description}")


def _fmt(seconds: float) -> str:
    total = int(seconds)
    h, r  = divmod(total, 3600)
    m, s  = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
