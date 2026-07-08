"""
run_eda.py
----------
Punto de entrada del Análisis Exploratorio de Datos (EDA) — Fase A.

Ejecuta el flujo completo en orden:
    1. Carga del dataset
    2. Análisis de calidad
    3. Análisis de etiquetas
    4. Análisis de prompts
    5. Generación de visualizaciones
    6. Generación del reporte Markdown

Uso desde la raíz del proyecto:
    python -m src.nlp.eda.run_eda
    # o directamente:
    python src/nlp/eda/run_eda.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Ajuste del PYTHONPATH para ejecución directa ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.nlp.eda.dataset_loader import load_dataset
from src.nlp.eda.quality_analysis import analyze_quality
from src.nlp.eda.label_analysis import analyze_labels
from src.nlp.eda.prompt_analysis import analyze_prompts
from src.nlp.eda.visualizations import generate_visualizations
from src.nlp.eda.report_generator import generate_report

# ── Rutas del proyecto ────────────────────────────────────────────────────────
DATASET_PATH  = _PROJECT_ROOT / "dataset" / "processed" / "dataset_training.csv"
REPORTS_DIR   = _PROJECT_ROOT / "dataset" / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
LABELS_DIR    = REPORTS_DIR / "label_distributions"
REPORT_FILE   = REPORTS_DIR / "eda_report.md"


def _step(number: int, description: str) -> None:
    """Imprime el encabezado de un paso del flujo EDA."""
    print()
    print(f"{'━' * 60}")
    print(f"  PASO {number}/6 — {description}")
    print(f"{'━' * 60}")


def run() -> None:
    """Ejecuta el flujo completo del EDA y mide el tiempo total."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║       EDA — FASE A: ANÁLISIS EXPLORATORIO DE DATOS       ║")
    print("║       Proyecto: Tesis Maestría SVG-NLP                   ║")
    print("╚" + "═" * 58 + "╝")

    t0 = time.perf_counter()

    # ── Paso 1: Cargar dataset ───────────────────────────────────────────────
    _step(1, "Carga del Dataset")
    df = load_dataset(DATASET_PATH)

    # ── Paso 2: Análisis de calidad ──────────────────────────────────────────
    _step(2, "Análisis de Calidad del Dataset")
    quality_results = analyze_quality(df)

    # ── Paso 3: Análisis de etiquetas ────────────────────────────────────────
    _step(3, "Distribución de Etiquetas")
    label_distributions = analyze_labels(df, output_dir=LABELS_DIR)

    # ── Paso 4: Análisis de prompts ──────────────────────────────────────────
    _step(4, "Análisis de Prompts")
    prompt_stats = analyze_prompts(df)

    # ── Paso 5: Visualizaciones ──────────────────────────────────────────────
    _step(5, "Generación de Visualizaciones")
    saved_figures = generate_visualizations(df, prompt_stats, figures_dir=FIGURES_DIR)

    # ── Paso 6: Reporte Markdown ─────────────────────────────────────────────
    _step(6, "Generación del Reporte Automático")
    report_path = generate_report(
        df=df,
        quality_results=quality_results,
        label_distributions=label_distributions,
        prompt_stats=prompt_stats,
        figures_dir=FIGURES_DIR,
        output_path=REPORT_FILE,
    )

    # ── Resumen final ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print("╔" + "═" * 58 + "╗")
    print("║                  EDA COMPLETADO                         ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Tiempo total  : {elapsed:.2f} s")
    print(f"  Figuras       : {len(saved_figures)} archivos en {FIGURES_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"  Etiquetas CSV : {LABELS_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"  Reporte       : {report_path.relative_to(_PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    run()
