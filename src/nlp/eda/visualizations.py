"""
visualizations.py
-----------------
Genera todas las gráficas del EDA usando exclusivamente matplotlib y las
guarda en ``dataset/reports/figures/``.

Gráficas producidas
-------------------
- Distribución de frecuencias: color, estilo, elemento, posicion.
- Histograma de longitud de prompts (caracteres y palabras).

Funciones públicas
------------------
generate_visualizations(df, prompt_stats, figures_dir) -> list[Path]
    Genera y guarda todas las figuras; devuelve la lista de rutas creadas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Backend no interactivo para evitar ventanas emergentes al ejecutar en terminal
matplotlib.use("Agg")

LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]

# Paleta de colores neutral (no depende de seaborn)
BAR_COLOR = "#4C72B0"
HIST_COLOR = "#55A868"


def generate_visualizations(
    df: pd.DataFrame,
    prompt_stats: dict,
    figures_dir: str | Path,
) -> list[Path]:
    """Genera y guarda todas las visualizaciones del EDA.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    prompt_stats : dict
        Diccionario devuelto por ``prompt_analysis.analyze_prompts``.
        Debe contener las claves ``char_series`` y ``word_series``.
    figures_dir : str | Path
        Directorio de salida para las imágenes PNG.

    Returns
    -------
    list[Path]
        Lista de rutas absolutas de los archivos guardados.
    """
    out_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    sep = "=" * 60
    print(sep)
    print("  GENERACIÓN DE VISUALIZACIONES")
    print(sep)

    # ── Distribuciones de etiquetas ──────────────────────────────────────────
    for col in LABEL_COLUMNS:
        path = _plot_label_distribution(df, col, out_path)
        saved_paths.append(path)
        print(f"  [OK] {path.name}")

    # ── Histograma longitud por caracteres ───────────────────────────────────
    path = _plot_prompt_length_histogram(
        prompt_stats["char_series"],
        metric_label="Longitud en caracteres",
        filename="histograma_longitud_caracteres.png",
        out_path=out_path,
        color=HIST_COLOR,
    )
    saved_paths.append(path)
    print(f"  [OK] {path.name}")

    # ── Histograma longitud por palabras ─────────────────────────────────────
    path = _plot_prompt_length_histogram(
        prompt_stats["word_series"],
        metric_label="Longitud en palabras",
        filename="histograma_longitud_palabras.png",
        out_path=out_path,
        color="#C44E52",
    )
    saved_paths.append(path)
    print(f"  [OK] {path.name}")

    print(f"\n  Total figuras guardadas: {len(saved_paths)}")
    print(f"  Directorio: {out_path.resolve()}")
    print(sep)
    print()

    return saved_paths


# ── Helpers privados ─────────────────────────────────────────────────────────

def _plot_label_distribution(
    df: pd.DataFrame,
    column: str,
    out_path: Path,
) -> Path:
    """Genera un gráfico de barras horizontales para una etiqueta.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    column : str
        Nombre de la columna de etiqueta.
    out_path : Path
        Directorio de salida.

    Returns
    -------
    Path
        Ruta del archivo PNG generado.
    """
    counts = df[column].value_counts().sort_values(ascending=True)
    n_classes = len(counts)

    fig_height = max(3, n_classes * 0.55)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    bars = ax.barh(counts.index.astype(str), counts.values, color=BAR_COLOR)

    # Etiquetas de valor al final de cada barra
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + counts.values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Frecuencia", fontsize=10)
    ax.set_title(f"Distribución de '{column}'  ({n_classes} clases)", fontsize=12, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, counts.values.max() * 1.15)

    fig.tight_layout()

    file_path = out_path / f"distribucion_{column}.png"
    fig.savefig(file_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return file_path


def _plot_prompt_length_histogram(
    series: pd.Series,
    metric_label: str,
    filename: str,
    out_path: Path,
    color: str = HIST_COLOR,
) -> Path:
    """Genera un histograma de la distribución de longitud de prompts.

    Parameters
    ----------
    series : pd.Series
        Serie con los valores de longitud (caracteres o palabras).
    metric_label : str
        Nombre del eje X para el histograma.
    filename : str
        Nombre del archivo de salida.
    out_path : Path
        Directorio de salida.
    color : str
        Color de las barras del histograma.

    Returns
    -------
    Path
        Ruta del archivo PNG generado.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(series.dropna(), bins=30, color=color, edgecolor="white", linewidth=0.6)

    mean_val = series.mean()
    median_val = series.median()

    ax.axvline(mean_val, color="#E07B39", linestyle="--", linewidth=1.5,
               label=f"Media: {mean_val:.1f}")
    ax.axvline(median_val, color="#8172B2", linestyle="-.", linewidth=1.5,
               label=f"Mediana: {median_val:.1f}")

    ax.set_xlabel(metric_label, fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title(f"Histograma — {metric_label}", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    file_path = out_path / filename
    fig.savefig(file_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return file_path
