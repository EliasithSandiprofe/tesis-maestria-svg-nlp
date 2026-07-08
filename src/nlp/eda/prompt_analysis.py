"""
prompt_analysis.py
------------------
Calcula métricas estadísticas sobre la longitud de los prompts del dataset:
longitud en caracteres y en palabras.

Funciones públicas
------------------
analyze_prompts(df) -> dict
    Genera las métricas estadísticas completas de los prompts.
"""

from __future__ import annotations

import pandas as pd


def analyze_prompts(df: pd.DataFrame) -> dict:
    """Analiza las características de longitud de la columna ``prompt``.

    Métricas calculadas (para caracteres y palabras):
    - Mínimo
    - Máximo
    - Promedio (media)
    - Mediana
    - Desviación estándar

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo con la columna ``prompt``.

    Returns
    -------
    dict
        Diccionario con claves:
        ``char_lengths``, ``word_lengths``, ``stats_table``,
        ``char_series``, ``word_series``.

        - ``char_lengths`` y ``word_lengths`` son Series de pandas con los
          valores por fila.
        - ``stats_table`` es un DataFrame resumen con ambas métricas.
        - ``char_series`` / ``word_series`` son alias de las Series anteriores
          para uso en visualizaciones.
    """
    char_lengths: pd.Series = df["prompt"].str.len()
    word_lengths: pd.Series = df["prompt"].str.split().str.len()

    stats_table = _build_stats_table(char_lengths, word_lengths)

    _print_prompt_report(stats_table)

    return {
        "char_lengths": char_lengths,
        "word_lengths": word_lengths,
        "char_series": char_lengths,
        "word_series": word_lengths,
        "stats_table": stats_table,
    }


# ── Helpers privados ─────────────────────────────────────────────────────────

def _compute_stats(series: pd.Series) -> dict:
    """Calcula las cinco métricas básicas de una serie numérica."""
    return {
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "promedio": round(float(series.mean()), 2),
        "mediana": round(float(series.median()), 2),
        "desv_std": round(float(series.std()), 2),
    }


def _build_stats_table(
    char_lengths: pd.Series,
    word_lengths: pd.Series,
) -> pd.DataFrame:
    """Construye un DataFrame comparativo de estadísticas de longitud.

    Parameters
    ----------
    char_lengths : pd.Series
        Longitud en caracteres de cada prompt.
    word_lengths : pd.Series
        Longitud en palabras de cada prompt.

    Returns
    -------
    pd.DataFrame
        Tabla con métricas para caracteres y palabras.
    """
    char_stats = _compute_stats(char_lengths)
    word_stats = _compute_stats(word_lengths)

    metrics = ["min", "max", "promedio", "mediana", "desv_std"]
    return pd.DataFrame({
        "metrica": metrics,
        "longitud_caracteres": [char_stats[m] for m in metrics],
        "longitud_palabras": [word_stats[m] for m in metrics],
    })


def _print_prompt_report(stats_table: pd.DataFrame) -> None:
    """Imprime el reporte de análisis de prompts en consola."""
    sep = "=" * 60
    print(sep)
    print("  ANÁLISIS DE PROMPTS")
    print(sep)
    print()
    print("  Estadísticas de longitud de los prompts:")
    print(stats_table.to_string(index=False))
    print()
    print(sep)
    print()
