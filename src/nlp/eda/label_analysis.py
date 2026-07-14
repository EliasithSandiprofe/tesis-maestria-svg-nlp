"""
label_analysis.py
-----------------
Calcula la distribución de frecuencias de cada columna de etiqueta
(color, estilo, elemento, posicion) y exporta las tablas a CSV.

Funciones públicas
------------------
analyze_labels(df, output_dir) -> dict[str, pd.DataFrame]
    Ejecuta el análisis para cada etiqueta y guarda los resultados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]


def analyze_labels(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Calcula la distribución de frecuencias de cada etiqueta SVG.

    Para cada columna de etiqueta se genera una tabla con:
    - Frecuencia absoluta de cada clase.
    - Porcentaje sobre el total de registros.
    - Número de clases distintas.

    Los resultados se guardan como archivos CSV en ``output_dir``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    output_dir : str | Path
        Directorio donde se guardarán los CSV de distribución.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario ``{nombre_etiqueta: tabla_distribución}``.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    distributions: dict[str, pd.DataFrame] = {}

    sep = "=" * 60
    print(sep)
    print("  DISTRIBUCIÓN DE ETIQUETAS")
    print(sep)

    for col in LABEL_COLUMNS:
        dist = _compute_distribution(df, col)
        distributions[col] = dist

        # Guardar CSV
        csv_file = out_path / f"distribucion_{col}.csv"
        dist.to_csv(csv_file, index=False, encoding="utf-8")

        # Imprimir en consola
        n_classes = len(dist)
        print(f"\n  [{col.upper()}]  —  {n_classes} clases")
        print(dist.to_string(index=False))
        print(f"  Guardado: {csv_file}")

    print()
    print(sep)
    print()

    return distributions


# ── Helpers privados ─────────────────────────────────────────────────────────

def _compute_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Devuelve la tabla de distribución de frecuencias de una columna.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    column : str
        Nombre de la columna a analizar.

    Returns
    -------
    pd.DataFrame
        Tabla con columnas: ``valor``, ``frecuencia``, ``porcentaje_%``.
        Ordenada de mayor a menor frecuencia.
    """
    counts = df[column].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)

    dist = pd.DataFrame({
        "valor": counts.index,
        "frecuencia": counts.values,
        "porcentaje_%": pct.values,
    })

    # Reemplazar NaN en la columna valor por la cadena "<nulo>"
    dist["valor"] = dist["valor"].fillna("<nulo>")

    return dist.reset_index(drop=True)
