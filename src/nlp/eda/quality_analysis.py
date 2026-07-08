"""
quality_analysis.py
-------------------
Analiza la calidad del dataset: valores nulos, duplicados, cardinalidad
de columnas y consistencia general.

Funciones públicas
------------------
analyze_quality(df) -> dict
    Ejecuta todos los análisis de calidad y devuelve un diccionario con
    los resultados estructurados.
"""

from __future__ import annotations

import pandas as pd


# Columnas de etiquetas del problema de clasificación multietiqueta
LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]


def analyze_quality(df: pd.DataFrame) -> dict:
    """Realiza un análisis completo de calidad del dataset.

    Comprende:
    - Conteo de valores nulos por columna.
    - Detección de registros duplicados (completos y solo por prompt).
    - Cardinalidad (valores únicos) por columna.
    - Consistencia: valores vacíos o solo espacios en blanco.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cargado por ``dataset_loader.load_dataset``.

    Returns
    -------
    dict
        Diccionario con claves:
        ``nulls``, ``duplicates``, ``unique_values``, ``blank_values``,
        ``summary_table``.
    """
    results: dict = {}

    results["nulls"] = _null_analysis(df)
    results["duplicates"] = _duplicate_analysis(df)
    results["unique_values"] = _unique_values(df)
    results["blank_values"] = _blank_values(df)
    results["summary_table"] = _build_summary_table(df, results)

    _print_quality_report(results)

    return results


# ── Helpers privados ─────────────────────────────────────────────────────────

def _null_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una tabla con el conteo y porcentaje de nulos por columna."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return pd.DataFrame({
        "columna": null_counts.index,
        "nulos": null_counts.values,
        "porcentaje_%": null_pct.values,
    })


def _duplicate_analysis(df: pd.DataFrame) -> dict:
    """Detecta filas completamente duplicadas y duplicados en la columna prompt."""
    full_dupes = int(df.duplicated().sum())
    prompt_dupes = int(df.duplicated(subset=["prompt"]).sum())
    return {
        "filas_duplicadas_completas": full_dupes,
        "prompts_duplicados": prompt_dupes,
        "prompts_unicos": int(df["prompt"].nunique()),
    }


def _unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la cardinalidad de cada columna."""
    data = [
        {"columna": col, "valores_unicos": int(df[col].nunique())}
        for col in df.columns
    ]
    return pd.DataFrame(data)


def _blank_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta celdas que contienen solo espacios en blanco (str vacíos)."""
    data = []
    for col in df.select_dtypes(include="object").columns:
        blank = int(df[col].str.strip().eq("").sum())
        data.append({"columna": col, "valores_en_blanco": blank})
    return pd.DataFrame(data)


def _build_summary_table(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Construye una tabla resumen unificada de calidad por columna."""
    nulls_df = results["nulls"].set_index("columna")
    unique_df = results["unique_values"].set_index("columna")

    rows = []
    for col in df.columns:
        rows.append({
            "columna": col,
            "tipo": str(df[col].dtype),
            "nulos": int(nulls_df.loc[col, "nulos"]),
            "nulos_%": float(nulls_df.loc[col, "porcentaje_%"]),
            "valores_unicos": int(unique_df.loc[col, "valores_unicos"]),
        })
    return pd.DataFrame(rows)


def _print_quality_report(results: dict) -> None:
    """Imprime el reporte de calidad en consola."""
    sep = "=" * 60
    print(sep)
    print("  ANÁLISIS DE CALIDAD DEL DATASET")
    print(sep)

    # Nulos
    print("\n  Valores nulos por columna:")
    print(results["nulls"].to_string(index=False))

    # Duplicados
    dup = results["duplicates"]
    print(f"\n  Filas completamente duplicadas : {dup['filas_duplicadas_completas']}")
    print(f"  Prompts duplicados             : {dup['prompts_duplicados']}")
    print(f"  Prompts únicos                 : {dup['prompts_unicos']}")

    # Cardinalidad
    print("\n  Cardinalidad por columna (valores únicos):")
    print(results["unique_values"].to_string(index=False))

    # Blancos
    print("\n  Valores en blanco (solo espacios) por columna:")
    print(results["blank_values"].to_string(index=False))

    # Tabla resumen
    print("\n  Tabla resumen de calidad:")
    print(results["summary_table"].to_string(index=False))
    print(sep)
    print()
