"""
dataset_loader.py
-----------------
Responsable de leer y validar el dataset SVG-NLP desde el CSV procesado.

Funciones públicas
------------------
load_dataset(path) -> pd.DataFrame
    Carga el CSV, valida su existencia y estructura, e imprime un resumen
    informativo del contenido.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Columnas que deben estar presentes en el dataset
REQUIRED_COLUMNS: list[str] = ["prompt", "color", "estilo", "elemento", "posicion"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga el dataset de entrenamiento SVG-NLP desde un archivo CSV.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si alguna columna requerida está ausente.
    """
    csv_path = Path(path)

    # ── Validación de existencia ─────────────────────────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {csv_path.resolve()}"
        )

    # ── Carga ────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, encoding="utf-8")

    # ── Validación de columnas ───────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"El dataset no contiene las columnas requeridas: {missing_cols}"
        )

    # ── Resumen en consola ───────────────────────────────────────────────────
    _print_summary(df, csv_path)

    return df


# ── Helpers privados ─────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, path: Path) -> None:
    """Imprime un resumen informativo del dataset cargado."""
    mem_kb = df.memory_usage(deep=True).sum() / 1024

    sep = "=" * 60
    print(sep)
    print("  CARGA DEL DATASET")
    print(sep)
    print(f"  Archivo     : {path.resolve()}")
    print(f"  Registros   : {len(df):,}")
    print(f"  Columnas    : {df.shape[1]}")
    print(f"  Memoria     : {mem_kb:.1f} KB")
    print()
    print("  Columnas y tipos de datos:")
    for col, dtype in df.dtypes.items():
        print(f"    • {col:<12} {dtype}")
    print()
    print("  Primeras 5 filas:")
    print(df.head().to_string(index=False))
    print(sep)
    print()
