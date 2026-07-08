"""
clean_dataset.py
----------------
Fase 1 del preprocesamiento: limpieza del dataset mediante la eliminación
de registros completamente duplicados.

El archivo CSV original NO es modificado; se genera un nuevo archivo limpio.

Funciones públicas
------------------
clean_dataset(input_path, output_path) -> dict
    Elimina duplicados exactos, guarda el resultado y devuelve estadísticas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Elimina filas completamente duplicadas del dataset.

    Considera duplicado cualquier registro donde los cinco campos
    (``prompt``, ``color``, ``estilo``, ``elemento``, ``posicion``)
    son idénticos a otro registro existente.

    El archivo original en ``input_path`` NO es modificado.

    Parameters
    ----------
    input_path : str | Path
        Ruta al CSV original (``dataset_training.csv``).
    output_path : str | Path
        Ruta donde guardar el CSV limpio (``dataset_training_clean.csv``).

    Returns
    -------
    dict
        Estadísticas con las claves:
        ``original``, ``removed``, ``final``, ``pct_removed``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_original = pd.read_csv(input_path, encoding="utf-8")
    n_original = len(df_original)

    df_clean = df_original.drop_duplicates().reset_index(drop=True)
    n_removed = n_original - len(df_clean)
    pct_removed = round(n_removed / n_original * 100, 2)

    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    stats = {
        "original": n_original,
        "removed": n_removed,
        "final": len(df_clean),
        "pct_removed": pct_removed,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }

    _print_summary(stats)
    return stats


# ── Helpers privados ─────────────────────────────────────────────────────────

def _print_summary(stats: dict) -> None:
    """Imprime las estadísticas de limpieza en consola."""
    sep = "=" * 60
    print(sep)
    print("  FASE 1 — LIMPIEZA: DEDUPLICACIÓN")
    print(sep)
    print(f"  Registros originales   : {stats['original']:,}")
    print(f"  Duplicados eliminados  : {stats['removed']:,}  ({stats['pct_removed']}%)")
    print(f"  Registros finales      : {stats['final']:,}")
    print(f"  Archivo generado       : {Path(stats['output_path']).name}")
    print(sep)
    print()
