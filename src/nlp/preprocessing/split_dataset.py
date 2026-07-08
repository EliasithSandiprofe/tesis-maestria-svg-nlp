"""
split_dataset.py
----------------
Fase 3 del preprocesamiento: división del dataset en conjuntos de
entrenamiento (70%), validación (15%) y prueba (15%).

DECISIÓN METODOLÓGICA — ESTRATIFICACIÓN MULTIETIQUETA
------------------------------------------------------
El problema posee cuatro variables objetivo simultáneas (``color``,
``estilo``, ``elemento``, ``posicion``). ``train_test_split`` de sklearn
acepta un único array unidimensional en el parámetro ``stratify``, por lo que
la estratificación conjunta sobre las cuatro etiquetas no es directamente
soportada.

La clave compuesta (concatenación de los cuatro valores) genera hasta
6×5×10×4 = 1 200 combinaciones. Con 1 920 muestras tras la deduplicación,
la frecuencia media por combinación es ≈ 1.6 y 602 combinaciones aparecen
una única vez. sklearn exige al menos 2 muestras por clase para poder
estratificar, por lo que este enfoque es inviable.

Estrategia adoptada (dos etapas, estratificación por ``elemento``):
  1. Se estratifica sobre la etiqueta ``elemento`` (10 clases, ~192
     muestras/clase), que posee la mayor cardinalidad entre las cuatro
     etiquetas y es la más difícil de balancear por azar.
  2. Dado que las cuatro etiquetas están bien balanceadas (ratio min/max ≥ 0.89
     confirmado en el EDA), estratificar sobre ``elemento`` preserva
     distribuciones estadísticamente comparables en las demás etiquetas.
  3. Se aplica un split en dos fases con ``random_state=42`` para garantizar
     la reproducibilidad completa del experimento.

Esta aproximación es la práctica estándar aceptada en clasificación
multi-output con etiquetas balanceadas cuando no se dispone de librerías
de estratificación multietiqueta (p. ej., ``iterative-stratification``).

Funciones públicas
------------------
split_dataset(df, output_dir, val_size, test_size, random_state)
    -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]
STRATIFY_COLUMN: str = "elemento"  # Mayor cardinalidad entre las etiquetas


def split_dataset(
    df: pd.DataFrame,
    output_dir: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    La división se realiza en dos etapas estratificadas sobre la columna
    ``elemento``. Consultar la documentación del módulo para la justificación
    metodológica completa.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset codificado (salida de ``encode_labels``).
    output_dir : str | Path
        Directorio donde guardar los tres archivos CSV resultantes.
    val_size : float
        Proporción del conjunto de validación (por defecto 0.15).
    test_size : float
        Proporción del conjunto de prueba (por defecto 0.15).
    random_state : int
        Semilla para reproducibilidad (por defecto 42).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
        ``(train_df, val_df, test_df, stats)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(df)
    temp_fraction = val_size + test_size          # 0.30
    relative_val = val_size / temp_fraction       # 0.50

    # ── Etapa 1: train vs (val + test) ───────────────────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_fraction,
        random_state=random_state,
        stratify=df[STRATIFY_COLUMN],
    )

    # ── Etapa 2: val vs test (dentro de temp) ────────────────────────────────
    # Verificar que todas las clases de estratificación siguen teniendo ≥ 2
    # muestras en el subconjunto temporal (garantizado por el tamaño).
    strat_temp = temp_df[STRATIFY_COLUMN]
    if strat_temp.value_counts().min() < 2:
        # Salvaguarda: si alguna clase quedó con < 2, no estratificar.
        strat_temp = None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - relative_val,
        random_state=random_state,
        stratify=strat_temp,
    )

    # ── Reset de índices ─────────────────────────────────────────────────────
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── Guardar CSVs ─────────────────────────────────────────────────────────
    train_df.to_csv(output_dir / "dataset_train.csv",      index=False, encoding="utf-8")
    val_df.to_csv(  output_dir / "dataset_validation.csv", index=False, encoding="utf-8")
    test_df.to_csv( output_dir / "dataset_test.csv",       index=False, encoding="utf-8")

    # ── Estadísticas ─────────────────────────────────────────────────────────
    stats = _build_stats(
        train_df, val_df, test_df,
        n_total=n_total,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    _print_summary(stats)

    return train_df, val_df, test_df, stats


# ── Helpers privados ─────────────────────────────────────────────────────────

def _build_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_total: int,
    val_size: float,
    test_size: float,
    random_state: int,
) -> dict:
    """Construye el diccionario de estadísticas del split."""
    n_used = len(train_df) + len(val_df) + len(test_df)

    distributions: dict = {}
    for col in LABEL_COLUMNS:
        distributions[col] = {
            "train": train_df[col].value_counts().sort_index().to_dict(),
            "val":   val_df[col].value_counts().sort_index().to_dict(),
            "test":  test_df[col].value_counts().sort_index().to_dict(),
        }

    return {
        "strategy":     f"two_stage_stratified_on_{STRATIFY_COLUMN}",
        "random_state": random_state,
        "val_size":     val_size,
        "test_size":    test_size,
        "n_total":      n_total,
        "train_count":  len(train_df),
        "val_count":    len(val_df),
        "test_count":   len(test_df),
        "train_pct":    round(len(train_df) / n_used * 100, 2),
        "val_pct":      round(len(val_df)   / n_used * 100, 2),
        "test_pct":     round(len(test_df)  / n_used * 100, 2),
        "distributions": distributions,
    }


def _print_summary(stats: dict) -> None:
    """Imprime el resumen de la división en consola."""
    sep = "=" * 60
    print(sep)
    print("  FASE 3 — DIVISIÓN DEL DATASET")
    print(sep)
    print(f"  Estrategia    : {stats['strategy']}")
    print(f"  random_state  : {stats['random_state']}")
    print()
    print(f"  {'Conjunto':<14} {'Registros':>10} {'Porcentaje':>12}")
    print(f"  {'-' * 38}")
    print(f"  {'Train':<14} {stats['train_count']:>10,} {stats['train_pct']:>11.2f}%")
    print(f"  {'Validation':<14} {stats['val_count']:>10,} {stats['val_pct']:>11.2f}%")
    print(f"  {'Test':<14} {stats['test_count']:>10,} {stats['test_pct']:>11.2f}%")
    print()
    print("  Distribución de 'elemento' por conjunto:")
    elem_dist = stats["distributions"]["elemento"]
    all_classes = sorted(
        set(list(elem_dist["train"]) + list(elem_dist["val"]) + list(elem_dist["test"]))
    )
    print(f"  {'Clase':<14} {'Train':>6} {'Val':>6} {'Test':>6}")
    print(f"  {'-' * 35}")
    for cls in all_classes:
        t = elem_dist["train"].get(cls, 0)
        v = elem_dist["val"].get(cls, 0)
        te = elem_dist["test"].get(cls, 0)
        print(f"  {cls:<14} {t:>6} {v:>6} {te:>6}")
    print(sep)
    print()
