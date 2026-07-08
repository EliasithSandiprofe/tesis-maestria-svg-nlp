"""
label_encoder.py
----------------
Fase 2 del preprocesamiento: codificación de las cuatro etiquetas objetivo
mediante ``sklearn.preprocessing.LabelEncoder``.

Se genera un encoder independiente por etiqueta, se serializa cada uno con
``joblib`` y se produce un archivo Markdown con todas las equivalencias.

NOTA DE DISEÑO
--------------
Los encoders se ajustan sobre el conjunto limpio completo (antes del split)
porque el espacio de clases es conocido a priori y estático. Esto garantiza
que los conjuntos de validación y prueba utilicen los mismos índices enteros
que el conjunto de entrenamiento, evitando inconsistencias en las cabezas de
clasificación del modelo.

Funciones públicas
------------------
encode_labels(df, encoders_dir, mapping_path) -> tuple[pd.DataFrame, dict]
    Ajusta y serializa los encoders; devuelve el DataFrame con las columnas
    ``<etiqueta>_enc`` añadidas y el diccionario de encoders.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]


def encode_labels(
    df: pd.DataFrame,
    encoders_dir: str | Path,
    mapping_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Codifica las cuatro etiquetas objetivo con LabelEncoder.

    Añade al DataFrame cuatro columnas adicionales con el sufijo ``_enc``
    (``color_enc``, ``estilo_enc``, ``elemento_enc``, ``posicion_enc``) que
    contienen los valores enteros correspondientes.

    Cada encoder se serializa como ``<etiqueta>_encoder.pkl`` en
    ``encoders_dir`` y se genera ``label_mapping.md`` con todas las
    equivalencias.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio (salida de ``clean_dataset``).
    encoders_dir : str | Path
        Directorio donde guardar los archivos ``.pkl``.
    mapping_path : str | Path
        Ruta del archivo ``label_mapping.md`` a generar.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        DataFrame con columnas ``_enc`` añadidas y diccionario
        ``{nombre_etiqueta: LabelEncoder}``.
    """
    encoders_dir = Path(encoders_dir)
    encoders_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    df_encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        df_encoded[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le

        pkl_path = encoders_dir / f"{col}_encoder.pkl"
        joblib.dump(le, pkl_path)

    _generate_mapping_md(encoders, mapping_path)
    _print_summary(encoders, encoders_dir)

    return df_encoded, encoders


# ── Helpers privados ─────────────────────────────────────────────────────────

def _generate_mapping_md(
    encoders: dict[str, LabelEncoder],
    path: Path,
) -> None:
    """Genera el archivo Markdown con todas las equivalencias clase → código."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Label Mapping — Codificación de Etiquetas SVG-NLP",
        "",
        f"> **Generado:** {timestamp}  ",
        "> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG",
        "",
        "Cada tabla muestra la equivalencia entre la clase original (string) y su",
        "representación entera utilizada durante el entrenamiento del modelo.",
        "",
    ]

    for col, le in encoders.items():
        lines += [
            f"## `{col}`  ({len(le.classes_)} clases)",
            "",
            "| Clase | Código entero |",
            "|---|---|",
        ]
        for code, cls in enumerate(le.classes_):
            lines.append(f"| `{cls}` | {code} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(
    encoders: dict[str, LabelEncoder],
    encoders_dir: Path,
) -> None:
    """Imprime el resumen de codificación en consola."""
    sep = "=" * 60
    print(sep)
    print("  FASE 2 — CODIFICACIÓN DE ETIQUETAS")
    print(sep)
    for col, le in encoders.items():
        mapping = ", ".join(f"{cls}→{i}" for i, cls in enumerate(le.classes_))
        print(f"  {col:<10}  ({len(le.classes_)} clases)  {mapping}")
    print()
    print(f"  Encoders guardados en : {encoders_dir}")
    print(sep)
    print()
