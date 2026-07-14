"""
report_generator.py
-------------------
Genera el reporte automático del EDA en formato Markdown:
``dataset/reports/eda_report.md``.

El reporte integra todos los resultados de los módulos anteriores y
produce conclusiones automáticas sobre la idoneidad del dataset para
entrenamiento.

Funciones públicas
------------------
generate_report(df, quality_results, label_distributions,
                prompt_stats, figures_dir, output_path) -> Path
    Escribe el archivo Markdown y devuelve su ruta.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


LABEL_COLUMNS: list[str] = ["color", "estilo", "elemento", "posicion"]

# Umbrales para conclusiones automáticas
MAX_NULL_PCT: float = 1.0       # % máximo aceptable de nulos por columna
MAX_DUPE_PCT: float = 5.0       # % máximo aceptable de duplicados
MIN_CLASSES: int = 2            # mínimo de clases por etiqueta
MIN_IMBALANCE_RATIO: float = 0.10  # clase minoritaria debe ser > 10 % del total


def generate_report(
    df: pd.DataFrame,
    quality_results: dict,
    label_distributions: dict[str, pd.DataFrame],
    prompt_stats: dict,
    figures_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Genera el reporte EDA completo en Markdown.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    quality_results : dict
        Resultado de ``quality_analysis.analyze_quality``.
    label_distributions : dict[str, pd.DataFrame]
        Resultado de ``label_analysis.analyze_labels``.
    prompt_stats : dict
        Resultado de ``prompt_analysis.analyze_prompts``.
    figures_dir : str | Path
        Ruta al directorio de figuras (relativa desde el reporte).
    output_path : str | Path
        Ruta de salida del archivo Markdown.

    Returns
    -------
    Path
        Ruta absoluta del archivo generado.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    figures_path = Path(figures_dir)
    conclusions = _build_conclusions(df, quality_results, label_distributions, prompt_stats)

    lines: list[str] = []

    _section_header(lines, df, figures_path)
    _section_quality(lines, quality_results, len(df))
    _section_labels(lines, label_distributions, figures_path)
    _section_prompts(lines, prompt_stats, figures_path)
    _section_interpretation(lines, quality_results, label_distributions, prompt_stats, len(df))
    _section_conclusions(
        lines, conclusions, quality_results, label_distributions, prompt_stats, len(df)
    )

    content = "\n".join(lines)
    out_file.write_text(content, encoding="utf-8")

    print("=" * 60)
    print("  REPORTE GENERADO")
    print("=" * 60)
    print(f"  Archivo : {out_file.resolve()}")
    print(f"  Tamaño  : {out_file.stat().st_size / 1024:.1f} KB")
    print()

    return out_file


# ── Secciones del reporte ─────────────────────────────────────────────────────

def _section_header(lines: list[str], df: pd.DataFrame, figures_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "# Reporte de Análisis Exploratorio de Datos (EDA)",
        "",
        "> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  ",
        f"> **Generado:** {timestamp}  ",
        f"> **Fase:** A — Análisis Exploratorio de Datos  ",
        "",
        "---",
        "",
        "## 1. Resumen del Dataset",
        "",
        f"| Métrica | Valor |",
        f"|---|---|",
        f"| Total de registros | {len(df):,} |",
        f"| Total de columnas | {df.shape[1]} |",
        f"| Columnas | {', '.join(f'`{c}`' for c in df.columns)} |",
        f"| Tipos de datos | {', '.join(str(t) for t in df.dtypes.unique())} |",
        f"| Uso de memoria | {df.memory_usage(deep=True).sum() / 1024:.1f} KB |",
        "",
        "### Primeras 5 filas",
        "",
        _df_to_md(df.head()),
        "",
    ]


def _section_quality(lines: list[str], results: dict, n_rows: int) -> None:
    dup = results["duplicates"]
    dupe_pct = round(dup["filas_duplicadas_completas"] / n_rows * 100, 2)
    prompt_dupe_pct = round(dup["prompts_duplicados"] / n_rows * 100, 2)

    lines += [
        "---",
        "",
        "## 2. Calidad del Dataset",
        "",
        "### 2.1 Valores Nulos",
        "",
        _df_to_md(results["nulls"]),
        "",
        "### 2.2 Registros Duplicados",
        "",
        f"| Métrica | Valor | % |",
        f"|---|---|---|",
        f"| Filas completamente duplicadas | {dup['filas_duplicadas_completas']:,} | {dupe_pct}% |",
        f"| Prompts duplicados | {dup['prompts_duplicados']:,} | {prompt_dupe_pct}% |",
        f"| Prompts únicos | {dup['prompts_unicos']:,} | {100 - prompt_dupe_pct:.2f}% |",
        "",
        "### 2.3 Cardinalidad por Columna",
        "",
        _df_to_md(results["unique_values"]),
        "",
        "### 2.4 Valores en Blanco",
        "",
        _df_to_md(results["blank_values"]),
        "",
        "### 2.5 Tabla Resumen de Calidad",
        "",
        _df_to_md(results["summary_table"]),
        "",
    ]


def _section_labels(
    lines: list[str],
    distributions: dict[str, pd.DataFrame],
    figures_path: Path,
) -> None:
    lines += [
        "---",
        "",
        "## 3. Distribución de Etiquetas",
        "",
    ]
    for col in LABEL_COLUMNS:
        dist = distributions[col]
        n_classes = len(dist)
        most_common = dist.iloc[0]["valor"]
        most_common_pct = dist.iloc[0]["porcentaje_%"]
        least_common = dist.iloc[-1]["valor"]
        least_common_pct = dist.iloc[-1]["porcentaje_%"]

        img_rel = figures_path / f"distribucion_{col}.png"

        lines += [
            f"### 3.{LABEL_COLUMNS.index(col) + 1} Etiqueta: `{col}`",
            "",
            f"- **Clases distintas:** {n_classes}",
            f"- **Clase más frecuente:** `{most_common}` ({most_common_pct}%)",
            f"- **Clase menos frecuente:** `{least_common}` ({least_common_pct}%)",
            "",
            _df_to_md(dist),
            "",
            f"![Distribución {col}]({img_rel})",
            "",
        ]


def _section_prompts(
    lines: list[str],
    prompt_stats: dict,
    figures_path: Path,
) -> None:
    img_chars = figures_path / "histograma_longitud_caracteres.png"
    img_words = figures_path / "histograma_longitud_palabras.png"

    lines += [
        "---",
        "",
        "## 4. Análisis de los Prompts",
        "",
        "### 4.1 Estadísticas de Longitud",
        "",
        _df_to_md(prompt_stats["stats_table"]),
        "",
        "### 4.2 Histograma — Longitud en Caracteres",
        "",
        f"![Histograma caracteres]({img_chars})",
        "",
        "### 4.3 Histograma — Longitud en Palabras",
        "",
        f"![Histograma palabras]({img_words})",
        "",
    ]


def _section_interpretation(
    lines: list[str],
    quality_results: dict,
    label_distributions: dict[str, pd.DataFrame],
    prompt_stats: dict,
    n_rows: int,
) -> None:
    dup = quality_results["duplicates"]
    char_stats = prompt_stats["stats_table"]
    char_row = char_stats[char_stats["metrica"] == "promedio"]
    char_mean = float(char_row["longitud_caracteres"].values[0]) if not char_row.empty else 0.0
    word_row = char_stats[char_stats["metrica"] == "promedio"]
    word_mean = float(word_row["longitud_palabras"].values[0]) if not word_row.empty else 0.0

    lines += [
        "---",
        "",
        "## 5. Interpretación de Resultados",
        "",
        "### 5.1 Calidad General",
        "",
    ]

    null_total = int(quality_results["nulls"]["nulos"].sum())
    if null_total == 0:
        lines.append("- **Sin valores nulos.** El dataset está completamente poblado.")
    else:
        lines.append(f"- Se detectaron **{null_total} valores nulos** que deben ser tratados antes del entrenamiento.")

    dupe_pct = round(dup["filas_duplicadas_completas"] / n_rows * 100, 2)
    if dupe_pct <= MAX_DUPE_PCT:
        lines.append(f"- **Duplicados aceptables** ({dupe_pct}% ≤ {MAX_DUPE_PCT}%). Se recomienda eliminarlos antes del entrenamiento.")
    else:
        lines.append(f"- **Alta tasa de duplicados** ({dupe_pct}%). Se requiere deduplicación.")

    lines += [
        "",
        "### 5.2 Balance de Clases",
        "",
    ]
    for col in LABEL_COLUMNS:
        dist = label_distributions[col]
        min_pct = float(dist["porcentaje_%"].min())
        max_pct = float(dist["porcentaje_%"].max())
        ratio = min_pct / max_pct if max_pct > 0 else 0
        if ratio >= MIN_IMBALANCE_RATIO:
            lines.append(f"- `{col}`: distribución **equilibrada** (ratio min/max = {ratio:.2f}).")
        else:
            lines.append(f"- `{col}`: distribución **desbalanceada** (ratio min/max = {ratio:.2f}). Considerar técnicas de balanceo.")

    lines += [
        "",
        "### 5.3 Longitud de Prompts",
        "",
        f"- Longitud media: **{char_mean:.1f} caracteres** / **{word_mean:.1f} palabras** por prompt.",
        "- Una longitud uniforme y moderada es favorable para el entrenamiento con modelos transformer.",
        "",
    ]


def _section_conclusions(
    lines: list[str],
    conclusions: dict,
    quality_results: dict,
    label_distributions: dict[str, pd.DataFrame],
    prompt_stats: dict,
    n_rows: int,
) -> None:
    """Genera la sección de conclusiones con narrativa académica y validación automática."""
    verdict = conclusions["verdict"]

    # ── Datos necesarios para la narrativa ──────────────────────────────────
    null_total = int(quality_results["nulls"]["nulos"].sum())
    blank_total = int(quality_results["blank_values"]["valores_en_blanco"].sum())
    dup = quality_results["duplicates"]
    n_dupes = dup["filas_duplicadas_completas"]
    dupe_pct = round(n_dupes / n_rows * 100, 2)

    char_stats = prompt_stats["stats_table"]
    char_mean_row = char_stats[char_stats["metrica"] == "promedio"]
    char_std_row  = char_stats[char_stats["metrica"] == "desv_std"]
    char_mean = float(char_mean_row["longitud_caracteres"].values[0]) if not char_mean_row.empty else 0.0
    char_std  = float(char_std_row["longitud_caracteres"].values[0])  if not char_std_row.empty  else 0.0
    word_mean_row = char_stats[char_stats["metrica"] == "promedio"]
    word_mean = float(word_mean_row["longitud_palabras"].values[0]) if not word_mean_row.empty else 0.0

    all_balanced = all(
        float(label_distributions[col]["porcentaje_%"].min())
        / float(label_distributions[col]["porcentaje_%"].max()) >= MIN_IMBALANCE_RATIO
        for col in LABEL_COLUMNS
        if float(label_distributions[col]["porcentaje_%"].max()) > 0
    )

    # ── Narrativa académica ──────────────────────────────────────────────────
    null_sentence = (
        "El análisis de calidad no detectó valores nulos en ninguna de las columnas del dataset."
        if null_total == 0
        else f"Se identificaron {null_total} valores nulos que deberán ser tratados antes del entrenamiento."
    )
    blank_sentence = (
        " Tampoco se encontraron valores en blanco, lo que confirma la integridad textual de los registros."
        if blank_total == 0
        else f" Sin embargo, se detectaron {blank_total} valores en blanco que requieren atención."
    )
    dupe_sentence = (
        f" Se identificaron {n_dupes} registros duplicados ({dupe_pct}% del total),"
        " los cuales serán eliminados durante la fase de preparación del dataset,"
        " previa a la división en conjuntos de entrenamiento, validación y prueba."
        if n_dupes > 0
        else " No se detectaron registros duplicados."
    )
    balance_sentence = (
        " La distribución de clases en las cuatro variables objetivo"
        f" ({', '.join(f'`{c}`' for c in LABEL_COLUMNS)})"
        " muestra un balance adecuado entre categorías, lo que favorece el aprendizaje"
        " uniforme del modelo sin requerir estrategias adicionales de sobre-muestreo o"
        " sub-muestreo en esta etapa."
        if all_balanced
        else " Se detectó desbalance de clases en una o más variables objetivo;"
        " se recomienda evaluar estrategias de balanceo antes del entrenamiento."
    )
    prompt_sentence = (
        f" Los prompts presentan una longitud homogénea, con una media de {char_mean:.1f} caracteres"
        f" ({word_mean:.1f} palabras) y una desviación estándar de {char_std:.1f} caracteres,"
        " lo que es consistente con los requisitos de entrada de modelos transformer"
        " basados en la arquitectura BERT."
    )
    verdict_sentence = (
        " En conjunto, el dataset reúne la calidad suficiente para proceder con la"
        " preparación de datos y el entrenamiento del modelo DistilBERT propuesto"
        " en esta investigación."
        if verdict == "APTO"
        else " No obstante, se identificaron aspectos que requieren corrección antes de"
        " proceder con el entrenamiento del modelo."
    )

    narrative = (
        null_sentence
        + blank_sentence
        + dupe_sentence
        + balance_sentence
        + prompt_sentence
        + verdict_sentence
    )

    lines += [
        "---",
        "",
        "## 6. Conclusión",
        "",
        narrative,
        "",
        "### Validación automática",
        "",
    ]
    for item in conclusions["points"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "---",
        "",
        "> *Reporte generado automáticamente por el módulo EDA — Fase A del proyecto de tesis.*",
    ]


# ── Conclusiones automáticas ──────────────────────────────────────────────────

def _build_conclusions(
    df: pd.DataFrame,
    quality_results: dict,
    label_distributions: dict[str, pd.DataFrame],
    prompt_stats: dict,
) -> dict:
    """Evalúa criterios de idoneidad y construye el veredicto final.

    Returns
    -------
    dict
        ``{"verdict": "APTO" | "REQUIERE REVISIÓN", "points": [str, ...]}``.
    """
    points: list[str] = []
    issues: int = 0

    n_rows = len(df)

    # ── Criterio 1: Nulos ────────────────────────────────────────────────────
    null_max_pct = float(quality_results["nulls"]["porcentaje_%"].max())
    if null_max_pct <= MAX_NULL_PCT:
        points.append(f"Sin valores nulos significativos (máx. {null_max_pct}% por columna). [OK]")
    else:
        points.append(f"Columnas con valores nulos ({null_max_pct}% máx.). Se requiere limpieza. [ADVERTENCIA]")
        issues += 1

    # ── Criterio 2: Duplicados ────────────────────────────────────────────────
    dup_pct = round(quality_results["duplicates"]["filas_duplicadas_completas"] / n_rows * 100, 2)
    if dup_pct <= MAX_DUPE_PCT:
        points.append(f"Tasa de duplicados dentro del umbral aceptable ({dup_pct}% ≤ {MAX_DUPE_PCT}%). [OK]")
    else:
        points.append(f"Alta tasa de duplicados ({dup_pct}%). Deduplicar antes del entrenamiento. [ADVERTENCIA]")
        issues += 1

    # ── Criterio 3: Cardinalidad de etiquetas ─────────────────────────────────
    for col in LABEL_COLUMNS:
        n_classes = len(label_distributions[col])
        if n_classes >= MIN_CLASSES:
            points.append(f"Etiqueta `{col}` con {n_classes} clases válidas. [OK]")
        else:
            points.append(f"Etiqueta `{col}` tiene menos de {MIN_CLASSES} clases. [ERROR]")
            issues += 2

    # ── Criterio 4: Balance de clases ─────────────────────────────────────────
    for col in LABEL_COLUMNS:
        dist = label_distributions[col]
        min_pct = float(dist["porcentaje_%"].min())
        max_pct = float(dist["porcentaje_%"].max())
        ratio = min_pct / max_pct if max_pct > 0 else 0
        if ratio >= MIN_IMBALANCE_RATIO:
            points.append(f"Balance de clases en `{col}` aceptable (ratio {ratio:.2f}). [OK]")
        else:
            points.append(f"Desbalance en `{col}` (ratio min/max = {ratio:.2f}). Evaluar estrategias de balanceo. [ADVERTENCIA]")
            issues += 1

    # ── Criterio 5: Tamaño del dataset ────────────────────────────────────────
    if n_rows >= 500:
        points.append(f"Tamaño del dataset suficiente para entrenamiento ({n_rows:,} registros). [OK]")
    else:
        points.append(f"Dataset pequeño ({n_rows:,} registros). Considerar aumento de datos. [ADVERTENCIA]")
        issues += 1

    verdict = "APTO" if issues == 0 else "REQUIERE REVISIÓN"
    return {"verdict": verdict, "points": points}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a tabla Markdown."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, separator] + rows)
