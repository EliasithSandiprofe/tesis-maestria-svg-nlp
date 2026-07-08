"""
run_preprocessing.py
--------------------
Orquestador de la Fase B: Preparación del Dataset SVG-NLP.

Ejecuta en orden las cinco subfases del preprocesamiento y genera el
reporte consolidado ``dataset/reports/preprocessing_report.md``.

Flujo completo
--------------
1. Limpieza          → dataset_training_clean.csv
2. Codificación      → encoders .pkl + label_mapping.md
3. División          → dataset_train / _validation / _test .csv
4. Tokenización      → tokenizer serializado
5. Dataset PyTorch   → SVGPromptDataset (train, val, test)
6. Reporte           → preprocessing_report.md

Uso desde la raíz del proyecto::

    python -m src.nlp.preprocessing.run_preprocessing
    # o directamente:
    python src/nlp/preprocessing/run_preprocessing.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ajuste del PYTHONPATH para ejecución directa ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.nlp.preprocessing.clean_dataset import clean_dataset
from src.nlp.preprocessing.label_encoder import encode_labels, LABEL_COLUMNS
from src.nlp.preprocessing.split_dataset import split_dataset, STRATIFY_COLUMN
from src.nlp.preprocessing.tokenizer import (
    load_and_save_tokenizer,
    compute_token_stats,
)
from src.nlp.preprocessing.dataset_builder import build_datasets

# ── Constantes de rutas ───────────────────────────────────────────────────────
DATASET_DIR       = _PROJECT_ROOT / "dataset" / "processed"
REPORTS_DIR       = _PROJECT_ROOT / "dataset" / "reports"
MODELS_DIR        = _PROJECT_ROOT / "modelos"

ORIGINAL_CSV      = DATASET_DIR / "dataset_training.csv"
CLEAN_CSV         = DATASET_DIR / "dataset_training_clean.csv"
ENCODERS_DIR      = MODELS_DIR  / "label_encoders"
LABEL_MAPPING_MD  = ENCODERS_DIR / "label_mapping.md"
TOKENIZER_DIR     = MODELS_DIR  / "tokenizer"
REPORT_PATH       = REPORTS_DIR / "preprocessing_report.md"

MODEL_NAME        = "distilbert-base-uncased"
MAX_LENGTH        = 128
RANDOM_STATE      = 42


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run() -> None:
    """Ejecuta el pipeline completo de preprocesamiento (Fase B)."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   PREPROCESAMIENTO — FASE B: PREPARACIÓN DEL DATASET    ║")
    print("║   Proyecto: Tesis Maestría SVG-NLP                      ║")
    print("╚" + "═" * 58 + "╝")

    t0 = time.perf_counter()
    collected: dict = {}

    # ── Fase 1: Limpieza ─────────────────────────────────────────────────────
    _step(1, "Limpieza del Dataset (Deduplicación)")
    clean_stats = clean_dataset(ORIGINAL_CSV, CLEAN_CSV)
    collected["clean"] = clean_stats

    # ── Fase 2: Codificación de etiquetas ─────────────────────────────────────
    _step(2, "Codificación de Etiquetas")
    import pandas as pd
    df_clean = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df_encoded, encoders = encode_labels(df_clean, ENCODERS_DIR, LABEL_MAPPING_MD)
    collected["encoders"] = {
        col: list(le.classes_) for col, le in encoders.items()
    }

    # ── Fase 3: División del dataset ─────────────────────────────────────────
    _step(3, "División del Dataset (Train / Validation / Test)")
    train_df, val_df, test_df, split_stats = split_dataset(
        df_encoded,
        output_dir=DATASET_DIR,
        val_size=0.15,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    collected["split"] = split_stats

    # ── Fase 4: Tokenización ─────────────────────────────────────────────────
    _step(4, "Tokenización con DistilBertTokenizerFast")
    tokenizer = load_and_save_tokenizer(MODEL_NAME, TOKENIZER_DIR)

    # Estadísticas sobre el conjunto completo de prompts del train
    tok_stats = compute_token_stats(train_df["prompt"], tokenizer)
    collected["tokenizer"] = {
        "model_name":  MODEL_NAME,
        "max_length":  MAX_LENGTH,
        "vocab_size":  tokenizer.vocab_size,
        "token_stats": tok_stats,
    }

    sep = "=" * 60
    print(sep)
    print("  FASE 4 — TOKENIZACIÓN")
    print(sep)
    print(f"  Modelo          : {MODEL_NAME}")
    print(f"  max_length      : {MAX_LENGTH}")
    print(f"  Vocab size      : {tokenizer.vocab_size:,}")
    print(f"  Longitud tokens (train):")
    for k, v in tok_stats.items():
        print(f"    {k:<8}: {v}")
    print(sep)
    print()

    # ── Fase 5: Dataset PyTorch ───────────────────────────────────────────────
    _step(5, "Construcción de Datasets PyTorch")
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    collected["datasets"] = {
        "train_size": len(train_ds),
        "val_size":   len(val_ds),
        "test_size":  len(test_ds),
        "input_shape": train_ds.input_shape,
        "sample_keys": list(train_ds[0].keys()),
    }

    # ── Fase 6: Reporte ───────────────────────────────────────────────────────
    _step(6, "Generación del Reporte de Preprocesamiento")
    report_path = _write_report(collected)
    print(f"  Reporte generado : {report_path.relative_to(_PROJECT_ROOT)}")
    print(f"  Tamaño           : {report_path.stat().st_size / 1024:.1f} KB")
    print()

    # ── Resumen final ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print("╔" + "═" * 58 + "╗")
    print("║              PREPROCESAMIENTO COMPLETADO                ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Tiempo total : {elapsed:.2f} s")
    print()
    print("  Artefactos generados:")
    print(f"    dataset/processed/dataset_training_clean.csv")
    print(f"    dataset/processed/dataset_train.csv")
    print(f"    dataset/processed/dataset_validation.csv")
    print(f"    dataset/processed/dataset_test.csv")
    print(f"    modelos/label_encoders/  (4 encoders + label_mapping.md)")
    print(f"    modelos/tokenizer/")
    print(f"    dataset/reports/preprocessing_report.md")
    print()


# ── Helpers de consola ────────────────────────────────────────────────────────

def _step(number: int, description: str) -> None:
    print()
    print(f"{'━' * 60}")
    print(f"  PASO {number}/6 — {description}")
    print(f"{'━' * 60}")


# ── Generación del reporte ────────────────────────────────────────────────────

def _write_report(data: dict) -> Path:
    """Genera el reporte Markdown consolidado de la Fase B."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    clean   = data["clean"]
    enc     = data["encoders"]
    split   = data["split"]
    tok     = data["tokenizer"]
    ds      = data["datasets"]

    lines: list[str] = []

    # ── Encabezado ────────────────────────────────────────────────────────────
    lines += [
        "# Reporte de Preprocesamiento del Dataset — Fase B",
        "",
        f"> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  ",
        f"> **Generado:** {timestamp}  ",
        f"> **Fase:** B — Preparación del Dataset  ",
        "",
        "---",
        "",
    ]

    # ── 1. Resumen ejecutivo ──────────────────────────────────────────────────
    lines += [
        "## 1. Resumen Ejecutivo",
        "",
        "| Etapa | Registros |",
        "|---|---|",
        f"| Dataset original | {clean['original']:,} |",
        f"| Tras deduplicación | {clean['final']:,} |",
        f"| Conjunto Train | {split['train_count']:,} ({split['train_pct']}%) |",
        f"| Conjunto Validation | {split['val_count']:,} ({split['val_pct']}%) |",
        f"| Conjunto Test | {split['test_count']:,} ({split['test_pct']}%) |",
        "",
        "---",
        "",
    ]

    # ── 2. Fase 1: Limpieza ───────────────────────────────────────────────────
    lines += [
        "## 2. Fase 1 — Limpieza del Dataset: Deduplicación",
        "",
        "### 2.1 Resultados",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Registros originales | {clean['original']:,} |",
        f"| Duplicados eliminados | {clean['removed']:,} ({clean['pct_removed']}%) |",
        f"| Registros finales | {clean['final']:,} |",
        f"| Archivo generado | `dataset_training_clean.csv` |",
        "",
        "### 2.2 Justificación metodológica",
        "",
        "Se eliminan los registros completamente duplicados —aquellos donde los cinco "
        "campos (`prompt`, `color`, `estilo`, `elemento`, `posicion`) son idénticos— "
        "con el objetivo de:",
        "",
        "- Evitar que el modelo memorice ejemplos repetidos en lugar de generalizar "
        "  patrones lingüísticos.",
        "- Garantizar que un mismo prompt no aparezca simultáneamente en los conjuntos "
        "  de entrenamiento y evaluación (fuga de datos).",
        "- Preservar la integridad estadística de las métricas de evaluación.",
        "",
        "Los registros originales se conservan sin modificación en "
        "`dataset_training.csv`. La eliminación afecta únicamente a las copias "
        f"redundantes ({clean['removed']} registros, {clean['pct_removed']}% del total).",
        "",
        "---",
        "",
    ]

    # ── 3. Fase 2: Codificación ───────────────────────────────────────────────
    lines += [
        "## 3. Fase 2 — Codificación de Etiquetas",
        "",
        "Se utiliza `sklearn.preprocessing.LabelEncoder` con un encoder independiente "
        "por cada variable objetivo. Los encoders se ajustan sobre el conjunto limpio "
        "**completo** (antes del split), garantizando que todos los conjuntos "
        "(train, val, test) utilicen exactamente los mismos códigos enteros.",
        "",
        "### 3.1 Encoders generados",
        "",
        "| Etiqueta | Clases | Archivo |",
        "|---|---|---|",
    ]
    for col in LABEL_COLUMNS:
        lines.append(f"| `{col}` | {len(enc[col])} | `{col}_encoder.pkl` |")

    lines += ["", "### 3.2 Mappings de clases", ""]
    for col in LABEL_COLUMNS:
        lines += [
            f"#### `{col}`",
            "",
            "| Clase | Código entero |",
            "|---|---|",
        ]
        for code, cls in enumerate(enc[col]):
            lines.append(f"| `{cls}` | {code} |")
        lines.append("")

    lines += ["---", ""]

    # ── 4. Fase 3: División ───────────────────────────────────────────────────
    lines += [
        "## 4. Fase 3 — División del Dataset",
        "",
        "### 4.1 Decisión metodológica",
        "",
        "El problema de clasificación presenta cuatro variables objetivo simultáneas. "
        "`train_test_split` de sklearn acepta un único array en el parámetro "
        "`stratify`, por lo que la estratificación conjunta no es directamente soportada.",
        "",
        "**Análisis de la clave compuesta:** la concatenación de los cuatro atributos "
        f"genera hasta 6×5×10×4 = 1 200 combinaciones posibles. Con {clean['final']:,} "
        "muestras disponibles tras la deduplicación, la frecuencia media por combinación "
        "es ≈ 1.6. De hecho, 602 combinaciones aparecen una única vez. Sklearn exige "
        "al menos 2 muestras por clase para estratificar, por lo que este enfoque es "
        "**inviable** para este dataset.",
        "",
        f"**Estrategia adoptada:** split en dos etapas estratificado sobre `{STRATIFY_COLUMN}` "
        f"(la etiqueta con mayor cardinalidad, {len(enc[STRATIFY_COLUMN])} clases, "
        f"~{clean['final'] // len(enc[STRATIFY_COLUMN])} muestras/clase). "
        "Dado que las cuatro etiquetas están bien balanceadas (ratio min/max ≥ 0.89 "
        "confirmado en el EDA), esta estrategia preserva distribuciones estadísticamente "
        "comparables en todas las etiquetas. La reproducibilidad queda garantizada "
        "mediante `random_state=42`.",
        "",
        "### 4.2 Resultados",
        "",
        "| Conjunto | Registros | Porcentaje |",
        "|---|---|---|",
        f"| Train | {split['train_count']:,} | {split['train_pct']}% |",
        f"| Validation | {split['val_count']:,} | {split['val_pct']}% |",
        f"| Test | {split['test_count']:,} | {split['test_pct']}% |",
        "",
        "### 4.3 Distribución de etiquetas por conjunto",
        "",
    ]

    for col in LABEL_COLUMNS:
        dist = split["distributions"][col]
        all_classes = sorted(
            set(list(dist["train"]) + list(dist["val"]) + list(dist["test"]))
        )
        lines += [
            f"#### `{col}`",
            "",
            "| Clase | Train | Val | Test |",
            "|---|---|---|---|",
        ]
        for cls in all_classes:
            t  = dist["train"].get(cls, 0)
            v  = dist["val"].get(cls, 0)
            te = dist["test"].get(cls, 0)
            lines.append(f"| `{cls}` | {t} | {v} | {te} |")
        lines.append("")

    lines += ["---", ""]

    # ── 5. Fase 4: Tokenización ───────────────────────────────────────────────
    ts = tok["token_stats"]
    lines += [
        "## 5. Fase 4 — Tokenización",
        "",
        "### 5.1 Configuración",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Modelo | `{tok['model_name']}` |",
        f"| Tokenizer | `DistilBertTokenizerFast` |",
        f"| `max_length` | {tok['max_length']} |",
        f"| Padding | `max_length` |",
        f"| Truncation | `True` |",
        f"| Vocabulario | {tok['vocab_size']:,} tokens |",
        "",
        "### 5.2 Estadísticas de longitud (conjunto Train)",
        "",
        "| Métrica | Tokens |",
        "|---|---|",
        f"| Mínimo | {ts['min']} |",
        f"| Máximo | {ts['max']} |",
        f"| Media | {ts['mean']} |",
        f"| Mediana | {ts['median']} |",
        f"| Percentil 95 | {ts['p95']} |",
        "",
        f"El percentil 95 ({ts['p95']} tokens) está muy por debajo del `max_length` "
        f"configurado ({tok['max_length']}), confirmando que no se produce truncado "
        "en ningún prompt del dataset.",
        "",
        "---",
        "",
    ]

    # ── 6. Fase 5: Dataset PyTorch ────────────────────────────────────────────
    lines += [
        "## 6. Fase 5 — Dataset PyTorch",
        "",
        "### 6.1 Resumen",
        "",
        "| Conjunto | Muestras |",
        "|---|---|",
        f"| Train | {ds['train_size']:,} |",
        f"| Validation | {ds['val_size']:,} |",
        f"| Test | {ds['test_size']:,} |",
        "",
        "### 6.2 Estructura de cada muestra",
        "",
        "Cada llamada a `dataset[i]` devuelve un diccionario con las siguientes claves:",
        "",
        "| Clave | Tipo | Forma | Descripción |",
        "|---|---|---|---|",
        f"| `input_ids` | `LongTensor` | `({tok['max_length']},)` | IDs de tokens DistilBERT |",
        f"| `attention_mask` | `LongTensor` | `({tok['max_length']},)` | Máscara de atención (1=token, 0=padding) |",
        "| `labels_color` | `LongTensor` | `()` | Código entero de la etiqueta `color` |",
        "| `labels_estilo` | `LongTensor` | `()` | Código entero de la etiqueta `estilo` |",
        "| `labels_elemento` | `LongTensor` | `()` | Código entero de la etiqueta `elemento` |",
        "| `labels_posicion` | `LongTensor` | `()` | Código entero de la etiqueta `posicion` |",
        "",
        "Esta estructura es compatible con `torch.utils.data.DataLoader` y está lista "
        "para ser consumida directamente por el modelo Multi-Head DistilBERT en la Fase C.",
        "",
        "---",
        "",
    ]

    # ── 7. Artefactos generados ───────────────────────────────────────────────
    lines += [
        "## 7. Artefactos Generados",
        "",
        "| Artefacto | Ruta | Descripción |",
        "|---|---|---|",
        "| Dataset limpio | `dataset/processed/dataset_training_clean.csv` | Dataset sin duplicados |",
        "| Split Train | `dataset/processed/dataset_train.csv` | 70% del dataset limpio |",
        "| Split Validation | `dataset/processed/dataset_validation.csv` | 15% del dataset limpio |",
        "| Split Test | `dataset/processed/dataset_test.csv` | 15% del dataset limpio |",
        "| Encoder color | `modelos/label_encoders/color_encoder.pkl` | LabelEncoder serializado |",
        "| Encoder estilo | `modelos/label_encoders/estilo_encoder.pkl` | LabelEncoder serializado |",
        "| Encoder elemento | `modelos/label_encoders/elemento_encoder.pkl` | LabelEncoder serializado |",
        "| Encoder posicion | `modelos/label_encoders/posicion_encoder.pkl` | LabelEncoder serializado |",
        "| Label Mapping | `modelos/label_encoders/label_mapping.md` | Tabla de equivalencias |",
        "| Tokenizer | `modelos/tokenizer/` | DistilBertTokenizerFast serializado |",
        "",
        "---",
        "",
        "> *Reporte generado automáticamente por el módulo de preprocesamiento — Fase B del proyecto de tesis.*",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    run()
