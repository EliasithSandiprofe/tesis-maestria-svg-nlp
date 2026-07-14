"""
evaluation_utils.py
-------------------
Utilidades de la Fase E: carga del checkpoint entrenado, serialización de
métricas, generación de matrices de confusión y reporte Markdown final.

Funciones públicas
------------------
load_model_from_checkpoint   : Carga best_model.pt y devuelve el modelo listo.
load_label_encoders_for_eval : Carga los LabelEncoder desde disco.
save_metrics_json            : Serializa las métricas de evaluación a JSON.
save_classification_report_md: Guarda el classification_report en Markdown.
plot_confusion_matrix        : Genera y guarda la matriz de confusión como PNG.
generate_evaluation_report   : Genera evaluation_report.md completo.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")   # backend no interactivo

_TASKS: list[str] = ["color", "estilo", "elemento", "posicion"]


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    project_root: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Carga ``best_model.pt`` y devuelve el modelo listo para inferencia.

    Pasos internos:
    1. Carga el checkpoint con ``torch.load``.
    2. Construye ``SVGModelConfig`` a partir de los encoders en disco.
    3. Instancia ``MultiTaskDistilBERT``.
    4. Carga el ``model_state_dict`` del checkpoint.
    5. Mueve el modelo al dispositivo y lo pone en modo ``eval()``.

    Parameters
    ----------
    checkpoint_path : str | Path
        Ruta al archivo ``best_model.pt``.
    project_root : str | Path
        Raíz del proyecto (para localizar ``modelos/label_encoders/``).
    device : torch.device
        Dispositivo destino.

    Returns
    -------
    tuple[nn.Module, dict]
        ``(model, checkpoint_info)`` donde ``checkpoint_info`` contiene
        ``epoch``, ``val_loss`` y ``val_metrics`` del checkpoint.

    Raises
    ------
    FileNotFoundError
        Si el archivo del checkpoint no existe.
    """
    import sys
    project_root = Path(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.nlp.modeling.model_config import SVGModelConfig
    from src.nlp.modeling.model_utils import build_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {checkpoint_path}\n"
            "Ejecuta primero train.py (Fase D)."
        )

    print(f"  Cargando checkpoint: {checkpoint_path.name}...", end=" ", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print("OK")

    # Construir modelo con la configuración guardada en el checkpoint
    model_config = SVGModelConfig.from_project(project_root)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"  Checkpoint: época {checkpoint.get('epoch', '?')}  |  "
        f"val_loss = {checkpoint.get('val_loss', float('nan')):.4f}"
    )

    info = {
        "epoch":       checkpoint.get("epoch"),
        "val_loss":    checkpoint.get("val_loss"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "num_classes": checkpoint.get("num_classes", {}),
    }
    return model, info


# ── Carga de encoders ─────────────────────────────────────────────────────────

def load_label_encoders_for_eval(
    encoders_dir: str | Path,
) -> dict[str, LabelEncoder]:
    """Carga los LabelEncoder serializados desde ``modelos/label_encoders/``.

    Parameters
    ----------
    encoders_dir : str | Path
        Directorio con los archivos ``<tarea>_encoder.pkl``.

    Returns
    -------
    dict[str, LabelEncoder]
        Diccionario ``{nombre_tarea: LabelEncoder}``.
    """
    encoders_dir = Path(encoders_dir)
    encoders: dict[str, LabelEncoder] = {}
    for task in _TASKS:
        path = encoders_dir / f"{task}_encoder.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Encoder no encontrado: {path}")
        encoders[task] = joblib.load(path)
    return encoders


# ── Persistencia de métricas ──────────────────────────────────────────────────

def save_metrics_json(
    results: dict,
    path: str | Path,
    checkpoint_path: str | Path | None = None,
    test_csv_path: str | Path | None = None,
) -> None:
    """Serializa las métricas de evaluación en un archivo JSON.

    Excluye ``y_true`` y ``y_pred`` (arrays grandes) para mantener el JSON
    legible.  Incluye ``confusion_matrix`` como lista de listas.

    Parameters
    ----------
    results : dict
        Salida de ``MultiTaskEvaluator.run()``.
    path : str | Path
        Ruta del archivo JSON de salida.
    checkpoint_path : str | Path | None
        Ruta del checkpoint evaluado (para metadatos).
    test_csv_path : str | Path | None
        Ruta del CSV de prueba (para metadatos).
    """
    payload: dict = {
        "generated":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint":   str(checkpoint_path) if checkpoint_path else "unknown",
        "test_dataset": str(test_csv_path)   if test_csv_path   else "unknown",
        "global":       results.get("global", {}),
        "per_task":     {},
    }

    for task in _TASKS:
        t = results[task]
        payload["per_task"][task] = {
            "n_classes":       t["n_classes"],
            "class_names":     t["class_names"],
            "accuracy":        t["accuracy"],
            "precision_macro": t["precision_macro"],
            "recall_macro":    t["recall_macro"],
            "f1_macro":        t["f1_macro"],
            "confusion_matrix": t["confusion_matrix"],
        }

    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def save_classification_report_md(
    report_str: str,
    task: str,
    path: str | Path,
) -> None:
    """Guarda el ``classification_report`` de sklearn en formato Markdown.

    Parameters
    ----------
    report_str : str
        Cadena producida por ``sklearn.metrics.classification_report``.
    task : str
        Nombre de la tarea (para el título del documento).
    path : str | Path
        Ruta del archivo Markdown de salida.
    """
    lines = [
        f"# Classification Report — `{task}`",
        "",
        f"> Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "```",
        report_str.rstrip(),
        "```",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Visualización: matrices de confusión ──────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    task: str,
    path: str | Path,
) -> None:
    """Genera y guarda la matriz de confusión como imagen PNG.

    Utiliza exclusivamente ``matplotlib``.  Las celdas de la diagonal
    principal se destacan en azul más intenso y el texto se adapta al
    contraste del fondo.

    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusión de forma ``(n_classes, n_classes)``.
    class_names : list[str]
        Nombres de las clases en el mismo orden que los ejes de ``cm``.
    task : str
        Nombre de la tarea (para el título de la figura).
    path : str | Path
        Ruta del archivo PNG de salida.
    """
    n = len(class_names)
    fig_w = max(6.0, n * 0.85)
    fig_h = max(5.0, n * 0.75)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Anotaciones numéricas en cada celda
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=9, color=color,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xlabel("Predicción", fontsize=10)
    ax.set_ylabel("Etiqueta real", fontsize=10)
    ax.set_title(f"Matriz de Confusión — `{task}`", fontsize=12, pad=14)

    fig.tight_layout()
    fig.savefig(Path(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Reporte Markdown final ────────────────────────────────────────────────────

def generate_evaluation_report(
    results: dict,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    test_csv_path: str | Path | None = None,
    checkpoint_info: dict | None = None,
    figures_rel_dir: str = ".",
) -> Path:
    """Genera el reporte Markdown completo de la Fase E.

    Parameters
    ----------
    results : dict
        Salida de ``MultiTaskEvaluator.run()``.
    output_path : str | Path
        Ruta del archivo Markdown a generar.
    checkpoint_path : str | Path | None
        Ruta del checkpoint evaluado.
    test_csv_path : str | Path | None
        Ruta del CSV de prueba.
    checkpoint_info : dict | None
        Información extra del checkpoint (``epoch``, ``val_loss``).
    figures_rel_dir : str
        Prefijo relativo para las rutas de imágenes en el Markdown.

    Returns
    -------
    Path
        Ruta absoluta del archivo generado.
    """
    output_path = Path(output_path)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    glob        = results.get("global", {})
    ckpt_info   = checkpoint_info or {}
    n_samples   = glob.get("n_test_samples", "N/A")

    lines: list[str] = [
        "# Reporte de Evaluación Final — Fase E",
        "",
        "> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  ",
        f"> **Generado:** {timestamp}  ",
        "> **Fase:** E — Evaluación final del modelo  ",
        "",
        "---",
        "",
        "## 1. Configuración de la Evaluación",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Dispositivo | `{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}` |",
        f"| Checkpoint evaluado | `{Path(checkpoint_path).name if checkpoint_path else 'best_model.pt'}` |",
        f"| Época del checkpoint | {ckpt_info.get('epoch', 'N/A')} |",
        f"| Val loss (entrenamiento) | {ckpt_info.get('val_loss', float('nan')):.4f} |" if ckpt_info.get('val_loss') else f"| Val loss (entrenamiento) | N/A |",
        f"| Dataset de prueba | `{Path(test_csv_path).name if test_csv_path else 'dataset_test.csv'}` |",
        f"| Muestras test | {n_samples:,} |" if isinstance(n_samples, int) else f"| Muestras test | {n_samples} |",
        "",
        "---",
        "",
        "## 2. Métricas Globales",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| **Mean Accuracy** | **{glob.get('mean_accuracy', 0)*100:.2f}%** |",
        f"| **Mean F1 Macro** | **{glob.get('mean_f1_macro', 0)*100:.2f}%** |",
        "",
        "---",
        "",
        "## 3. Métricas por Tarea",
        "",
    ]

    for idx, task in enumerate(_TASKS, start=1):
        m = results[task]
        img_path = f"{figures_rel_dir}/confusion_matrix_{task}.png".lstrip("./")
        if figures_rel_dir == ".":
            img_path = f"confusion_matrix_{task}.png"

        lines += [
            f"### 3.{idx} `{task}` ({m['n_classes']} clases)",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| Accuracy | {m['accuracy']*100:.2f}% |",
            f"| Precision Macro | {m['precision_macro']*100:.2f}% |",
            f"| Recall Macro | {m['recall_macro']*100:.2f}% |",
            f"| F1 Macro | {m['f1_macro']*100:.2f}% |",
            "",
            f"**Clases:** {', '.join(f'`{c}`' for c in m['class_names'])}",
            "",
            f"![Matriz de confusión — {task}]({img_path})",
            "",
        ]

    lines += [
        "---",
        "",
        "## 4. Interpretación de Resultados",
        "",
        _generate_interpretation(results),
        "",
        "---",
        "",
        "## 5. Artefactos Generados",
        "",
        "| Artefacto | Descripción |",
        "|---|---|",
        "| `evaluation_metrics.json` | Métricas completas en formato JSON |",
        "| `evaluation_report.md` | Este reporte |",
    ]
    for task in _TASKS:
        lines.append(f"| `classification_report_{task}.md` | Reporte detallado por clase — `{task}` |")
        lines.append(f"| `confusion_matrix_{task}.png` | Matriz de confusión — `{task}` |")

    lines += [
        "",
        "---",
        "",
        "> *Generado automáticamente por `evaluate.py` — Fase E del proyecto de tesis.*",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_interpretation(results: dict) -> str:
    """Genera un párrafo de interpretación académica basado en las métricas."""
    glob    = results.get("global", {})
    mean_acc = glob.get("mean_accuracy", 0)
    mean_f1  = glob.get("mean_f1_macro", 0)

    if mean_acc >= 0.90:
        nivel = "excelente"
    elif mean_acc >= 0.75:
        nivel = "satisfactorio"
    elif mean_acc >= 0.60:
        nivel = "moderado"
    else:
        nivel = "bajo, lo que sugiere que el modelo requiere mayor entrenamiento o ajuste de hiperparámetros"

    accs      = {t: results[t]["accuracy"] for t in _TASKS}
    best_task  = max(accs, key=accs.get)
    worst_task = min(accs, key=accs.get)

    return (
        f"El modelo MultiTaskDistilBERT alcanzó un desempeño **{nivel}** sobre el conjunto de prueba, "
        f"con una accuracy media de **{mean_acc*100:.1f}%** y un F1 macro medio de **{mean_f1*100:.1f}%** "
        f"considerando las cuatro tareas de clasificación simultáneas.\n\n"
        f"La tarea con **mejor desempeño** fue `{best_task}` "
        f"(accuracy = {accs[best_task]*100:.1f}%), lo que indica que el modelo captura adecuadamente "
        f"los patrones lingüísticos asociados a este atributo de diseño SVG. "
        f"La tarea con **menor desempeño** fue `{worst_task}` "
        f"(accuracy = {accs[worst_task]*100:.1f}%), lo que puede atribuirse a una mayor ambigüedad "
        f"léxica en los prompts o a la mayor cardinalidad de clases en esta dimensión.\n\n"
        f"Las matrices de confusión por tarea permiten identificar las clases con mayor tasa de error "
        f"e informar decisiones de mejora en fases futuras del proyecto."
    )


class _NumpyEncoder(json.JSONEncoder):
    """Encoder JSON que maneja tipos de NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
