"""
Evaluation Package — Fase E: Evaluación final del modelo Multi-Head DistilBERT.

Módulos
-------
evaluator         : Clase ``MultiTaskEvaluator`` — inferencia y métricas sklearn.
evaluation_utils  : Carga de checkpoints, serialización de resultados,
                    matrices de confusión y reporte Markdown.
evaluate          : Script principal (entrypoint para Colab y local).
"""

from .evaluator import MultiTaskEvaluator
from .evaluation_utils import (
    load_model_from_checkpoint,
    load_label_encoders_for_eval,
    save_metrics_json,
    save_classification_report_md,
    plot_confusion_matrix,
    generate_evaluation_report,
)

__all__ = [
    "MultiTaskEvaluator",
    "load_model_from_checkpoint",
    "load_label_encoders_for_eval",
    "save_metrics_json",
    "save_classification_report_md",
    "plot_confusion_matrix",
    "generate_evaluation_report",
]
