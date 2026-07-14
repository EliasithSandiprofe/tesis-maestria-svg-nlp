"""
Modeling Package — Fase C: Construcción del modelo Multi-Head DistilBERT.

Módulos
-------
model_config         : Configuración centralizada del modelo (carga automática
                       del número de clases desde los LabelEncoder serializados).
multitask_distilbert : Clase ``MultiTaskDistilBERT`` (PyTorch nn.Module).
model_utils          : Utilidades de carga, inspección y conteo de parámetros.
test_model_forward   : Script de validación del forward pass con batch real.
"""

from .model_config import SVGModelConfig
from .multitask_distilbert import MultiTaskDistilBERT
from .model_utils import (
    load_label_encoders,
    get_num_classes,
    build_model,
    count_parameters,
    print_model_summary,
)

__all__ = [
    "SVGModelConfig",
    "MultiTaskDistilBERT",
    "load_label_encoders",
    "get_num_classes",
    "build_model",
    "count_parameters",
    "print_model_summary",
]
