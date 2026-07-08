"""
EDA Package — Análisis Exploratorio de Datos para el dataset SVG-NLP.

Módulos
-------
dataset_loader    : Carga y validación del CSV.
quality_analysis  : Análisis de calidad (nulos, duplicados, únicos).
label_analysis    : Distribución y estadísticas por etiqueta.
prompt_analysis   : Métricas de longitud de los prompts.
visualizations    : Gráficas con matplotlib.
report_generator  : Reporte automático en Markdown.
run_eda           : Punto de entrada que ejecuta todo el flujo.
"""

from .dataset_loader import load_dataset
from .quality_analysis import analyze_quality
from .label_analysis import analyze_labels
from .prompt_analysis import analyze_prompts
from .visualizations import generate_visualizations
from .report_generator import generate_report

__all__ = [
    "load_dataset",
    "analyze_quality",
    "analyze_labels",
    "analyze_prompts",
    "generate_visualizations",
    "generate_report",
]
