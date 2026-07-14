"""
Preprocessing Package — Fase B: Preparación del Dataset SVG-NLP.

Módulos
-------
clean_dataset    : Deduplicación del dataset original.
label_encoder    : Codificación de etiquetas con sklearn LabelEncoder.
split_dataset    : División en Train / Validation / Test.
tokenizer        : Tokenización con DistilBertTokenizerFast.
dataset_builder  : Clase Dataset compatible con PyTorch.
run_preprocessing: Orquestador del flujo completo.
"""

from .clean_dataset import clean_dataset
from .label_encoder import encode_labels
from .split_dataset import split_dataset
from .tokenizer import load_and_save_tokenizer, tokenize_prompts
from .dataset_builder import SVGPromptDataset, build_datasets

__all__ = [
    "clean_dataset",
    "encode_labels",
    "split_dataset",
    "load_and_save_tokenizer",
    "tokenize_prompts",
    "SVGPromptDataset",
    "build_datasets",
]
