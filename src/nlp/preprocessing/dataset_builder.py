"""
dataset_builder.py
------------------
Fase 5 del preprocesamiento: construcción de los objetos Dataset compatibles
con PyTorch a partir de los conjuntos tokenizados y las etiquetas codificadas.

La clase ``SVGPromptDataset`` es reutilizable directamente con
``torch.utils.data.DataLoader`` en la Fase C de entrenamiento.

Estructura de cada muestra (``__getitem__``)
--------------------------------------------
::

    {
        "input_ids":       LongTensor  (max_length,)
        "attention_mask":  LongTensor  (max_length,)
        "labels_color":    LongTensor  (escalar)
        "labels_estilo":   LongTensor  (escalar)
        "labels_elemento": LongTensor  (escalar)
        "labels_posicion": LongTensor  (escalar)
    }

Funciones públicas
------------------
build_datasets(train_df, val_df, test_df, tokenizer, max_length)
    -> tuple[SVGPromptDataset, SVGPromptDataset, SVGPromptDataset]
    Tokeniza los prompts de cada split y construye los tres Dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import tokenize_prompts


class SVGPromptDataset(Dataset):
    """Dataset PyTorch para la tarea de clasificación multi-head SVG-NLP.

    Almacena los tensores precomputados (``input_ids``, ``attention_mask``
    y las cuatro etiquetas) y los expone por índice a través de
    ``__getitem__``.

    Parameters
    ----------
    input_ids : np.ndarray
        Array de forma ``(n, max_length)`` con los identificadores de token.
    attention_mask : np.ndarray
        Array de forma ``(n, max_length)`` con la máscara de atención.
    labels_color : np.ndarray
        Array 1-D con los códigos enteros de la etiqueta ``color``.
    labels_estilo : np.ndarray
        Array 1-D con los códigos enteros de la etiqueta ``estilo``.
    labels_elemento : np.ndarray
        Array 1-D con los códigos enteros de la etiqueta ``elemento``.
    labels_posicion : np.ndarray
        Array 1-D con los códigos enteros de la etiqueta ``posicion``.
    """

    def __init__(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels_color: np.ndarray,
        labels_estilo: np.ndarray,
        labels_elemento: np.ndarray,
        labels_posicion: np.ndarray,
    ) -> None:
        self._input_ids      = torch.tensor(input_ids,       dtype=torch.long)
        self._attention_mask = torch.tensor(attention_mask,  dtype=torch.long)
        self._labels_color    = torch.tensor(labels_color,   dtype=torch.long)
        self._labels_estilo   = torch.tensor(labels_estilo,  dtype=torch.long)
        self._labels_elemento = torch.tensor(labels_elemento,dtype=torch.long)
        self._labels_posicion = torch.tensor(labels_posicion,dtype=torch.long)

    def __len__(self) -> int:
        """Devuelve el número de muestras del conjunto."""
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Devuelve la muestra en la posición ``idx`` como un diccionario.

        Parameters
        ----------
        idx : int
            Índice de la muestra (0-based).

        Returns
        -------
        dict[str, torch.Tensor]
            Diccionario con las claves ``input_ids``, ``attention_mask``,
            ``labels_color``, ``labels_estilo``, ``labels_elemento``,
            ``labels_posicion``.
        """
        return {
            "input_ids":       self._input_ids[idx],
            "attention_mask":  self._attention_mask[idx],
            "labels_color":    self._labels_color[idx],
            "labels_estilo":   self._labels_estilo[idx],
            "labels_elemento": self._labels_elemento[idx],
            "labels_posicion": self._labels_posicion[idx],
        }

    @property
    def input_shape(self) -> tuple[int, int]:
        """Forma de los tensores de entrada: ``(n_muestras, max_length)``."""
        return tuple(self._input_ids.shape)


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int = 128,
) -> tuple[SVGPromptDataset, SVGPromptDataset, SVGPromptDataset]:
    """Construye los tres SVGPromptDataset tokenizando cada conjunto.

    Tokeniza los prompts de cada split mediante ``tokenize_prompts`` y
    extrae las etiquetas codificadas (columnas ``*_enc``) del DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        Conjunto de entrenamiento (salida de ``split_dataset``).
    val_df : pd.DataFrame
        Conjunto de validación.
    test_df : pd.DataFrame
        Conjunto de prueba.
    tokenizer : DistilBertTokenizerFast
        Tokenizer serializado y listo para usar.
    max_length : int
        Longitud máxima de la secuencia (por defecto 128).

    Returns
    -------
    tuple[SVGPromptDataset, SVGPromptDataset, SVGPromptDataset]
        ``(train_dataset, val_dataset, test_dataset)``
    """
    sep = "=" * 60
    print(sep)
    print("  FASE 5 — CONSTRUCCIÓN DE DATASETS PYTORCH")
    print(sep)

    split_names = ["Train", "Validation", "Test"]
    datasets = []

    for name, df in zip(split_names, [train_df, val_df, test_df]):
        print(f"  Tokenizando {name} ({len(df):,} muestras)...", end=" ", flush=True)

        tokens = tokenize_prompts(df["prompt"], tokenizer, max_length=max_length)

        ds = SVGPromptDataset(
            input_ids       = tokens["input_ids"],
            attention_mask  = tokens["attention_mask"],
            labels_color    = df["color_enc"].values,
            labels_estilo   = df["estilo_enc"].values,
            labels_elemento = df["elemento_enc"].values,
            labels_posicion = df["posicion_enc"].values,
        )
        datasets.append(ds)
        print("OK")

    train_ds, val_ds, test_ds = datasets

    # Verificar una muestra de ejemplo
    sample = train_ds[0]
    print()
    print("  Estructura de una muestra (train_ds[0]):")
    for key, tensor in sample.items():
        print(f"    {key:<20} shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print(sep)
    print()

    return train_ds, val_ds, test_ds
