"""
tokenizer.py
------------
Fase 4 del preprocesamiento: tokenización de los prompts mediante
``DistilBertTokenizerFast`` del modelo ``distilbert-base-uncased``.

ALCANCE
-------
Este módulo únicamente carga el tokenizer preentrenado, lo aplica sobre
los prompts y serializa el tokenizer en disco para uso posterior.
NO se crea ni modifica ningún modelo DistilBERT. No se realiza
entrenamiento alguno.

Funciones públicas
------------------
load_and_save_tokenizer(model_name, save_dir) -> DistilBertTokenizerFast
    Carga el tokenizer desde Hugging Face y lo guarda en disco.

tokenize_prompts(prompts, tokenizer, max_length) -> dict[str, np.ndarray]
    Tokeniza una lista de prompts y devuelve ``input_ids`` y
    ``attention_mask`` como arrays de NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast


def load_and_save_tokenizer(
    model_name: str,
    save_dir: str | Path,
) -> DistilBertTokenizerFast:
    """Carga el tokenizer preentrenado y lo serializa en disco.

    En la primera ejecución descarga los archivos del vocabulario desde
    Hugging Face Hub (~270 KB). Las ejecuciones posteriores los cargan desde
    la caché local.

    Parameters
    ----------
    model_name : str
        Identificador del modelo en Hugging Face (p. ej.
        ``"distilbert-base-uncased"``).
    save_dir : str | Path
        Directorio donde serializar el tokenizer para uso offline.

    Returns
    -------
    DistilBertTokenizerFast
        Instancia del tokenizer lista para usar.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Cargando tokenizer '{model_name}' desde Hugging Face Hub...")
    tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(model_name)

    tokenizer.save_pretrained(str(save_dir))
    print(f"  Tokenizer serializado en: {save_dir}")

    return tokenizer


def tokenize_prompts(
    prompts: list[str] | pd.Series,
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 128,
) -> dict[str, np.ndarray]:
    """Tokeniza una colección de prompts.

    Aplica truncado y relleno (padding) hasta ``max_length`` tokens.
    Los prompts del dataset tienen entre 11 y 14 palabras (64–100 caracteres),
    lo que produce entre 15 y 25 sub-tokens aproximadamente, muy por debajo
    del límite de 512 de DistilBERT. El valor ``max_length=128`` es
    suficientemente holgado y reduce el consumo de memoria con respecto al
    máximo del modelo.

    Parameters
    ----------
    prompts : list[str] | pd.Series
        Colección de prompts de texto.
    tokenizer : DistilBertTokenizerFast
        Instancia del tokenizer preentrenado.
    max_length : int
        Longitud máxima de la secuencia tokenizada (por defecto 128).

    Returns
    -------
    dict[str, np.ndarray]
        Diccionario con las claves ``input_ids`` y ``attention_mask``,
        ambas de forma ``(n_prompts, max_length)``.
    """
    encoding = tokenizer(
        list(prompts),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return {
        "input_ids":      np.array(encoding["input_ids"],      dtype=np.int64),
        "attention_mask": np.array(encoding["attention_mask"], dtype=np.int64),
    }


def compute_token_stats(
    prompts: list[str] | pd.Series,
    tokenizer: DistilBertTokenizerFast,
) -> dict:
    """Calcula estadísticas de longitud de tokens para los prompts dados.

    Útil para verificar que ``max_length`` es suficiente y cuantificar
    el porcentaje de truncado.

    Parameters
    ----------
    prompts : list[str] | pd.Series
        Colección de prompts.
    tokenizer : DistilBertTokenizerFast
        Instancia del tokenizer.

    Returns
    -------
    dict
        Estadísticas: ``min``, ``max``, ``mean``, ``median``, ``p95``.
    """
    lengths = [
        len(tokenizer.encode(p, add_special_tokens=True))
        for p in list(prompts)
    ]
    arr = np.array(lengths)
    return {
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "mean":   round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95":    int(np.percentile(arr, 95)),
    }
