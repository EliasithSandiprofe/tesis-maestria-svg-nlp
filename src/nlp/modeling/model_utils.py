"""
model_utils.py
--------------
Utilidades para la gestión del modelo MultiTaskDistilBERT: carga de
encoders, obtención automática de clases, creación de instancias e
inspección de parámetros.

Funciones públicas
------------------
load_label_encoders(encoders_dir)     -> dict[str, LabelEncoder]
get_num_classes(encoders_dir)         -> dict[str, int]
build_model(config)                   -> MultiTaskDistilBERT
count_parameters(model)               -> tuple[int, int]
print_model_summary(model, config)    -> None
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

from .model_config import SVGModelConfig, TASK_NAMES
from .multitask_distilbert import MultiTaskDistilBERT


def load_label_encoders(
    encoders_dir: str | Path,
) -> dict[str, LabelEncoder]:
    """Carga los cuatro ``LabelEncoder`` serializados desde disco.

    Parameters
    ----------
    encoders_dir : str | Path
        Directorio que contiene ``<tarea>_encoder.pkl`` para cada tarea.

    Returns
    -------
    dict[str, LabelEncoder]
        Diccionario ``{nombre_tarea: LabelEncoder}``.

    Raises
    ------
    FileNotFoundError
        Si algún archivo ``.pkl`` no existe.
    """
    encoders_dir = Path(encoders_dir)
    encoders: dict[str, LabelEncoder] = {}
    for task in TASK_NAMES:
        path = encoders_dir / f"{task}_encoder.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Encoder no encontrado: {path}")
        encoders[task] = joblib.load(path)
    return encoders


def get_num_classes(encoders_dir: str | Path) -> dict[str, int]:
    """Devuelve el número de clases por tarea leyendo los encoders.

    Parameters
    ----------
    encoders_dir : str | Path
        Directorio con los archivos ``.pkl``.

    Returns
    -------
    dict[str, int]
        Diccionario ``{nombre_tarea: número_de_clases}``.
    """
    encoders = load_label_encoders(encoders_dir)
    return {task: int(len(le.classes_)) for task, le in encoders.items()}


def build_model(config: SVGModelConfig) -> MultiTaskDistilBERT:
    """Crea e inicializa una instancia de ``MultiTaskDistilBERT``.

    Descarga los pesos de DistilBERT desde Hugging Face Hub en la primera
    ejecución; las siguientes los cargan desde la caché local.

    Parameters
    ----------
    config : SVGModelConfig
        Configuración del modelo.

    Returns
    -------
    MultiTaskDistilBERT
        Modelo instanciado con los pesos preentrenados de DistilBERT.
    """
    return MultiTaskDistilBERT(config)


def count_parameters(model: MultiTaskDistilBERT) -> tuple[int, int]:
    """Cuenta los parámetros totales y entrenables del modelo.

    Parameters
    ----------
    model : MultiTaskDistilBERT
        Instancia del modelo.

    Returns
    -------
    tuple[int, int]
        ``(total_params, trainable_params)``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(
    model: MultiTaskDistilBERT,
    config: SVGModelConfig,
) -> None:
    """Imprime un resumen del modelo en consola.

    Muestra: modelo base, dropout, número de clases por tarea,
    parámetros totales, entrenables y congelados.

    Parameters
    ----------
    model : MultiTaskDistilBERT
        Instancia del modelo.
    config : SVGModelConfig
        Configuración asociada al modelo.
    """
    total, trainable = count_parameters(model)
    frozen = total - trainable

    sep = "=" * 60
    print(sep)
    print("  RESUMEN DEL MODELO — MultiTaskDistilBERT")
    print(sep)
    print(f"  Encoder base    : {config.model_name}")
    print(f"  Dropout         : {config.dropout}")
    print()
    print("  Cabezas de clasificación:")
    for task, n_cls in config.num_classes.items():
        hidden = model.distilbert.config.hidden_size
        print(f"    • {task:<12}  Linear({hidden}, {n_cls})")
    print()
    print(f"  Parámetros totales      : {total:>12,}")
    print(f"  Parámetros entrenables  : {trainable:>12,}")
    print(f"  Parámetros congelados   : {frozen:>12,}")
    print(sep)
    print()
