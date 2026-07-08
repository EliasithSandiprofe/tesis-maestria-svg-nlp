"""
training_utils.py
-----------------
Utilidades del pipeline de entrenamiento (Fase D): reproducibilidad,
gestión de checkpoints, métricas, I/O y presentación en consola.

Funciones públicas
------------------
set_seed                 : Fija la semilla en random, numpy, torch y CUDA.
create_directories       : Crea los directorios necesarios si no existen.
save_checkpoint          : Guarda un checkpoint con torch.save.
load_checkpoint          : Carga un checkpoint con torch.load.
save_metrics_csv         : Escribe el historial de métricas a CSV.
save_config_json         : Serializa la configuración de entrenamiento a JSON.
move_batch_to_device     : Mueve todos los tensores de un batch al dispositivo.
format_time              : Formatea segundos como HH:MM:SS.
print_epoch_summary      : Imprime el resumen de una época.
print_training_header    : Imprime el encabezado del entrenamiento.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ── Reproducibilidad ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fija todas las semillas aleatorias para reproducibilidad completa.

    Afecta a: módulo ``random`` de Python, NumPy, PyTorch (CPU y CUDA).

    Parameters
    ----------
    seed : int
        Valor de la semilla.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Directorios ───────────────────────────────────────────────────────────────

def create_directories(*dirs: str | Path) -> None:
    """Crea los directorios indicados si no existen.

    Parameters
    ----------
    *dirs : str | Path
        Rutas de los directorios a crear.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── Checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str | Path) -> None:
    """Guarda un checkpoint de PyTorch en disco.

    Parameters
    ----------
    state : dict
        Diccionario con ``model_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict``, ``epoch``, ``val_loss``, ``val_metrics``,
        ``training_config`` y ``num_classes``.
    path : str | Path
        Ruta del archivo ``.pt`` de salida.
    """
    torch.save(state, Path(path))


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict:
    """Carga un checkpoint desde disco.

    Parameters
    ----------
    path : str | Path
        Ruta del archivo ``.pt``.
    map_location : str | torch.device | None
        Dispositivo destino de la carga (p. ej. ``"cpu"`` cuando se carga
        un modelo entrenado en GPU en una máquina sin GPU).

    Returns
    -------
    dict
        Contenido del checkpoint.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


# ── Persistencia de métricas y configuración ──────────────────────────────────

def save_metrics_csv(history: list[dict], path: str | Path) -> None:
    """Guarda el historial de métricas por época en un archivo CSV.

    Parameters
    ----------
    history : list[dict]
        Lista de diccionarios, uno por época, con todas las métricas.
    path : str | Path
        Ruta del archivo CSV de salida.
    """
    pd.DataFrame(history).to_csv(Path(path), index=False, encoding="utf-8")


def save_config_json(config_dict: dict, path: str | Path) -> None:
    """Serializa la configuración de entrenamiento en JSON.

    Convierte automáticamente objetos no serializables (``Path``,
    ``torch.device``) a cadenas de texto.

    Parameters
    ----------
    config_dict : dict
        Diccionario con la configuración (resultado de ``config.to_dict()``).
    path : str | Path
        Ruta del archivo JSON de salida.
    """
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, torch.device):
                return str(obj)
            return super().default(obj)

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_dict,
    }
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=_Encoder, ensure_ascii=False)


# ── Batch ─────────────────────────────────────────────────────────────────────

def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mueve todos los tensores de un batch al dispositivo indicado.

    Parameters
    ----------
    batch : dict[str, torch.Tensor]
        Batch de datos tal como lo devuelve ``SVGPromptDataset.__getitem__``.
    device : torch.device
        Dispositivo destino (CPU o CUDA).

    Returns
    -------
    dict[str, torch.Tensor]
        Mismo diccionario con todos los tensores en ``device``.
    """
    return {key: val.to(device) for key, val in batch.items()}


# ── Formato de tiempo ─────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convierte segundos a formato ``HH:MM:SS``.

    Parameters
    ----------
    seconds : float
        Tiempo en segundos.

    Returns
    -------
    str
        Cadena con formato ``HH:MM:SS``.
    """
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Presentación en consola ───────────────────────────────────────────────────

def print_training_header(
    config,
    n_train: int,
    n_val: int,
) -> None:
    """Imprime el encabezado del entrenamiento antes del primer epoch.

    Parameters
    ----------
    config : TrainingConfig
        Configuración del entrenamiento.
    n_train : int
        Número de muestras de entrenamiento.
    n_val : int
        Número de muestras de validación.
    """
    sep = "═" * 62
    print()
    print(f"  {sep}")
    print(f"  ENTRENAMIENTO — MultiTaskDistilBERT SVG-NLP")
    print(f"  {sep}")
    print(f"  Dispositivo          : {str(config.device).upper()}")
    print(f"  Épocas               : {config.epochs}")
    print(f"  Batch size           : {config.batch_size}")
    print(f"  Learning rate        : {config.learning_rate}")
    print(f"  Weight decay         : {config.weight_decay}")
    print(f"  Gradient clipping    : {config.max_grad_norm}")
    print(f"  Warmup ratio         : {config.warmup_ratio * 100:.0f}%")
    print(f"  Early stopping pat.  : {config.patience}")
    print(f"  Random seed          : {config.random_seed}")
    print(f"  Train samples        : {n_train:,}")
    print(f"  Val   samples        : {n_val:,}")
    print(f"  {sep}")
    print()


def print_epoch_summary(
    epoch: int,
    n_epochs: int,
    metrics: dict,
    elapsed: float,
    is_best: bool,
) -> None:
    """Imprime el resumen de una época de entrenamiento.

    Parameters
    ----------
    epoch : int
        Número de la época actual (1-based).
    n_epochs : int
        Número total de épocas planificadas.
    metrics : dict
        Diccionario con todas las métricas de la época.
    elapsed : float
        Tiempo transcurrido en la época (segundos).
    is_best : bool
        ``True`` si esta época tiene el mejor ``val_loss`` hasta ahora.
    """
    time_str = format_time(elapsed)
    best_tag = "  ★ MEJOR MODELO" if is_best else ""

    # ── Extracción de métricas ────────────────────────────────────────────────
    tl  = metrics["train_loss"]
    tc  = metrics["train_loss_color"]
    te  = metrics["train_loss_estilo"]
    tel = metrics["train_loss_elemento"]
    tp  = metrics["train_loss_posicion"]

    vl  = metrics["val_loss"]
    vc  = metrics["val_loss_color"]
    ve  = metrics["val_loss_estilo"]
    vel = metrics["val_loss_elemento"]
    vp  = metrics["val_loss_posicion"]

    ac  = metrics["accuracy_color"]    * 100
    ae  = metrics["accuracy_estilo"]   * 100
    ael = metrics["accuracy_elemento"] * 100
    ap  = metrics["accuracy_posicion"] * 100
    ma  = metrics["mean_accuracy"]     * 100

    # ── Impresión ─────────────────────────────────────────────────────────────
    bar = "━" * 62
    print()
    print(f"  {bar}")
    print(f"  Época {epoch}/{n_epochs}  │  {time_str}{best_tag}")
    print(f"  {'─' * 62}")
    print(
        f"  Train loss : {tl:.4f}"
        f"  (col={tc:.4f} | est={te:.4f} | ele={tel:.4f} | pos={tp:.4f})"
    )
    print(
        f"  Valid loss : {vl:.4f}"
        f"  (col={vc:.4f} | est={ve:.4f} | ele={vel:.4f} | pos={vp:.4f})"
    )
    print(
        f"  Accuracy   : color={ac:5.1f}% | estilo={ae:5.1f}%"
        f" | elem={ael:5.1f}% | pos={ap:5.1f}%"
    )
    print(f"  Mean Acc   : {ma:.1f}%")
    print(f"  {bar}")
