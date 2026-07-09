"""
inference.py
------------
Módulo de inferencia de producción para el modelo MultiTaskDistilBERT SVG-NLP.

Expone una única función pública:

    predict_attributes(prompt: str) -> dict

que recibe un prompt en español y devuelve los cuatro atributos SVG
(color, estilo, elemento, posicion) decodificados como strings.

El modelo, el tokenizer y los LabelEncoders se cargan una sola vez al primer
llamado (patrón lazy-singleton) y se reutilizan en todas las peticiones
posteriores.  Esto lo hace seguro para Flask: el modelo vive en memoria
durante toda la vida del proceso sin recargarse.

Estructura de la respuesta
--------------------------
    {
        "prompt": "<texto original>",
        "atributos": {
            "color":    "<valor predicho>",
            "estilo":   "<valor predicho>",
            "elemento": "<valor predicho>",
            "posicion": "<valor predicho>"
        }
    }

Archivos requeridos (no incluidos en el repositorio por tamaño)
----------------------------------------------------------------
    modelos/checkpoints/best_model.pt       (~780 MB)
    modelos/label_encoders/color_encoder.pkl
    modelos/label_encoders/estilo_encoder.pkl
    modelos/label_encoders/elemento_encoder.pkl
    modelos/label_encoders/posicion_encoder.pkl
    modelos/tokenizer/                      (vocab.txt, tokenizer.json, …)

Uso desde Flask
---------------
    from src.nlp.inference import predict_attributes

    @app.route("/predict", methods=["POST"])
    def predict():
        prompt = request.json.get("prompt")
        return jsonify(predict_attributes(prompt))

Uso directo
-----------
    python src/nlp/inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast

# ── Ajuste del PYTHONPATH ─────────────────────────────────────────────────────
# Ubicación de este archivo:  src/nlp/inference.py
#   parents[0] → src/nlp/
#   parents[1] → src/
#   parents[2] → project root  (contiene modelos/, dataset/, src/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Estas importaciones requieren que _PROJECT_ROOT esté en sys.path
from src.nlp.evaluation.evaluation_utils import load_model_from_checkpoint
from src.nlp.modeling.model_utils import load_label_encoders

# ── Rutas canónicas ───────────────────────────────────────────────────────────
_CHECKPOINT_PATH = _PROJECT_ROOT / "modelos" / "checkpoints" / "best_model.pt"
_ENCODERS_DIR    = _PROJECT_ROOT / "modelos" / "label_encoders"
_TOKENIZER_DIR   = _PROJECT_ROOT / "modelos" / "tokenizer"

_TASKS      = ["color", "estilo", "elemento", "posicion"]
_MAX_LENGTH = 128

# ── Estado del singleton ──────────────────────────────────────────────────────
_model:     Optional[torch.nn.Module]           = None
_tokenizer: Optional[DistilBertTokenizerFast]   = None
_encoders:  Optional[dict[str, LabelEncoder]]   = None
_device:    Optional[torch.device]              = None


# ── Inicialización lazy ───────────────────────────────────────────────────────

def _load_engine() -> None:
    """Carga el modelo, el tokenizer y los LabelEncoders una sola vez.

    Utiliza el patrón lazy-singleton: si ya están cargados, retorna
    inmediatamente sin repetir la carga.  No es thread-safe a propósito
    para mantener el código mínimo; en Flask el worker carga el motor
    durante la primera petición y luego reutiliza el estado.

    Raises
    ------
    FileNotFoundError
        Si ``best_model.pt`` no existe en ``modelos/checkpoints/``.
    """
    global _model, _tokenizer, _encoders, _device

    if _model is not None:
        return  # Motor ya inicializado; nada que hacer

    # ── Verificar existencia del checkpoint ───────────────────────────────────
    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "\n"
            + "=" * 62 + "\n"
            "  ERROR: Modelo entrenado no encontrado.\n\n"
            f"  Ruta esperada:\n"
            f"    {_CHECKPOINT_PATH}\n\n"
            "  best_model.pt no está en el repositorio porque pesa ~780 MB.\n\n"
            "  Para continuar:\n"
            "    1. Descarga best_model.pt desde el almacenamiento compartido\n"
            "       del proyecto (Google Drive / servidor de artefactos).\n"
            "    2. Colócalo en:  modelos/checkpoints/best_model.pt\n"
            "    3. Vuelve a ejecutar.\n"
            + "=" * 62 + "\n"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Dispositivo: {_device}")

    # ── Cargar modelo (reutiliza load_model_from_checkpoint de evaluation) ────
    print("[inference] Cargando modelo desde checkpoint...", end=" ", flush=True)
    _model, ckpt_info = load_model_from_checkpoint(
        checkpoint_path=_CHECKPOINT_PATH,
        project_root=_PROJECT_ROOT,
        device=_device,
    )
    # load_model_from_checkpoint ya llama model.eval() internamente
    print(f"OK  (época {ckpt_info.get('epoch', '?')}  |  "
          f"val_loss {ckpt_info.get('val_loss', float('nan')):.4f})")

    # ── Cargar tokenizer desde disco (sin acceso a Hugging Face Hub) ──────────
    print("[inference] Cargando tokenizer...", end=" ", flush=True)
    _tokenizer = DistilBertTokenizerFast.from_pretrained(str(_TOKENIZER_DIR))
    print("OK")

    # ── Cargar LabelEncoders ─────────────────────────────────────────────────
    print("[inference] Cargando LabelEncoders...", end=" ", flush=True)
    _encoders = load_label_encoders(encoders_dir=_ENCODERS_DIR)
    print("OK")

    print("[inference] Motor de inferencia listo.\n")


# ── Función pública ───────────────────────────────────────────────────────────

def predict_attributes(prompt: str) -> dict:
    """Predice los cuatro atributos SVG a partir de un prompt en español.

    Flujo interno:
    1. Inicialización lazy del modelo, tokenizer y encoders (solo 1.ª vez).
    2. Tokenización con ``DistilBertTokenizerFast`` (max_length=128).
    3. Forward pass sin gradientes sobre ``MultiTaskDistilBERT``.
    4. Argmax sobre los logits de cada cabeza → índice de clase.
    5. Decodificación con ``LabelEncoder.inverse_transform`` → string.

    Parameters
    ----------
    prompt : str
        Texto libre en español que describe el diseño SVG.
        Ejemplo: ``"Diseña una camiseta vintage roja con una guitarra centrada"``

    Returns
    -------
    dict
        Diccionario con la siguiente estructura::

            {
                "prompt": "<texto original>",
                "atributos": {
                    "color":    "<valor predicho>",
                    "estilo":   "<valor predicho>",
                    "elemento": "<valor predicho>",
                    "posicion": "<valor predicho>"
                }
            }

    Raises
    ------
    ValueError
        Si el prompt está vacío o contiene solo espacios en blanco.
    FileNotFoundError
        Si ``best_model.pt`` no existe en ``modelos/checkpoints/``.
    """
    if not prompt or not prompt.strip():
        raise ValueError("El prompt no puede estar vacío.")

    # Inicialización lazy: solo en la primera llamada
    _load_engine()

    # ── Tokenización ──────────────────────────────────────────────────────────
    encoding = _tokenizer(
        prompt,
        max_length=_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(_device)       # (1, 128)
    attention_mask = encoding["attention_mask"].to(_device)  # (1, 128)

    # ── Forward pass (sin gradientes, modelo ya en eval()) ───────────────────
    with torch.no_grad():
        outputs = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # No se pasan etiquetas → loss = None
        )

    # ── Decodificación: logits → índice → etiqueta string ────────────────────
    atributos: dict[str, str] = {}
    for task in _TASKS:
        logits = outputs[f"logits_{task}"]                    # (1, num_classes)
        idx    = int(torch.argmax(logits, dim=1).item())
        label  = str(_encoders[task].inverse_transform([idx])[0])
        atributos[task] = label

    return {
        "prompt":    prompt,
        "atributos": atributos,
    }


# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    resultado = predict_attributes(
        "Diseña una camiseta vintage roja con una guitarra centrada"
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
