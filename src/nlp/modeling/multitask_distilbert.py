"""
multitask_distilbert.py
-----------------------
Arquitectura Multi-Head DistilBERT para clasificación multi-tarea SVG-NLP.

DISEÑO DE LA ARQUITECTURA
--------------------------
El modelo sigue el patrón de *hard parameter sharing* (compartición de
parámetros rígida):

    ┌─────────────────────────────────────────────────────┐
    │  Encoder compartido: DistilBertModel                │
    │  (66 M parámetros, pesos preentrenados)             │
    └───────────────────────────┬─────────────────────────┘
                                │ [CLS] embedding (768-d)
                           Dropout(p)
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼               ▼
        color_head       estilo_head     elemento_head   posicion_head
        Linear(768, 6)  Linear(768, 5)  Linear(768,10)  Linear(768, 4)

Ventajas de este enfoque para la tesis:
- El encoder aprende representaciones compartidas útiles para las cuatro
  tareas simultáneamente (transferencia de aprendizaje multi-tarea).
- Cada cabeza se especializa en predecir un único atributo SVG.
- La pérdida total es la suma de las cuatro pérdidas individuales, lo que
  produce señales de gradiente balanceadas hacia el encoder.

Clases
------
MultiTaskDistilBERT(nn.Module)
    Modelo completo listo para entrenamiento con PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertModel

from .model_config import SVGModelConfig


class MultiTaskDistilBERT(nn.Module):
    """Modelo de clasificación multi-cabeza basado en DistilBERT.

    Utiliza ``DistilBertModel`` como encoder compartido y cuatro cabezas
    lineales independientes para predecir ``color``, ``estilo``,
    ``elemento`` y ``posicion``.

    Parameters
    ----------
    config : SVGModelConfig
        Configuración del modelo.  Debe contener ``model_name``,
        ``dropout`` y ``num_classes`` (dict con las cuatro tareas).

    Examples
    --------
    >>> config = SVGModelConfig.from_project(project_root)
    >>> model  = MultiTaskDistilBERT(config)
    >>> out    = model(input_ids, attention_mask,
    ...               labels_color=lc, labels_estilo=le,
    ...               labels_elemento=lel, labels_posicion=lp)
    >>> out["loss"]          # pérdida total (escalar)
    >>> out["logits_color"]  # (batch, 6)
    """

    def __init__(self, config: SVGModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Encoder compartido ────────────────────────────────────────────────
        self.distilbert: DistilBertModel = DistilBertModel.from_pretrained(
            config.model_name
        )
        hidden_size: int = self.distilbert.config.hidden_size  # 768

        # ── Regularización ────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=config.dropout)

        # ── Función de pérdida (compartida entre cabezas) ─────────────────────
        self._loss_fn = nn.CrossEntropyLoss()

        # ── Cabezas de clasificación independientes ───────────────────────────
        self.color_classifier    = nn.Linear(hidden_size, config.num_classes["color"])
        self.estilo_classifier   = nn.Linear(hidden_size, config.num_classes["estilo"])
        self.elemento_classifier = nn.Linear(hidden_size, config.num_classes["elemento"])
        self.posicion_classifier = nn.Linear(hidden_size, config.num_classes["posicion"])

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_color: torch.Tensor | None = None,
        labels_estilo: torch.Tensor | None = None,
        labels_elemento: torch.Tensor | None = None,
        labels_posicion: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Ejecuta el forward pass del modelo.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor de forma ``(batch, seq_len)`` con los IDs de tokens.
        attention_mask : torch.Tensor
            Tensor de forma ``(batch, seq_len)`` con la máscara de atención.
        labels_color : torch.Tensor | None
            Etiquetas enteras de color, forma ``(batch,)``.  Si se omite,
            ``loss`` es ``None``.
        labels_estilo : torch.Tensor | None
            Etiquetas enteras de estilo, forma ``(batch,)``.
        labels_elemento : torch.Tensor | None
            Etiquetas enteras de elemento, forma ``(batch,)``.
        labels_posicion : torch.Tensor | None
            Etiquetas enteras de posicion, forma ``(batch,)``.

        Returns
        -------
        dict
            Diccionario con las siguientes claves:

            - ``logits_color``    : ``(batch, num_classes_color)``
            - ``logits_estilo``   : ``(batch, num_classes_estilo)``
            - ``logits_elemento`` : ``(batch, num_classes_elemento)``
            - ``logits_posicion`` : ``(batch, num_classes_posicion)``
            - ``loss``            : escalar (suma de las cuatro pérdidas) o ``None``
            - ``loss_color``      : escalar (solo si se proporcionan etiquetas)
            - ``loss_estilo``     : escalar
            - ``loss_elemento``   : escalar
            - ``loss_posicion``   : escalar
        """
        # ── 1. Encoder DistilBERT ─────────────────────────────────────────────
        encoder_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # last_hidden_state: (batch, seq_len, hidden_size)
        # El primer token [CLS] es la representación global del prompt.
        hidden_state = encoder_output.last_hidden_state  # (batch, seq_len, 768)
        cls_output   = hidden_state[:, 0, :]              # (batch, 768)
        cls_output   = self.dropout(cls_output)

        # ── 2. Logits por cabeza ──────────────────────────────────────────────
        logits_color    = self.color_classifier(cls_output)      # (batch, 6)
        logits_estilo   = self.estilo_classifier(cls_output)     # (batch, 5)
        logits_elemento = self.elemento_classifier(cls_output)   # (batch, 10)
        logits_posicion = self.posicion_classifier(cls_output)   # (batch, 4)

        result: dict[str, torch.Tensor | None] = {
            "logits_color":    logits_color,
            "logits_estilo":   logits_estilo,
            "logits_elemento": logits_elemento,
            "logits_posicion": logits_posicion,
        }

        # ── 3. Pérdidas (solo cuando se pasan etiquetas) ─────────────────────
        if labels_color is not None:
            loss_color    = self._loss_fn(logits_color,    labels_color)
            loss_estilo   = self._loss_fn(logits_estilo,   labels_estilo)
            loss_elemento = self._loss_fn(logits_elemento, labels_elemento)
            loss_posicion = self._loss_fn(logits_posicion, labels_posicion)

            # Pérdida total: suma sin ponderar de las cuatro tareas.
            # Equivalente a asignar un peso de 1.0 a cada tarea.
            total_loss: torch.Tensor = (
                loss_color + loss_estilo + loss_elemento + loss_posicion
            )

            result.update({
                "loss":          total_loss,
                "loss_color":    loss_color,
                "loss_estilo":   loss_estilo,
                "loss_elemento": loss_elemento,
                "loss_posicion": loss_posicion,
            })
        else:
            result["loss"] = None

        return result
