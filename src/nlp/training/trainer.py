"""
trainer.py
----------
Implementa ``MultiTaskTrainer``: el loop de entrenamiento completo para
el modelo Multi-Head DistilBERT SVG-NLP.

Responsabilidades
-----------------
- Optimización con AdamW y scheduler lineal con warmup.
- Gradient clipping por norma máxima.
- Cálculo de loss por tarea y loss total en cada epoch.
- Cálculo de accuracy por tarea durante validación.
- Guardado automático de ``best_model.pt`` y ``last_model.pt``.
- Early stopping basado en ``val_loss``.
- Registro del historial completo de métricas.

Clases
------
MultiTaskTrainer
    Trainer auto-contenido compatible con CPU y CUDA.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .training_config import TrainingConfig
from .training_utils import (
    format_time,
    move_batch_to_device,
    print_epoch_summary,
    save_checkpoint,
)

# Tareas del problema de clasificación multi-head
_TASKS: list[str] = ["color", "estilo", "elemento", "posicion"]


class MultiTaskTrainer:
    """Entrenador del modelo MultiTaskDistilBERT con soporte para multi-tarea.

    Parameters
    ----------
    model : nn.Module
        Instancia de ``MultiTaskDistilBERT`` inicializada y movida al
        dispositivo correcto.
    train_loader : DataLoader
        DataLoader del conjunto de entrenamiento.
    val_loader : DataLoader
        DataLoader del conjunto de validación.
    config : TrainingConfig
        Configuración del entrenamiento.

    Attributes
    ----------
    optimizer : AdamW
        Optimizador AdamW con weight_decay configurado.
    scheduler : LambdaLR
        Scheduler lineal con warmup.
    history : list[dict]
        Historial de métricas por época (se acumula durante ``fit``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = config.device

        # ── Optimizador ───────────────────────────────────────────────────────
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ── Scheduler lineal con warmup ───────────────────────────────────────
        total_steps  = len(train_loader) * config.epochs
        warmup_steps = max(1, int(total_steps * config.warmup_ratio))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self._total_steps  = total_steps
        self._warmup_steps = warmup_steps

        self.history: list[dict] = []

    # ── Epoch de entrenamiento ────────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        """Ejecuta una época completa de entrenamiento.

        Realiza forward pass, backward pass, gradient clipping y paso del
        optimizador y del scheduler para cada batch.

        Returns
        -------
        dict
            Métricas de entrenamiento:
            ``train_loss``, ``train_loss_color``, ``train_loss_estilo``,
            ``train_loss_elemento``, ``train_loss_posicion``.
        """
        self.model.train()

        total_loss  = 0.0
        task_losses = {t: 0.0 for t in _TASKS}
        n_batches   = 0

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids       = batch["input_ids"],
                attention_mask  = batch["attention_mask"],
                labels_color    = batch["labels_color"],
                labels_estilo   = batch["labels_estilo"],
                labels_elemento = batch["labels_elemento"],
                labels_posicion = batch["labels_posicion"],
            )

            loss = outputs["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            for t in _TASKS:
                task_losses[t] += outputs[f"loss_{t}"].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "train_loss":          total_loss / n,
            "train_loss_color":    task_losses["color"]    / n,
            "train_loss_estilo":   task_losses["estilo"]   / n,
            "train_loss_elemento": task_losses["elemento"] / n,
            "train_loss_posicion": task_losses["posicion"] / n,
        }

    # ── Epoch de validación ───────────────────────────────────────────────────

    def validate_one_epoch(self) -> dict:
        """Ejecuta una época completa de validación.

        Calcula pérdidas y accuracy para cada tarea sobre el conjunto de
        validación completo, sin acumular grafos de cómputo.

        Returns
        -------
        dict
            Métricas de validación:
            ``val_loss``, ``val_loss_<tarea>``,
            ``accuracy_<tarea>``, ``mean_accuracy``.
        """
        self.model.eval()

        total_loss    = 0.0
        task_losses   = {t: 0.0 for t in _TASKS}
        correct       = {t: 0   for t in _TASKS}
        total_samples = 0
        n_batches     = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = move_batch_to_device(batch, self.device)
                bs = batch["input_ids"].shape[0]
                total_samples += bs
                n_batches     += 1

                outputs = self.model(
                    input_ids       = batch["input_ids"],
                    attention_mask  = batch["attention_mask"],
                    labels_color    = batch["labels_color"],
                    labels_estilo   = batch["labels_estilo"],
                    labels_elemento = batch["labels_elemento"],
                    labels_posicion = batch["labels_posicion"],
                )

                total_loss += outputs["loss"].item()
                for t in _TASKS:
                    task_losses[t] += outputs[f"loss_{t}"].item()
                    preds = torch.argmax(outputs[f"logits_{t}"], dim=1)
                    correct[t] += int((preds == batch[f"labels_{t}"]).sum().item())

        n = max(n_batches, 1)
        ts = max(total_samples, 1)

        accuracies   = {t: correct[t] / ts for t in _TASKS}
        mean_accuracy = sum(accuracies.values()) / len(_TASKS)

        return {
            "val_loss":          total_loss / n,
            "val_loss_color":    task_losses["color"]    / n,
            "val_loss_estilo":   task_losses["estilo"]   / n,
            "val_loss_elemento": task_losses["elemento"] / n,
            "val_loss_posicion": task_losses["posicion"] / n,
            "accuracy_color":    accuracies["color"],
            "accuracy_estilo":   accuracies["estilo"],
            "accuracy_elemento": accuracies["elemento"],
            "accuracy_posicion": accuracies["posicion"],
            "mean_accuracy":     mean_accuracy,
        }

    # ── Loop principal ────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Ejecuta el loop de entrenamiento completo con early stopping.

        Por cada época:
        1. Entrena con ``train_one_epoch``.
        2. Valida con ``validate_one_epoch``.
        3. Guarda ``last_model.pt`` (siempre).
        4. Guarda ``best_model.pt`` si ``val_loss`` mejora.
        5. Incrementa el contador de paciencia si no mejora.
        6. Detiene el entrenamiento si ``patience_counter >= config.patience``.

        Returns
        -------
        dict
            Resultado del entrenamiento con las claves:
            ``history``, ``best_epoch``, ``best_val_loss``,
            ``best_metrics``, ``total_epochs_trained``.
        """
        best_val_loss     = float("inf")
        best_epoch        = 0
        best_metrics: dict | None = None
        patience_counter  = 0

        print(
            f"\n  Scheduler: warmup={self._warmup_steps} steps / "
            f"total={self._total_steps} steps\n"
        )

        for epoch in range(1, self.config.epochs + 1):
            t_epoch_start = time.perf_counter()

            # ── Entrenamiento y validación ────────────────────────────────────
            train_metrics = self.train_one_epoch()
            val_metrics   = self.validate_one_epoch()

            elapsed = time.perf_counter() - t_epoch_start

            # ── Registro ─────────────────────────────────────────────────────
            epoch_record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "epoch_time_s": round(elapsed, 2),
            }
            self.history.append(epoch_record)

            # ── ¿Mejor modelo? ────────────────────────────────────────────────
            is_best = val_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss    = val_metrics["val_loss"]
                best_epoch       = epoch
                best_metrics     = val_metrics.copy()
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1

            # ── Guardar último checkpoint (siempre) ───────────────────────────
            self._save_checkpoint(epoch, val_metrics, is_best=False)

            # ── Resumen de la época ───────────────────────────────────────────
            print_epoch_summary(epoch, self.config.epochs, epoch_record, elapsed, is_best)

            # ── Early stopping ────────────────────────────────────────────────
            if patience_counter >= self.config.patience:
                print(
                    f"\n  [Early Stopping] Val loss sin mejora durante "
                    f"{self.config.patience} épocas consecutivas. "
                    f"Entrenamiento detenido en época {epoch}."
                )
                break

        return {
            "history":            self.history,
            "best_epoch":         best_epoch,
            "best_val_loss":      best_val_loss,
            "best_metrics":       best_metrics,
            "total_epochs_trained": len(self.history),
        }

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: dict,
        is_best: bool,
    ) -> None:
        """Construye y guarda el checkpoint en la ruta correspondiente."""
        # Extraer num_classes directamente del modelo para auto-documentación
        num_classes = {
            "color":    self.model.color_classifier.out_features,
            "estilo":   self.model.estilo_classifier.out_features,
            "elemento": self.model.elemento_classifier.out_features,
            "posicion": self.model.posicion_classifier.out_features,
        }

        state = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch":                epoch,
            "val_loss":             val_metrics["val_loss"],
            "val_metrics":          val_metrics,
            "training_config":      self.config.to_dict(),
            "num_classes":          num_classes,
        }

        path = (
            self.config.best_checkpoint_path
            if is_best
            else self.config.last_checkpoint_path
        )
        save_checkpoint(state, path)
