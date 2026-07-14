"""
evaluator.py
------------
Implementa ``MultiTaskEvaluator``: inferencia completa sobre el conjunto de
prueba y cálculo de métricas de clasificación para las cuatro tareas.

El evaluador NO pasa etiquetas al modelo durante la inferencia; solo utiliza
``input_ids`` y ``attention_mask`` para obtener los logits y calcula las
métricas comparando las predicciones (argmax) con los valores reales.

Clases
------
MultiTaskEvaluator
    Evaluador compatible con CPU y CUDA.  Devuelve un diccionario
    estructurado con todas las métricas de sklearn por tarea.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


_TASKS: list[str] = ["color", "estilo", "elemento", "posicion"]


class MultiTaskEvaluator:
    """Evaluador final del modelo MultiTaskDistilBERT sobre el conjunto test.

    Ejecuta el forward pass en modo inferencia (sin gradientes) para todos
    los batches del DataLoader de prueba, acumula predicciones y etiquetas
    reales, y calcula las métricas de clasificación estándar para cada tarea.

    Parameters
    ----------
    model : nn.Module
        Instancia de ``MultiTaskDistilBERT`` con los pesos del mejor
        checkpoint cargados y movida al dispositivo correcto.
    test_loader : DataLoader
        DataLoader del conjunto de prueba (``dataset_test.csv``).
    device : torch.device
        Dispositivo de inferencia (CPU o CUDA).
    label_encoders : dict[str, LabelEncoder]
        Diccionario ``{nombre_tarea: LabelEncoder}`` para obtener los
        nombres de clase originales.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        label_encoders: dict[str, LabelEncoder],
    ) -> None:
        self.model         = model
        self.test_loader   = test_loader
        self.device        = device
        self.label_encoders = label_encoders

    # ── Evaluación principal ─────────────────────────────────────────────────

    def run(self) -> dict:
        """Ejecuta la evaluación completa sobre el conjunto de prueba.

        Pasos:
        1. Modo ``eval()`` — desactiva dropout.
        2. Forward pass sin gradientes para cada batch.
        3. Predicción por argmax sobre los logits de cada cabeza.
        4. Cálculo de métricas con ``sklearn.metrics``.

        Returns
        -------
        dict
            Diccionario con claves ``"color"``, ``"estilo"``, ``"elemento"``,
            ``"posicion"`` y ``"global"``.  Cada tarea contiene:
            ``accuracy``, ``precision_macro``, ``recall_macro``,
            ``f1_macro``, ``classification_report``, ``confusion_matrix``,
            ``y_true``, ``y_pred``, ``class_names``.
            La clave ``"global"`` contiene ``mean_accuracy`` y
            ``mean_f1_macro``.
        """
        self.model.eval()

        # Acumuladores
        y_true: dict[str, list[int]] = {t: [] for t in _TASKS}
        y_pred: dict[str, list[int]] = {t: [] for t in _TASKS}

        with torch.no_grad():
            for batch in self.test_loader:
                # Mover tensores de entrada al dispositivo
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass solo con entradas (sin etiquetas)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                for task in _TASKS:
                    preds  = torch.argmax(outputs[f"logits_{task}"], dim=1)
                    labels = batch[f"labels_{task}"]

                    y_pred[task].extend(preds.cpu().numpy().tolist())
                    y_true[task].extend(labels.cpu().numpy().tolist())

        # Calcular métricas por tarea
        results: dict = {}
        for task in _TASKS:
            results[task] = self._compute_task_metrics(
                y_true=y_true[task],
                y_pred=y_pred[task],
                le=self.label_encoders[task],
                task=task,
            )

        # Métricas globales
        mean_accuracy = sum(results[t]["accuracy"] for t in _TASKS) / len(_TASKS)
        mean_f1_macro = sum(results[t]["f1_macro"] for t in _TASKS) / len(_TASKS)

        results["global"] = {
            "mean_accuracy": mean_accuracy,
            "mean_f1_macro": mean_f1_macro,
            "n_test_samples": len(y_true[_TASKS[0]]),
        }

        self._print_summary(results)
        return results

    # ── Helpers privados ─────────────────────────────────────────────────────

    def _compute_task_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        le: LabelEncoder,
        task: str,
    ) -> dict:
        """Calcula todas las métricas sklearn para una única tarea.

        Parameters
        ----------
        y_true : list[int]
            Etiquetas reales codificadas como enteros.
        y_pred : list[int]
            Predicciones del modelo codificadas como enteros.
        le : LabelEncoder
            Encoder de la tarea para obtener los nombres de clase.
        task : str
            Nombre de la tarea (para el ``classification_report``).

        Returns
        -------
        dict
            Métricas de la tarea.
        """
        class_names = list(le.classes_)

        acc       = accuracy_score(y_true, y_pred)
        prec_mac  = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_mac   = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mac    = f1_score(y_true, y_pred, average="macro", zero_division=0)
        clf_report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)

        return {
            "y_true":                y_true,
            "y_pred":                y_pred,
            "class_names":           class_names,
            "n_classes":             len(class_names),
            "accuracy":              float(acc),
            "precision_macro":       float(prec_mac),
            "recall_macro":          float(rec_mac),
            "f1_macro":              float(f1_mac),
            "classification_report": clf_report,
            "confusion_matrix":      cm.tolist(),
        }

    def _print_summary(self, results: dict) -> None:
        """Imprime el resumen de evaluación en consola."""
        glob = results["global"]
        sep  = "=" * 66

        print()
        print(f"  {sep}")
        print(f"  RESULTADOS DE EVALUACIÓN — conjunto test  ({glob['n_test_samples']} muestras)")
        print(f"  {sep}")
        print(f"  {'Tarea':<14} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1 Macro':>9}")
        print(f"  {'─' * 56}")
        for task in _TASKS:
            m = results[task]
            print(
                f"  {task:<14}"
                f" {m['accuracy']*100:>8.1f}%"
                f" {m['precision_macro']*100:>9.1f}%"
                f" {m['recall_macro']*100:>7.1f}%"
                f" {m['f1_macro']*100:>8.1f}%"
            )
        print(f"  {'─' * 56}")
        print(f"  {'Mean':<14} {glob['mean_accuracy']*100:>8.1f}%  {'':>9}  {'':>7}  {glob['mean_f1_macro']*100:>8.1f}%")
        print(f"  {sep}")
        print()
