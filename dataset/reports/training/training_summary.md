# Resumen del Entrenamiento — Fase D

> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  
> **Generado:** 2026-07-08 22:31:51  
> **Fase:** D — Entrenamiento del modelo  

---

## 1. Configuración del Entrenamiento

| Parámetro | Valor |
|---|---|
| Dispositivo | `cuda` |
| Modelo base | `distilbert-base-uncased` |
| Épocas planificadas | 5 |
| Épocas entrenadas | 5 |
| Batch size | 16 |
| Learning rate | 2e-05 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Patience (early stopping) | 2 |
| Warmup ratio | 10% |
| Random seed | 42 |
| Max length (tokens) | 128 |

---

## 2. Dataset

| Conjunto | Muestras |
|---|---|
| Train | 1,344 |
| Validation | 288 |

### Clases por tarea

| Tarea | Clases |
|---|---|
| `color` | 6 |
| `estilo` | 5 |
| `elemento` | 10 |
| `posicion` | 4 |

---

## 3. Mejor Época

| Métrica | Valor |
|---|---|
| Mejor época | 5 |
| Mejor val_loss | 0.284440 |
| Accuracy `color` | 100.00% |
| Accuracy `estilo` | 100.00% |
| Accuracy `elemento` | 100.00% |
| Accuracy `posicion` | 100.00% |
| Mean Accuracy | 100.00% |

---

## 4. Historial de Métricas

| Época | Train Loss | Val Loss | Acc Color | Acc Estilo | Acc Elem | Acc Pos | Mean Acc |
|---|---|---|---|---|---|---|---|
| 1 | 6.3983 | 4.2184 | 86.5% | 99.7% | 40.6% | 92.0% | 79.7% |
| 2 | 2.9448 | 1.4200 | 100.0% | 99.7% | 97.2% | 100.0% | 99.2% |
| 3 | 1.1620 | 0.5173 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | 0.6276 | 0.3299 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 5 | 0.4817 | 0.2844 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

---

## 5. Artefactos Generados

| Artefacto | Ruta |
|---|---|
| Mejor checkpoint | `modelos/checkpoints/best_model.pt` |
| Último checkpoint | `modelos/checkpoints/last_model.pt` |
| Historial CSV | `dataset/reports/training/training_history.csv` |
| Configuración JSON | `dataset/reports/training/training_config.json` |

---

> *Generado automáticamente por `train.py` — Fase D del proyecto de tesis.*