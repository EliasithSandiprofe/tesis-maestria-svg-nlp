# Reporte de Evaluación Final — Fase E

> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  
> **Generado:** 2026-07-08 16:45:36  
> **Fase:** E — Evaluación final del modelo  

---

## 1. Configuración de la Evaluación

| Parámetro | Valor |
|---|---|
| Dispositivo | `cpu` |
| Checkpoint evaluado | `best_model.pt` |
| Época del checkpoint | 1 |
| Val loss (entrenamiento) | 7.0727 |
| Dataset de prueba | `dataset_test.csv` |
| Muestras test | 32 |

---

## 2. Métricas Globales

| Métrica | Valor |
|---|---|
| **Mean Accuracy** | **24.22%** |
| **Mean F1 Macro** | **8.99%** |

---

## 3. Métricas por Tarea

### 3.1 `color` (6 clases)

| Métrica | Valor |
|---|---|
| Accuracy | 15.62% |
| Precision Macro | 2.60% |
| Recall Macro | 16.67% |
| F1 Macro | 4.50% |

**Clases:** `azul`, `blanco`, `gris`, `negro`, `rojo`, `verde`

![Matriz de confusión — color](confusion_matrix_color.png)

### 3.2 `estilo` (5 clases)

| Métrica | Valor |
|---|---|
| Accuracy | 25.00% |
| Precision Macro | 24.52% |
| Recall Macro | 23.33% |
| F1 Macro | 13.08% |

**Clases:** `deportivo`, `minimalista`, `retro`, `urbano`, `vintage`

![Matriz de confusión — estilo](confusion_matrix_estilo.png)

### 3.3 `elemento` (10 clases)

| Métrica | Valor |
|---|---|
| Accuracy | 9.38% |
| Precision Macro | 1.67% |
| Recall Macro | 4.29% |
| F1 Macro | 2.40% |

**Clases:** `automóvil`, `calavera`, `dragón`, `guitarra`, `lobo`, `montaña`, `sol`, `texto`, `águila`, `árbol`

![Matriz de confusión — elemento](confusion_matrix_elemento.png)

### 3.4 `posicion` (4 clases)

| Métrica | Valor |
|---|---|
| Accuracy | 46.88% |
| Precision Macro | 11.72% |
| Recall Macro | 25.00% |
| F1 Macro | 15.96% |

**Clases:** `centrado`, `espalda`, `esquina`, `pecho`

![Matriz de confusión — posicion](confusion_matrix_posicion.png)

---

## 4. Interpretación de Resultados

El modelo MultiTaskDistilBERT alcanzó un desempeño **bajo, lo que sugiere que el modelo requiere mayor entrenamiento o ajuste de hiperparámetros** sobre el conjunto de prueba, con una accuracy media de **24.2%** y un F1 macro medio de **9.0%** considerando las cuatro tareas de clasificación simultáneas.

La tarea con **mejor desempeño** fue `posicion` (accuracy = 46.9%), lo que indica que el modelo captura adecuadamente los patrones lingüísticos asociados a este atributo de diseño SVG. La tarea con **menor desempeño** fue `elemento` (accuracy = 9.4%), lo que puede atribuirse a una mayor ambigüedad léxica en los prompts o a la mayor cardinalidad de clases en esta dimensión.

Las matrices de confusión por tarea permiten identificar las clases con mayor tasa de error e informar decisiones de mejora en fases futuras del proyecto.

---

## 5. Artefactos Generados

| Artefacto | Descripción |
|---|---|
| `evaluation_metrics.json` | Métricas completas en formato JSON |
| `evaluation_report.md` | Este reporte |
| `classification_report_color.md` | Reporte detallado por clase — `color` |
| `confusion_matrix_color.png` | Matriz de confusión — `color` |
| `classification_report_estilo.md` | Reporte detallado por clase — `estilo` |
| `confusion_matrix_estilo.png` | Matriz de confusión — `estilo` |
| `classification_report_elemento.md` | Reporte detallado por clase — `elemento` |
| `confusion_matrix_elemento.png` | Matriz de confusión — `elemento` |
| `classification_report_posicion.md` | Reporte detallado por clase — `posicion` |
| `confusion_matrix_posicion.png` | Matriz de confusión — `posicion` |

---

> *Generado automáticamente por `evaluate.py` — Fase E del proyecto de tesis.*