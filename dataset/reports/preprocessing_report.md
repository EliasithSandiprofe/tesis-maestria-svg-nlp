# Reporte de Preprocesamiento del Dataset — Fase B

> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  
> **Generado:** 2026-07-08 15:06:41  
> **Fase:** B — Preparación del Dataset  

---

## 1. Resumen Ejecutivo

| Etapa | Registros |
|---|---|
| Dataset original | 2,000 |
| Tras deduplicación | 1,920 |
| Conjunto Train | 1,344 (70.0%) |
| Conjunto Validation | 288 (15.0%) |
| Conjunto Test | 288 (15.0%) |

---

## 2. Fase 1 — Limpieza del Dataset: Deduplicación

### 2.1 Resultados

| Métrica | Valor |
|---|---|
| Registros originales | 2,000 |
| Duplicados eliminados | 80 (4.0%) |
| Registros finales | 1,920 |
| Archivo generado | `dataset_training_clean.csv` |

### 2.2 Justificación metodológica

Se eliminan los registros completamente duplicados —aquellos donde los cinco campos (`prompt`, `color`, `estilo`, `elemento`, `posicion`) son idénticos— con el objetivo de:

- Evitar que el modelo memorice ejemplos repetidos en lugar de generalizar   patrones lingüísticos.
- Garantizar que un mismo prompt no aparezca simultáneamente en los conjuntos   de entrenamiento y evaluación (fuga de datos).
- Preservar la integridad estadística de las métricas de evaluación.

Los registros originales se conservan sin modificación en `dataset_training.csv`. La eliminación afecta únicamente a las copias redundantes (80 registros, 4.0% del total).

---

## 3. Fase 2 — Codificación de Etiquetas

Se utiliza `sklearn.preprocessing.LabelEncoder` con un encoder independiente por cada variable objetivo. Los encoders se ajustan sobre el conjunto limpio **completo** (antes del split), garantizando que todos los conjuntos (train, val, test) utilicen exactamente los mismos códigos enteros.

### 3.1 Encoders generados

| Etiqueta | Clases | Archivo |
|---|---|---|
| `color` | 6 | `color_encoder.pkl` |
| `estilo` | 5 | `estilo_encoder.pkl` |
| `elemento` | 10 | `elemento_encoder.pkl` |
| `posicion` | 4 | `posicion_encoder.pkl` |

### 3.2 Mappings de clases

#### `color`

| Clase | Código entero |
|---|---|
| `azul` | 0 |
| `blanco` | 1 |
| `gris` | 2 |
| `negro` | 3 |
| `rojo` | 4 |
| `verde` | 5 |

#### `estilo`

| Clase | Código entero |
|---|---|
| `deportivo` | 0 |
| `minimalista` | 1 |
| `retro` | 2 |
| `urbano` | 3 |
| `vintage` | 4 |

#### `elemento`

| Clase | Código entero |
|---|---|
| `automóvil` | 0 |
| `calavera` | 1 |
| `dragón` | 2 |
| `guitarra` | 3 |
| `lobo` | 4 |
| `montaña` | 5 |
| `sol` | 6 |
| `texto` | 7 |
| `águila` | 8 |
| `árbol` | 9 |

#### `posicion`

| Clase | Código entero |
|---|---|
| `centrado` | 0 |
| `espalda` | 1 |
| `esquina` | 2 |
| `pecho` | 3 |

---

## 4. Fase 3 — División del Dataset

### 4.1 Decisión metodológica

El problema de clasificación presenta cuatro variables objetivo simultáneas. `train_test_split` de sklearn acepta un único array en el parámetro `stratify`, por lo que la estratificación conjunta no es directamente soportada.

**Análisis de la clave compuesta:** la concatenación de los cuatro atributos genera hasta 6×5×10×4 = 1 200 combinaciones posibles. Con 1,920 muestras disponibles tras la deduplicación, la frecuencia media por combinación es ≈ 1.6. De hecho, 602 combinaciones aparecen una única vez. Sklearn exige al menos 2 muestras por clase para estratificar, por lo que este enfoque es **inviable** para este dataset.

**Estrategia adoptada:** split en dos etapas estratificado sobre `elemento` (la etiqueta con mayor cardinalidad, 10 clases, ~192 muestras/clase). Dado que las cuatro etiquetas están bien balanceadas (ratio min/max ≥ 0.89 confirmado en el EDA), esta estrategia preserva distribuciones estadísticamente comparables en todas las etiquetas. La reproducibilidad queda garantizada mediante `random_state=42`.

### 4.2 Resultados

| Conjunto | Registros | Porcentaje |
|---|---|---|
| Train | 1,344 | 70.0% |
| Validation | 288 | 15.0% |
| Test | 288 | 15.0% |

### 4.3 Distribución de etiquetas por conjunto

#### `color`

| Clase | Train | Val | Test |
|---|---|---|---|
| `azul` | 221 | 51 | 54 |
| `blanco` | 234 | 56 | 48 |
| `gris` | 228 | 53 | 44 |
| `negro` | 225 | 38 | 45 |
| `rojo` | 218 | 37 | 47 |
| `verde` | 218 | 53 | 50 |

#### `estilo`

| Clase | Train | Val | Test |
|---|---|---|---|
| `deportivo` | 266 | 73 | 57 |
| `minimalista` | 269 | 52 | 55 |
| `retro` | 268 | 55 | 58 |
| `urbano` | 273 | 55 | 60 |
| `vintage` | 268 | 53 | 58 |

#### `elemento`

| Clase | Train | Val | Test |
|---|---|---|---|
| `automóvil` | 132 | 28 | 29 |
| `calavera` | 135 | 29 | 29 |
| `dragón` | 135 | 29 | 29 |
| `guitarra` | 141 | 30 | 30 |
| `lobo` | 123 | 26 | 26 |
| `montaña` | 139 | 30 | 29 |
| `sol` | 135 | 29 | 29 |
| `texto` | 135 | 29 | 29 |
| `águila` | 137 | 29 | 30 |
| `árbol` | 132 | 29 | 28 |

#### `posicion`

| Clase | Train | Val | Test |
|---|---|---|---|
| `centrado` | 310 | 85 | 85 |
| `espalda` | 358 | 56 | 66 |
| `esquina` | 341 | 72 | 63 |
| `pecho` | 335 | 75 | 74 |

---

## 5. Fase 4 — Tokenización

### 5.1 Configuración

| Parámetro | Valor |
|---|---|
| Modelo | `distilbert-base-uncased` |
| Tokenizer | `DistilBertTokenizerFast` |
| `max_length` | 128 |
| Padding | `max_length` |
| Truncation | `True` |
| Vocabulario | 30,522 tokens |

### 5.2 Estadísticas de longitud (conjunto Train)

| Métrica | Tokens |
|---|---|
| Mínimo | 21 |
| Máximo | 32 |
| Media | 27.23 |
| Mediana | 28.0 |
| Percentil 95 | 31 |

El percentil 95 (31 tokens) está muy por debajo del `max_length` configurado (128), confirmando que no se produce truncado en ningún prompt del dataset.

---

## 6. Fase 5 — Dataset PyTorch

### 6.1 Resumen

| Conjunto | Muestras |
|---|---|
| Train | 1,344 |
| Validation | 288 |
| Test | 288 |

### 6.2 Estructura de cada muestra

Cada llamada a `dataset[i]` devuelve un diccionario con las siguientes claves:

| Clave | Tipo | Forma | Descripción |
|---|---|---|---|
| `input_ids` | `LongTensor` | `(128,)` | IDs de tokens DistilBERT |
| `attention_mask` | `LongTensor` | `(128,)` | Máscara de atención (1=token, 0=padding) |
| `labels_color` | `LongTensor` | `()` | Código entero de la etiqueta `color` |
| `labels_estilo` | `LongTensor` | `()` | Código entero de la etiqueta `estilo` |
| `labels_elemento` | `LongTensor` | `()` | Código entero de la etiqueta `elemento` |
| `labels_posicion` | `LongTensor` | `()` | Código entero de la etiqueta `posicion` |

Esta estructura es compatible con `torch.utils.data.DataLoader` y está lista para ser consumida directamente por el modelo Multi-Head DistilBERT en la Fase C.

---

## 7. Artefactos Generados

| Artefacto | Ruta | Descripción |
|---|---|---|
| Dataset limpio | `dataset/processed/dataset_training_clean.csv` | Dataset sin duplicados |
| Split Train | `dataset/processed/dataset_train.csv` | 70% del dataset limpio |
| Split Validation | `dataset/processed/dataset_validation.csv` | 15% del dataset limpio |
| Split Test | `dataset/processed/dataset_test.csv` | 15% del dataset limpio |
| Encoder color | `modelos/label_encoders/color_encoder.pkl` | LabelEncoder serializado |
| Encoder estilo | `modelos/label_encoders/estilo_encoder.pkl` | LabelEncoder serializado |
| Encoder elemento | `modelos/label_encoders/elemento_encoder.pkl` | LabelEncoder serializado |
| Encoder posicion | `modelos/label_encoders/posicion_encoder.pkl` | LabelEncoder serializado |
| Label Mapping | `modelos/label_encoders/label_mapping.md` | Tabla de equivalencias |
| Tokenizer | `modelos/tokenizer/` | DistilBertTokenizerFast serializado |

---

> *Reporte generado automáticamente por el módulo de preprocesamiento — Fase B del proyecto de tesis.*