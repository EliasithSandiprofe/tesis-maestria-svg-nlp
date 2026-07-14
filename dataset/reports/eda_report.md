# Reporte de Análisis Exploratorio de Datos (EDA)

> **Proyecto:** Tesis de Maestría — Interpretación de Prompts NLP para generación SVG  
> **Generado:** 2026-07-08 14:49:24  
> **Fase:** A — Análisis Exploratorio de Datos  

---

## 1. Resumen del Dataset

| Métrica | Valor |
|---|---|
| Total de registros | 2,000 |
| Total de columnas | 5 |
| Columnas | `prompt`, `color`, `estilo`, `elemento`, `posicion` |
| Tipos de datos | object |
| Uso de memoria | 966.5 KB |

### Primeras 5 filas

| prompt | color | estilo | elemento | posicion |
| --- | --- | --- | --- | --- |
| Diseño de camiseta: gráfico de montaña, color negro, estilo minimalista, posición centrado | negro | minimalista | montaña | centrado |
| Quiero una camiseta con diseño de montaña en estilo minimalista, color negro, posición pecho | negro | minimalista | montaña | pecho |
| Camiseta de color negro, diseño de montaña, estilo minimalista, zona esquina | negro | minimalista | montaña | esquina |
| Crear camiseta de estilo minimalista con motivo de montaña en color negro, zona espalda | negro | minimalista | montaña | espalda |
| Necesito una camiseta con gráfico de guitarra en zona centrado, color negro, estilo minimalista | negro | minimalista | guitarra | centrado |

---

## 2. Calidad del Dataset

### 2.1 Valores Nulos

| columna | nulos | porcentaje_% |
| --- | --- | --- |
| prompt | 0 | 0.0 |
| color | 0 | 0.0 |
| estilo | 0 | 0.0 |
| elemento | 0 | 0.0 |
| posicion | 0 | 0.0 |

### 2.2 Registros Duplicados

| Métrica | Valor | % |
|---|---|---|
| Filas completamente duplicadas | 80 | 4.0% |
| Prompts duplicados | 80 | 4.0% |
| Prompts únicos | 1,920 | 96.00% |

### 2.3 Cardinalidad por Columna

| columna | valores_unicos |
| --- | --- |
| prompt | 1920 |
| color | 6 |
| estilo | 5 |
| elemento | 10 |
| posicion | 4 |

### 2.4 Valores en Blanco

| columna | valores_en_blanco |
| --- | --- |
| prompt | 0 |
| color | 0 |
| estilo | 0 |
| elemento | 0 |
| posicion | 0 |

### 2.5 Tabla Resumen de Calidad

| columna | tipo | nulos | nulos_% | valores_unicos |
| --- | --- | --- | --- | --- |
| prompt | object | 0 | 0.0 | 1920 |
| color | object | 0 | 0.0 | 6 |
| estilo | object | 0 | 0.0 | 5 |
| elemento | object | 0 | 0.0 | 10 |
| posicion | object | 0 | 0.0 | 4 |

---

## 3. Distribución de Etiquetas

### 3.1 Etiqueta: `color`

- **Clases distintas:** 6
- **Clase más frecuente:** `blanco` (17.65%)
- **Clase menos frecuente:** `rojo` (15.75%)

| valor | frecuencia | porcentaje_% |
| --- | --- | --- |
| blanco | 353 | 17.65 |
| gris | 339 | 16.95 |
| azul | 338 | 16.9 |
| verde | 332 | 16.6 |
| negro | 323 | 16.15 |
| rojo | 315 | 15.75 |

![Distribución color](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\distribucion_color.png)

### 3.2 Etiqueta: `estilo`

- **Clases distintas:** 5
- **Clase más frecuente:** `deportivo` (20.45%)
- **Clase menos frecuente:** `minimalista` (19.5%)

| valor | frecuencia | porcentaje_% |
| --- | --- | --- |
| deportivo | 409 | 20.45 |
| urbano | 405 | 20.25 |
| vintage | 400 | 20.0 |
| retro | 396 | 19.8 |
| minimalista | 390 | 19.5 |

![Distribución estilo](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\distribucion_estilo.png)

### 3.3 Etiqueta: `elemento`

- **Clases distintas:** 10
- **Clase más frecuente:** `guitarra` (10.4%)
- **Clase menos frecuente:** `lobo` (9.25%)

| valor | frecuencia | porcentaje_% |
| --- | --- | --- |
| guitarra | 208 | 10.4 |
| dragón | 205 | 10.25 |
| sol | 204 | 10.2 |
| texto | 203 | 10.15 |
| montaña | 202 | 10.1 |
| calavera | 202 | 10.1 |
| águila | 202 | 10.1 |
| automóvil | 195 | 9.75 |
| árbol | 194 | 9.7 |
| lobo | 185 | 9.25 |

![Distribución elemento](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\distribucion_elemento.png)

### 3.4 Etiqueta: `posicion`

- **Clases distintas:** 4
- **Clase más frecuente:** `espalda` (25.1%)
- **Clase menos frecuente:** `pecho` (24.8%)

| valor | frecuencia | porcentaje_% |
| --- | --- | --- |
| espalda | 502 | 25.1 |
| centrado | 501 | 25.05 |
| esquina | 501 | 25.05 |
| pecho | 496 | 24.8 |

![Distribución posicion](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\distribucion_posicion.png)

---

## 4. Análisis de los Prompts

### 4.1 Estadísticas de Longitud

| metrica | longitud_caracteres | longitud_palabras |
| --- | --- | --- |
| min | 64.0 | 11.0 |
| max | 100.0 | 14.0 |
| promedio | 81.62 | 12.47 |
| mediana | 82.0 | 13.0 |
| desv_std | 7.51 | 1.11 |

### 4.2 Histograma — Longitud en Caracteres

![Histograma caracteres](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\histograma_longitud_caracteres.png)

### 4.3 Histograma — Longitud en Palabras

![Histograma palabras](D:\ProyectoIA\tesis-maestria-svg-nlp\dataset\reports\figures\histograma_longitud_palabras.png)

---

## 5. Interpretación de Resultados

### 5.1 Calidad General

- **Sin valores nulos.** El dataset está completamente poblado.
- **Duplicados aceptables** (4.0% ≤ 5.0%). Se recomienda eliminarlos antes del entrenamiento.

### 5.2 Balance de Clases

- `color`: distribución **equilibrada** (ratio min/max = 0.89).
- `estilo`: distribución **equilibrada** (ratio min/max = 0.95).
- `elemento`: distribución **equilibrada** (ratio min/max = 0.89).
- `posicion`: distribución **equilibrada** (ratio min/max = 0.99).

### 5.3 Longitud de Prompts

- Longitud media: **81.6 caracteres** / **12.5 palabras** por prompt.
- Una longitud uniforme y moderada es favorable para el entrenamiento con modelos transformer.

---

## 6. Conclusión

El análisis de calidad no detectó valores nulos en ninguna de las columnas del dataset. Tampoco se encontraron valores en blanco, lo que confirma la integridad textual de los registros. Se identificaron 80 registros duplicados (4.0% del total), los cuales serán eliminados durante la fase de preparación del dataset, previa a la división en conjuntos de entrenamiento, validación y prueba. La distribución de clases en las cuatro variables objetivo (`color`, `estilo`, `elemento`, `posicion`) muestra un balance adecuado entre categorías, lo que favorece el aprendizaje uniforme del modelo sin requerir estrategias adicionales de sobre-muestreo o sub-muestreo en esta etapa. Los prompts presentan una longitud homogénea, con una media de 81.6 caracteres (12.5 palabras) y una desviación estándar de 7.5 caracteres, lo que es consistente con los requisitos de entrada de modelos transformer basados en la arquitectura BERT. En conjunto, el dataset reúne la calidad suficiente para proceder con la preparación de datos y el entrenamiento del modelo DistilBERT propuesto en esta investigación.

### Validación automática

- Sin valores nulos significativos (máx. 0.0% por columna). [OK]
- Tasa de duplicados dentro del umbral aceptable (4.0% ≤ 5.0%). [OK]
- Etiqueta `color` con 6 clases válidas. [OK]
- Etiqueta `estilo` con 5 clases válidas. [OK]
- Etiqueta `elemento` con 10 clases válidas. [OK]
- Etiqueta `posicion` con 4 clases válidas. [OK]
- Balance de clases en `color` aceptable (ratio 0.89). [OK]
- Balance de clases en `estilo` aceptable (ratio 0.95). [OK]
- Balance de clases en `elemento` aceptable (ratio 0.89). [OK]
- Balance de clases en `posicion` aceptable (ratio 0.99). [OK]
- Tamaño del dataset suficiente para entrenamiento (2,000 registros). [OK]

---

> *Reporte generado automáticamente por el módulo EDA — Fase A del proyecto de tesis.*