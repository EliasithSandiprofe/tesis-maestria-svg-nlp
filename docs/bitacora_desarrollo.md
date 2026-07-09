# Bitácora de desarrollo del MVP

## Sesión 01: Configuración inicial del entorno

**Fecha:** 07/07/2026  
**Responsable:** Eliasith Sandi  
**Proyecto:** Sistema basado en NLP y aprendizaje automático para generar diseños SVG de camisetas desde prompts

### Objetivo de la sesión

Configurar el entorno inicial de desarrollo para comenzar la implementación del Producto Mínimo Viable (MVP), correspondiente a la Entrega 3 del proyecto.

### Actividades realizadas

- Activación del entorno virtual `venv`.
- Verificación de la versión de Python instalada.
- Actualización de `pip`, `setuptools` y `wheel`.
- Instalación de dependencias base:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Streamlit
- Verificación de instalación mediante una prueba de importación en Python.

### Comandos utilizados

```powershell
.\venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools wheel
pip install numpy pandas scikit-learn matplotlib streamlit
python -c "import numpy, pandas, sklearn, streamlit; print('Dependencias base instaladas correctamente')"
```


## Evidencia

![Configuración del entorno](capturas/EV-001-configuracion-entorno-dependencias-base.png)


---

# Sesión 02: Configuración del entorno NLP

**Fecha:** 07/07/2026  
**Responsable:** Eliasith Sandi  
**Proyecto:** Sistema basado en NLP y aprendizaje automático para generar diseños SVG de camisetas desde prompts

## Objetivo de la sesión

Preparar el entorno para el desarrollo del módulo de Procesamiento de Lenguaje Natural (NLP), asegurando la compatibilidad entre Python, spaCy y sus dependencias.

## Actividades realizadas

- Se detectó un problema al importar spaCy debido a incompatibilidades del entorno inicial.
- Se migró el proyecto a una ruta más corta para evitar conflictos con Windows.
- Se recreó el entorno virtual utilizando Python 3.10.11.
- Se actualizaron las herramientas base (`pip`, `setuptools` y `wheel`).
- Se instaló la biblioteca spaCy.
- Se resolvió la dependencia faltante (`click`).
- Se verificó correctamente la instalación de spaCy.

## Comandos utilizados

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install spacy
pip install click
python -c "import spacy; print(spacy.__version__)"
```

## Resultado obtenido

El entorno de desarrollo quedó correctamente configurado para trabajar con spaCy 3.8.14 utilizando Python 3.10.11. La instalación fue validada mediante una prueba de importación exitosa.

## Evidencia

![Instalación de spaCy](capturas/EV-002-instalacion-spacy.png)

## Observaciones

La migración del proyecto a una ruta más corta permitió eliminar los problemas encontrados durante la carga inicial de spaCy, garantizando un entorno estable para el desarrollo del módulo NLP.



# Sesión 03 – Inicio del desarrollo del módulo NLP

**Fecha:** _(coloca la fecha correspondiente)_  
**Estado:** ✅ Completada

## Objetivo

Iniciar el desarrollo del módulo de Procesamiento del Lenguaje Natural (NLP) del MVP mediante la implementación de una primera versión funcional basada en spaCy, capaz de procesar prompts en español y extraer información lingüística relevante para el proyecto.

---

## Actividades realizadas

### 1. Verificación del entorno de desarrollo

Se verificó el correcto funcionamiento del entorno configurado en las sesiones anteriores.

- Python 3.10.11
- Entorno virtual (venv) activo
- spaCy 3.8.14 instalado y operativo

**Resultado:** Entorno listo para iniciar el desarrollo del módulo NLP.

---

### 2. Instalación del modelo de español de spaCy

Se descargó e instaló el modelo lingüístico:

- `es_core_news_sm`

Posteriormente se verificó su correcta carga mediante una prueba de importación y ejecución.

**Resultado:** Modelo instalado y disponible para el procesamiento de texto en español.

---

### 3. Creación de la estructura inicial del módulo NLP

Se creó la estructura base del proyecto para el desarrollo del módulo de procesamiento de lenguaje natural.

```text
src/
│
├── main.py
│
└── nlp/
    ├── __init__.py
    ├── processor.py
    └── test_prompt.py
```

Se implementó la clase `NLPProcessor`, responsable de cargar el modelo de spaCy y realizar el análisis inicial de los prompts.

---

### 4. Primera prueba funcional

Se procesó el siguiente prompt de prueba:

> "Quiero una camiseta negra con un dragón rojo en estilo minimalista"

El sistema obtuvo correctamente:

- Tokens
- Lemas
- Sustantivos
- Adjetivos
- Entidades nombradas

Esta prueba confirma el correcto funcionamiento de la primera versión del módulo NLP.

---

## Resultados obtenidos

Al finalizar esta sesión el proyecto dispone de un módulo NLP funcional (Versión 0.1) capaz de:

- Procesar prompts escritos en español.
- Tokenizar el texto.
- Obtener lemas.
- Identificar sustantivos y adjetivos.
- Detectar entidades nombradas.
- Devolver la información estructurada mediante la clase `NLPProcessor`.

Este constituye el primer componente funcional del MVP.

---

## Decisiones técnicas

Durante la sesión se revisó el código generado inicialmente con GitHub Copilot.

Después de la revisión técnica se decidió:

- Mantener la arquitectura propuesta.
- Conservar la clase `NLPProcessor` como base del módulo NLP.
- Posponer mejoras avanzadas (logging, manejo de excepciones y tipado más específico) para etapas posteriores del proyecto.
- Considerar esta implementación como la **Versión 0.1 del módulo NLP**.

Asimismo, se revisó nuevamente la planificación general del proyecto y se confirmó que el desarrollo continuará siguiendo el ciclo completo de un proyecto de Inteligencia Artificial. Antes de integrar el motor SVG será necesario completar el diseño del dataset sintético, el entrenamiento supervisado del modelo DistilBERT y su evaluación mediante métricas de clasificación.

---

## Evidencias

**EV-003** – Primer procesamiento de un prompt utilizando el modelo `es_core_news_sm` de spaCy.

**Archivo:**

```
docs/capturas/EV-003_nlp_primer_prompt.png
```

La evidencia muestra la ejecución exitosa del módulo NLP y la extracción de información lingüística a partir de un prompt de prueba.

---

## Estado del proyecto

| Componente | Estado |
|------------|--------|
| Entorno de desarrollo | ✅ |
| spaCy | ✅ |
| Modelo español | ✅ |
| Módulo NLP | ✅ Versión 0.1 |
| Primera prueba funcional | ✅ |
| Evidencia EV-003 | ✅ |

---

## Próxima sesión

**Sesión 04**

Diseño e implementación del dataset sintético de 2,000 prompts para el entrenamiento supervisado del modelo DistilBERT, incluyendo la definición de atributos, estructura del conjunto de datos y estrategia de preparación para el entrenamiento.


## Sesión 04 – Diseño y generación del dataset sintético

**Fecha:** 07/07/2026  
**Objetivo:** Diseñar y construir el dataset sintético de prompts para el entrenamiento del modelo DistilBERT.

### Actividades realizadas

Durante esta sesión se diseñó y construyó el dataset sintético del proyecto, alineado con la propuesta de innovación definida en la Entrega 2. El dataset fue creado para entrenar un modelo NLP basado en DistilBERT, utilizando prompts relacionados con diseños SVG de camisetas personalizadas.

Se definieron los atributos principales del dataset:

- Color
- Estilo
- Elemento gráfico principal
- Posición del diseño

También se definieron las clases correspondientes a cada atributo, respetando el alcance del MVP planteado en la Entrega 2.

Posteriormente, se implementó un pipeline reproducible para generar automáticamente el dataset sintético. Este pipeline incluye carga de configuración, generación de combinaciones, creación de prompts, validación del dataset y exportación de archivos.

### Archivos generados

- `dataset/raw/dataset_prompts_svg_2000.csv`
- `dataset/processed/dataset_training.csv`
- `dataset/reports/dataset_report.md`

### Resultado obtenido

El dataset final contiene:

- 2,000 registros.
- 6 columnas en el archivo completo.
- 4 atributos de clasificación.
- 6 clases de color.
- 5 clases de estilo.
- 10 clases de elemento gráfico.
- 4 clases de posición.

La validación automática confirmó:

- 2,000 registros generados.
- 2,000 identificadores únicos.
- 0 errores de validación.
- 0 valores nulos.
- Balance adecuado entre clases.
- 1,920 prompts únicos.

### Evidencia generada

Se generó la evidencia:

`EV-004_dataset_sintetico_generado.png`

Esta evidencia muestra la ejecución exitosa del pipeline de generación del dataset sintético.

### Conclusión de la sesión

La Sesión 04 se considera completada exitosamente. El proyecto cuenta ahora con un dataset sintético reproducible, validado y listo para ser utilizado en la siguiente etapa del Sprint 1: la preparación de datos para el entrenamiento del modelo DistilBERT.



Sesión 05 – Desarrollo, entrenamiento y evaluación del modelo MultiTask DistilBERT

Fecha: 08/07/2026
Estado: ✅ Completada

Objetivo

Implementar el flujo completo de entrenamiento y evaluación del modelo de Inteligencia Artificial del proyecto, incluyendo el análisis exploratorio de datos, el preprocesamiento del dataset, la construcción de la arquitectura MultiTask DistilBERT, el entrenamiento supervisado utilizando GPU y la evaluación final sobre un conjunto de prueba independiente.

Actividades realizadas
1. Análisis Exploratorio de Datos (EDA)

Se desarrolló un módulo completo de Análisis Exploratorio de Datos (EDA) sobre el dataset sintético generado en la sesión anterior.

El análisis incluyó:

Verificación de valores nulos.
Identificación de registros duplicados.
Distribución de clases por atributo.
Estadísticas descriptivas de los prompts.
Análisis de longitud de texto.
Generación automática de reportes.
Generación automática de gráficos estadísticos.

Como resultado se obtuvo un diagnóstico completo del dataset antes del entrenamiento del modelo.

Resultado obtenido

2 000 registros analizados.
80 registros duplicados detectados.
Dataset balanceado entre clases.
Dataset validado para iniciar el entrenamiento.
2. Preprocesamiento del dataset

Se implementó un pipeline completo de preparación de datos para entrenamiento supervisado.

Las actividades realizadas fueron:

Limpieza del dataset.
Eliminación de registros duplicados.
Codificación de etiquetas mediante LabelEncoder.
División del dataset utilizando la estrategia:
70 % entrenamiento.
15 % validación.
15 % prueba.
Tokenización mediante DistilBertTokenizerFast.
Construcción del Dataset de PyTorch.
Construcción de los DataLoader para entrenamiento y validación.

Resultado obtenido

El conjunto de datos quedó dividido de la siguiente manera:

Conjunto	Registros
Train	1 344
Validation	288
Test	288
3. Construcción del modelo MultiTask DistilBERT

Se implementó la arquitectura del modelo de Inteligencia Artificial basada en DistilBERT.

La arquitectura desarrollada utiliza un encoder compartido y cuatro cabezas independientes de clasificación para predecir simultáneamente los atributos:

Color.
Estilo.
Elemento gráfico.
Posición del diseño.

Posteriormente se realizó la validación del Forward Pass, verificando correctamente:

carga del modelo;
carga del tokenizer;
construcción del dataset;
propagación hacia adelante;
dimensiones de entrada y salida;
cálculo de la función de pérdida para cada tarea.

Resultado obtenido

La arquitectura quedó validada correctamente y lista para iniciar el entrenamiento supervisado.

4. Entrenamiento del modelo

Debido a los requerimientos computacionales del modelo DistilBERT, el entrenamiento se realizó utilizando Google Colab con aceleración mediante GPU Tesla T4.

Se configuró el proceso de entrenamiento utilizando:

DistilBERT Base.
Batch Size = 16.
Learning Rate = 2×10⁻⁵.
Weight Decay = 0.01.
Scheduler lineal con warmup.
Gradient Clipping.
Early Stopping.
Cinco épocas de entrenamiento.

Durante el entrenamiento se generaron automáticamente los siguientes artefactos:

best_model.pt
last_model.pt
training_history.csv
training_config.json
training_summary.md

Resultado obtenido

El modelo convergió correctamente durante las cinco épocas planificadas, obteniendo una reducción progresiva de la función de pérdida y una mejora constante en las métricas de validación.

5. Evaluación del modelo

Finalizado el entrenamiento se ejecutó la evaluación utilizando exclusivamente el conjunto Test, independiente del entrenamiento y de la validación.

Durante esta fase se calcularon automáticamente las métricas:

Accuracy.
Precision.
Recall.
F1 Score.
Matrices de confusión.
Reportes de clasificación por clase.

Asimismo, se generaron automáticamente los reportes de evaluación y las matrices de confusión para cada una de las cuatro tareas de clasificación.

Resultado obtenido

El modelo alcanzó un desempeño del 100 % en Accuracy y F1 Score sobre el dataset sintético utilizado durante el MVP.

Se documentó que este comportamiento es consistente con la naturaleza completamente sintética y controlada del conjunto de datos, por lo que no representa necesariamente el desempeño esperado sobre datos reales.

6. Generación automática de figuras para la documentación

Con el propósito de documentar el desarrollo del proyecto y reutilizar los resultados en la memoria de la tesis, se desarrolló un módulo adicional para generar automáticamente figuras de calidad académica.

Entre las figuras generadas se incluyen:

Arquitectura del modelo MultiTask DistilBERT.
Curva de pérdida del entrenamiento.
Curva de Accuracy durante la validación.
Matriz de confusión para la clasificación de Color.
Matriz de confusión para la clasificación de Estilo.
Matriz de confusión para la clasificación de Elemento.
Matriz de confusión para la clasificación de Posición.

Estas figuras serán utilizadas posteriormente como apoyo visual en la Entrega 3 y en la memoria final del Trabajo Fin de Máster.

Resultados obtenidos

Al finalizar esta sesión el proyecto dispone de un pipeline completo de Inteligencia Artificial capaz de:

Analizar automáticamente el dataset.
Preparar los datos para entrenamiento.
Construir el modelo MultiTask DistilBERT.
Entrenar el modelo utilizando GPU.
Evaluar el desempeño mediante métricas de clasificación.
Generar reportes automáticos.
Generar figuras técnicas para la documentación.

El módulo NLP alcanza así un estado funcional correspondiente al Producto Mínimo Viable (MVP) definido para la Entrega 3.

Evidencias generadas

Durante esta sesión se generaron las siguientes evidencias y figuras:

docs/
│
├── capturas/
│   ├── EV-001-configuracion-entorno-dependencias-base.png
│   ├── EV-002-instalacion-spacy.png
│   ├── EV-003_nlp_primer_prompt.png
│   ├── EV-004_dataset_sintetico_generado.png
│
└── figuras/
    ├── Figura_01_Arquitectura_MultiTaskDistilBERT.png
    ├── Figura_02_Curva_Loss_Entrenamiento.png
    ├── Figura_03_Curva_Accuracy_Validacion.png
    ├── Figura_04_Matriz_Confusion_Color.png
    ├── Figura_05_Matriz_Confusion_Estilo.png
    ├── Figura_06_Matriz_Confusion_Elemento.png
    └── Figura_07_Matriz_Confusion_Posicion.png
Conclusión de la sesión

La Sesión 05 se considera completada exitosamente. Con esta fase se finaliza el desarrollo del MVP del módulo de Inteligencia Artificial, disponiendo de un modelo MultiTask DistilBERT completamente funcional, entrenado, evaluado y documentado. Los artefactos generados constituyen la base para la integración con el motor SVG y para la elaboración de la Entrega 3 y del Trabajo Fin de Máster.