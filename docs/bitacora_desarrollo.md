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
