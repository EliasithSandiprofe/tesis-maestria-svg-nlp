"""
build_dataset.py
Orquestador del pipeline completo de generación del dataset sintético.

Uso:
    python dataset/generator/build_dataset.py

Pipeline:
    1. Cargar labels.yaml y templates.yaml  (ConfigLoader)
    2. Generar 2 000 combinaciones           (CombinationGenerator)
    3. Generar prompts                       (PromptGenerator)
    4. Validar el dataset                    (DatasetValidator)
    5. Exportar archivos                     (DatasetExporter)
"""
import logging
import sys
from pathlib import Path

from config_loader import ConfigLoader
from combination_generator import CombinationGenerator
from prompt_generator import PromptGenerator
from validator import DatasetValidator
from exporter import DatasetExporter

# ------------------------------------------------------------------
# Constantes de configuración
# ------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIG_DIR: Path = _REPO_ROOT / "dataset" / "config"
_DATASET_DIR: Path = _REPO_ROOT / "dataset"

LABELS_PATH: Path = _CONFIG_DIR / "labels.yaml"
TEMPLATES_PATH: Path = _CONFIG_DIR / "templates.yaml"

TARGET_SIZE: int = 2_000
SEED: int = 42

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def main() -> None:
    """Ejecuta el pipeline completo de generación del dataset sintético.

    Pasos
    -----
    1. Carga ``labels.yaml`` y ``templates.yaml`` con :class:`ConfigLoader`.
    2. Genera ``TARGET_SIZE`` combinaciones con :class:`CombinationGenerator`.
    3. Genera los textos de prompt con :class:`PromptGenerator`.
    4. Valida el dataset con :class:`DatasetValidator`.
       Si la validación falla, el proceso se detiene con código de salida 1.
    5. Exporta los archivos con :class:`DatasetExporter`.
    6. Imprime un resumen final con las rutas generadas.

    Archivos producidos
    -------------------
    - ``dataset/raw/dataset_prompts_svg_2000.csv``
    - ``dataset/processed/dataset_training.csv``
    - ``dataset/reports/dataset_report.md``
    """

    # --- Paso 1: Cargar configuración ---
    log.info("Cargando configuración...")
    loader = ConfigLoader(labels_path=LABELS_PATH, templates_path=TEMPLATES_PATH)
    labels, templates = loader.load_all()
    log.info(
        "Labels cargados — colores: %d, estilos: %d, elementos: %d, posiciones: %d",
        len(labels["color"]),
        len(labels["estilo"]),
        len(labels["elemento"]),
        len(labels["posicion"]),
    )
    log.info("Plantillas cargadas: %d", len(templates))

    # --- Paso 2: Generar combinaciones ---
    log.info("Generando combinaciones (target=%d, seed=%d)...", TARGET_SIZE, SEED)
    gen = CombinationGenerator(labels=labels, target_size=TARGET_SIZE, seed=SEED)
    base = gen.generar_base()
    combinaciones = gen.completar(base)
    log.info(
        "Combinaciones base: %d | Total con remuestreo: %d",
        len(base),
        len(combinaciones),
    )

    # --- Paso 3: Generar prompts ---
    log.info("Generando prompts...")
    dataset = PromptGenerator(templates=templates, seed=SEED).generate_dataset(
        combinaciones
    )
    log.info("Prompts generados: %d", len(dataset))

    # --- Paso 4: Validar ---
    log.info("Validando dataset...")
    reporte = DatasetValidator(labels=labels, expected_size=TARGET_SIZE).validate(
        dataset
    )

    if not reporte["valido"]:
        log.error("Validación fallida. El dataset NO será exportado.")
        for err in reporte["errores"]:
            log.error("  → %s", err)
        sys.exit(1)

    log.info(
        "Validación exitosa — IDs únicos: %d | Prompts duplicados: %d",
        reporte["ids_unicos"],
        reporte["prompts_duplicados"],
    )

    # --- Paso 5: Exportar ---
    log.info("Exportando archivos...")
    rutas = DatasetExporter(base_path=_DATASET_DIR).export_all(dataset, reporte)

    # --- Resumen final ---
    separador = "-" * 60
    print(separador)
    print("DATASET GENERADO EXITOSAMENTE")
    print(separador)
    print(f"  Registros totales  : {reporte['total_registros']}")
    print(f"  IDs únicos         : {reporte['ids_unicos']}")
    print(f"  Prompts duplicados : {reporte['prompts_duplicados']}")
    print(f"  Errores            : {len(reporte['errores'])}")
    print(separador)
    print("Archivos exportados:")
    for clave, ruta in rutas.items():
        print(f"  {clave:<15}: {ruta}")
    print(separador)


if __name__ == "__main__":
    main()

