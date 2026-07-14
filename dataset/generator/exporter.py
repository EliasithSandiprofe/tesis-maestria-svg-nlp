"""
exporter.py
Exporta el dataset sintético validado a los formatos requeridos
y genera el reporte de validación en Markdown.
"""
from pathlib import Path
from datetime import datetime

import pandas as pd

# Nombres de archivo fijos
_NOMBRE_RAW = "dataset_prompts_svg_2000.csv"
_NOMBRE_TRAINING = "dataset_training.csv"
_NOMBRE_REPORTE = "dataset_report.md"

# Columnas para el CSV de entrenamiento (sin id)
_COLUMNAS_TRAINING = ["prompt", "color", "estilo", "elemento", "posicion"]


class DatasetExporter:
    """Guarda el dataset y el reporte de validación en disco.

    Estructura de salida
    --------------------
    base_path/
    ├── raw/
    │   └── dataset_prompts_svg_2000.csv   ← todas las columnas
    ├── processed/
    │   └── dataset_training.csv           ← solo columnas de entrenamiento
    └── reports/
        └── dataset_report.md              ← reporte de validación en Markdown
    """

    def __init__(self, base_path: str | Path) -> None:
        """
        Parámetros
        ----------
        base_path : str | Path
            Raíz que contiene las carpetas raw/, processed/ y reports/.
            Se crea automáticamente si no existe.
        """
        self.base_path = Path(base_path)
        self._raw_dir = self.base_path / "raw"
        self._processed_dir = self.base_path / "processed"
        self._reports_dir = self.base_path / "reports"

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def export_raw_csv(self, dataset: list[dict]) -> Path:
        """Exporta el dataset completo (todas las columnas) en raw/.

        Parámetros
        ----------
        dataset : list[dict]
            Lista de registros con columnas id, prompt, color, estilo,
            elemento, posicion.

        Retorna
        -------
        Path
            Ruta absoluta del archivo CSV generado.

        Lanza
        -----
        ValueError
            Si el dataset está vacío.
        """
        if not dataset:
            raise ValueError("El dataset no puede estar vacío.")

        destino = self._raw_dir / _NOMBRE_RAW
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(dataset).to_csv(destino, index=False, encoding="utf-8")
        return destino.resolve()

    def export_training_csv(self, dataset: list[dict]) -> Path:
        """Exporta el CSV de entrenamiento (sin id) en processed/.

        Contiene únicamente las columnas:
        prompt, color, estilo, elemento, posicion.

        Parámetros
        ----------
        dataset : list[dict]
            Lista de registros con al menos las columnas de entrenamiento.

        Retorna
        -------
        Path
            Ruta absoluta del archivo CSV generado.

        Lanza
        -----
        ValueError
            Si el dataset está vacío o faltan columnas de entrenamiento.
        """
        if not dataset:
            raise ValueError("El dataset no puede estar vacío.")

        faltantes = [c for c in _COLUMNAS_TRAINING if c not in dataset[0]]
        if faltantes:
            raise ValueError(
                f"El dataset no contiene las columnas requeridas: {faltantes}"
            )

        destino = self._processed_dir / _NOMBRE_TRAINING
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(dataset)[_COLUMNAS_TRAINING].to_csv(
            destino, index=False, encoding="utf-8"
        )
        return destino.resolve()

    def export_report(self, report: dict) -> Path:
        """Guarda el reporte de validación como Markdown en reports/.

        Parámetros
        ----------
        report : dict
            Diccionario devuelto por ``DatasetValidator.validate()``.
            Claves esperadas: total_registros, ids_unicos,
            prompts_duplicados, errores, valido.

        Retorna
        -------
        Path
            Ruta absoluta del archivo Markdown generado.
        """
        destino = self._reports_dir / _NOMBRE_REPORTE
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        estado = "VÁLIDO" if report.get("valido") else "INVÁLIDO"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lineas = [
            "# Reporte de validación del dataset",
            "",
            f"**Fecha:** {timestamp}  ",
            f"**Estado:** {estado}  ",
            "",
            "## Resumen",
            "",
            f"| Métrica | Valor |",
            f"|---|---|",
            f"| Total registros | {report.get('total_registros', '-')} |",
            f"| IDs únicos | {report.get('ids_unicos', '-')} |",
            f"| Prompts duplicados | {report.get('prompts_duplicados', '-')} |",
            f"| Errores encontrados | {len(report.get('errores', []))} |",
            "",
        ]

        errores = report.get("errores", [])
        if errores:
            lineas += ["## Errores", ""]
            for err in errores:
                lineas.append(f"- {err}")
            lineas.append("")
        else:
            lineas += ["## Errores", "", "_Sin errores._", ""]

        destino.write_text("\n".join(lineas), encoding="utf-8")
        return destino.resolve()

    def export_all(self, dataset: list[dict], report: dict) -> dict[str, Path]:
        """Ejecuta los tres métodos de exportación en una sola llamada.

        Parámetros
        ----------
        dataset : list[dict]
            Dataset completo generado y validado.
        report : dict
            Reporte devuelto por ``DatasetValidator.validate()``.

        Retorna
        -------
        dict[str, Path]
            Rutas absolutas bajo las claves ``raw_csv``, ``training_csv``
            y ``report``.
        """
        return {
            "raw_csv": self.export_raw_csv(dataset),
            "training_csv": self.export_training_csv(dataset),
            "report": self.export_report(report),
        }


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from config_loader import ConfigLoader
    from combination_generator import CombinationGenerator
    from prompt_generator import PromptGenerator
    from validator import DatasetValidator

    _BASE_CONFIG = Path(__file__).resolve().parents[2] / "dataset" / "config"
    _BASE_DATASET = Path(__file__).resolve().parents[2] / "dataset"

    # 1. Cargar configuración
    loader = ConfigLoader(
        labels_path=_BASE_CONFIG / "labels.yaml",
        templates_path=_BASE_CONFIG / "templates.yaml",
    )
    labels, templates = loader.load_all()

    # 2. Generar combinaciones
    combinaciones = CombinationGenerator(
        labels=labels, target_size=2_000, seed=42
    ).generar()

    # 3. Generar prompts
    dataset = PromptGenerator(templates=templates, seed=42).generate_dataset(
        combinaciones
    )

    # 4. Validar
    reporte = DatasetValidator(labels=labels, expected_size=2_000).validate(dataset)

    # 5. Exportar
    rutas = DatasetExporter(base_path=_BASE_DATASET).export_all(dataset, reporte)

    print("Archivos generados:")
    for clave, ruta in rutas.items():
        print(f"  {clave:<15}: {ruta}")
