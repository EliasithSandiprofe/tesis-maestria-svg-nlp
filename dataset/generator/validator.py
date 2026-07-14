"""
validator.py
Valida la integridad del dataset sintético antes de exportarlo.
"""
from pathlib import Path

# Columnas obligatorias en cada registro del dataset
_COLUMNAS_REQUERIDAS: frozenset[str] = frozenset(
    {"id", "prompt", "color", "estilo", "elemento", "posicion"}
)

# Categorías de etiqueta que se verifican contra labels.yaml
_CATEGORIAS_ETIQUETA: tuple[str, ...] = ("color", "estilo", "elemento", "posicion")


class DatasetValidator:
    """Valida el dataset sintético antes de exportarlo.

    Comprueba:
    - Tamaño exacto esperado.
    - Columnas obligatorias presentes en cada registro.
    - Prompts no vacíos.
    - Valores de etiqueta dentro de las clases aprobadas.
    - IDs únicos e igual al número de registros.
    """

    def __init__(
        self,
        labels: dict[str, list[str]],
        expected_size: int = 2_000,
    ) -> None:
        """
        Parámetros
        ----------
        labels : dict[str, list[str]]
            Clases válidas por categoría, cargadas desde labels.yaml.
        expected_size : int, optional
            Número exacto de registros que debe tener el dataset. Por defecto 2 000.

        Lanza
        -----
        ValueError
            Si ``labels`` no contiene alguna de las categorías requeridas.
        """
        faltantes = [c for c in _CATEGORIAS_ETIQUETA if c not in labels]
        if faltantes:
            raise ValueError(
                f"El diccionario de labels no contiene las categorías: {faltantes}"
            )

        self.labels = {k: set(v) for k, v in labels.items()}
        self.expected_size = expected_size

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------

    def validate(self, dataset: list[dict]) -> dict:
        """Ejecuta todas las validaciones y devuelve un reporte de integridad.

        Validaciones aplicadas
        ----------------------
        1. Total de registros igual a ``expected_size``.
        2. Columnas obligatorias presentes en cada registro.
        3. Ningún prompt vacío o solo espacios.
        4. Valores de color, estilo, elemento y posicion dentro de labels.
        5. IDs únicos y totales iguales a ``expected_size``.

        Nota: Las combinaciones repetidas son **aceptadas** porque el dataset
        incluye 800 entradas extra sobre las 1 200 combinaciones base.
        Los prompts duplicados se **cuentan pero no se rechazan**.

        Parámetros
        ----------
        dataset : list[dict]
            Lista de registros generados por ``PromptGenerator.generate_dataset()``.

        Retorna
        -------
        dict con claves:
            - ``total_registros``  (int)  número de registros recibidos.
            - ``ids_unicos``       (int)  cantidad de IDs distintos.
            - ``prompts_duplicados`` (int) prompts que aparecen más de una vez.
            - ``errores``          (list[str]) mensajes de error encontrados.
            - ``valido``           (bool) True si ``errores`` está vacío.
        """
        errores: list[str] = []

        # --- 1. Tamaño total ---
        total = len(dataset)
        if total != self.expected_size:
            errores.append(
                f"Tamaño incorrecto: se esperaban {self.expected_size} registros, "
                f"se recibieron {total}."
            )

        # --- 2. Columnas obligatorias ---
        for i, registro in enumerate(dataset):
            faltantes = _COLUMNAS_REQUERIDAS - registro.keys()
            if faltantes:
                errores.append(
                    f"Registro #{i + 1}: faltan columnas {sorted(faltantes)}."
                )

        # A partir de aquí solo continuamos si las columnas son correctas
        # para evitar KeyError en las validaciones siguientes.
        if errores:
            return self._reporte(total, dataset, errores)

        # --- 3. Prompts no vacíos ---
        for registro in dataset:
            if not str(registro["prompt"]).strip():
                errores.append(
                    f"Registro id={registro['id']}: el prompt está vacío."
                )

        # --- 4. Valores de etiqueta dentro de las clases aprobadas ---
        for registro in dataset:
            for categoria in _CATEGORIAS_ETIQUETA:
                valor = registro[categoria]
                if valor not in self.labels[categoria]:
                    errores.append(
                        f"Registro id={registro['id']}: valor '{valor}' no válido "
                        f"para la categoría '{categoria}'."
                    )

        # --- 5. IDs únicos ---
        ids = [registro["id"] for registro in dataset]
        ids_unicos = len(set(ids))
        if ids_unicos != self.expected_size:
            errores.append(
                f"IDs no únicos: se esperaban {self.expected_size} IDs distintos, "
                f"se encontraron {ids_unicos}."
            )

        return self._reporte(total, dataset, errores)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _reporte(
        self, total: int, dataset: list[dict], errores: list[str]
    ) -> dict:
        """Construye el diccionario de reporte final."""
        # Conteo de prompts duplicados (solo si las columnas están presentes)
        prompts_duplicados = 0
        if dataset and "prompt" in dataset[0]:
            from collections import Counter
            conteo = Counter(str(r["prompt"]) for r in dataset)
            prompts_duplicados = sum(
                cnt - 1 for cnt in conteo.values() if cnt > 1
            )

        ids_unicos = 0
        if dataset and "id" in dataset[0]:
            ids_unicos = len({r["id"] for r in dataset})

        return {
            "total_registros": total,
            "ids_unicos": ids_unicos,
            "prompts_duplicados": prompts_duplicados,
            "errores": errores,
            "valido": len(errores) == 0,
        }


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from config_loader import ConfigLoader
    from combination_generator import CombinationGenerator
    from prompt_generator import PromptGenerator

    _BASE = Path(__file__).resolve().parents[2] / "dataset" / "config"

    loader = ConfigLoader(
        labels_path=_BASE / "labels.yaml",
        templates_path=_BASE / "templates.yaml",
    )
    labels, templates = loader.load_all()

    combinaciones = CombinationGenerator(
        labels=labels, target_size=2_000, seed=42
    ).generar()

    dataset = PromptGenerator(templates=templates, seed=42).generate_dataset(
        combinaciones
    )

    reporte = DatasetValidator(labels=labels, expected_size=2_000).validate(dataset)

    print("=== Reporte de validación ===")
    print(f"  Total registros    : {reporte['total_registros']}")
    print(f"  IDs únicos         : {reporte['ids_unicos']}")
    print(f"  Prompts duplicados : {reporte['prompts_duplicados']}")
    print(f"  Errores            : {len(reporte['errores'])}")
    if reporte["errores"]:
        for err in reporte["errores"]:
            print(f"    - {err}")
    print(f"  Válido             : {reporte['valido']}")
