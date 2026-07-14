"""
config_loader.py
Carga y valida los archivos de configuración YAML del dataset
(labels.yaml y templates.yaml).
"""
from pathlib import Path

import yaml

# Claves obligatorias en labels.yaml
_LABELS_REQUERIDAS: frozenset[str] = frozenset({"color", "estilo", "elemento", "posicion"})


class ConfigLoader:
    """Carga y valida los archivos labels.yaml y templates.yaml."""

    def __init__(self, labels_path: str | Path, templates_path: str | Path) -> None:
        """
        Parámetros
        ----------
        labels_path : str | Path
            Ruta al archivo labels.yaml.
        templates_path : str | Path
            Ruta al archivo templates.yaml.
        """
        self.labels_path = Path(labels_path)
        self.templates_path = Path(templates_path)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _leer_yaml(self, ruta: Path) -> dict | list:
        """Lee un archivo YAML y aplica validaciones básicas de existencia y contenido.

        Parámetros
        ----------
        ruta : Path
            Ruta al archivo YAML a leer.

        Retorna
        -------
        dict | list
            Contenido deserializado del archivo.

        Lanza
        -----
        FileNotFoundError
            Si el archivo no existe en la ruta indicada.
        ValueError
            Si el archivo está vacío o su contenido no es un dict ni una list.
        """
        if not ruta.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

        raw = ruta.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"El archivo está vacío: {ruta}")

        contenido = yaml.safe_load(raw)
        if contenido is None:
            raise ValueError(f"El archivo no contiene datos YAML válidos: {ruta}")

        return contenido

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def load_labels(self) -> dict[str, list[str]]:
        """Lee labels.yaml y devuelve un diccionario con las listas de clases.

        Retorna
        -------
        dict[str, list[str]]
            Claves requeridas: color, estilo, elemento, posicion.

        Lanza
        -----
        ValueError
            Si faltan claves requeridas o alguna lista está vacía.
        """
        datos: dict = self._leer_yaml(self.labels_path)

        faltantes = _LABELS_REQUERIDAS - datos.keys()
        if faltantes:
            raise ValueError(
                f"labels.yaml no contiene las claves requeridas: {sorted(faltantes)}"
            )

        for clave in _LABELS_REQUERIDAS:
            if not datos[clave]:
                raise ValueError(f"La lista '{clave}' en labels.yaml está vacía.")

        return {clave: list(datos[clave]) for clave in _LABELS_REQUERIDAS}

    def load_templates(self) -> list[str]:
        """Lee templates.yaml y devuelve la lista de plantillas de texto.

        Retorna
        -------
        list[str]
            Lista de cadenas con las variables {color}, {estilo}, {elemento}, {posicion}.

        Lanza
        -----
        ValueError
            Si falta la clave 'templates' o la lista está vacía.
        """
        datos: dict = self._leer_yaml(self.templates_path)

        if "templates" not in datos:
            raise ValueError("templates.yaml no contiene la clave 'templates'.")

        plantillas: list[str] = datos["templates"]
        if not plantillas:
            raise ValueError("La lista 'templates' en templates.yaml está vacía.")

        return list(plantillas)

    def load_all(self) -> tuple[dict[str, list[str]], list[str]]:
        """Carga labels y plantillas en una sola llamada.

        Retorna
        -------
        tuple[dict[str, list[str]], list[str]]
            (labels, templates) — ambos ya validados.
        """
        return self.load_labels(), self.load_templates()


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------

if __name__ == "__main__":
    _BASE = Path(__file__).resolve().parents[2] / "dataset" / "config"

    loader = ConfigLoader(
        labels_path=_BASE / "labels.yaml",
        templates_path=_BASE / "templates.yaml",
    )

    labels, templates = loader.load_all()

    print(f"Colores   : {len(labels['color'])}")
    print(f"Estilos   : {len(labels['estilo'])}")
    print(f"Elementos : {len(labels['elemento'])}")
    print(f"Posiciones: {len(labels['posicion'])}")
    print(f"Plantillas: {len(templates)}")
