"""
prompt_generator.py
Construye los textos de prompt aplicando las plantillas de templates.yaml
sobre cada combinación de etiquetas.
"""
import random
from pathlib import Path

# Claves de atributo requeridas en cada combinación
_CLAVES_REQUERIDAS: tuple[str, ...] = ("color", "estilo", "elemento", "posicion")


class PromptGenerator:
    """Genera prompts en español a partir de plantillas y combinaciones de etiquetas.

    La selección de plantilla por fila es determinista gracias a ``seed``:
    se crea una instancia ``random.Random`` privada que no afecta el estado
    global del generador de Python.
    """

    def __init__(self, templates: list[str], seed: int = 42) -> None:
        """
        Parámetros
        ----------
        templates : list[str]
            Plantillas cargadas desde templates.yaml.
            Cada plantilla debe contener las variables
            {color}, {estilo}, {elemento}, {posicion}.
        seed : int, optional
            Semilla para la selección reproducible de plantillas. Por defecto 42.

        Lanza
        -----
        ValueError
            Si la lista de plantillas está vacía.
        """
        if not templates:
            raise ValueError("La lista de plantillas no puede estar vacía.")

        self.templates = list(templates)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def generate_prompt(self, combination: dict[str, str]) -> str:
        """Selecciona una plantilla al azar y la rellena con los atributos dados.

        Parámetros
        ----------
        combination : dict[str, str]
            Diccionario con claves color, estilo, elemento, posicion.

        Retorna
        -------
        str
            Texto del prompt generado.

        Lanza
        -----
        KeyError
            Si alguna clave requerida no está presente en ``combination``.
        """
        faltantes = [k for k in _CLAVES_REQUERIDAS if k not in combination]
        if faltantes:
            raise KeyError(
                f"La combinación no contiene las claves requeridas: {faltantes}"
            )

        plantilla: str = self._rng.choice(self.templates)
        return plantilla.format(**combination)

    def generate_dataset(
        self, combinations: list[dict[str, str]]
    ) -> list[dict]:
        """Genera un prompt para cada combinación y arma los registros del dataset.

        Parámetros
        ----------
        combinations : list[dict[str, str]]
            Lista de 2 000 dicts con claves color, estilo, elemento, posicion.

        Retorna
        -------
        list[dict]
            Lista de dicts con columnas: id, prompt, color, estilo, elemento, posicion.
            El id inicia en 1.

        Lanza
        -----
        ValueError
            Si la lista de combinaciones está vacía.
        """
        if not combinations:
            raise ValueError("La lista de combinaciones no puede estar vacía.")

        dataset: list[dict] = []
        for idx, combo in enumerate(combinations, start=1):
            registro = {
                "id": idx,
                "prompt": self.generate_prompt(combo),
                "color": combo["color"],
                "estilo": combo["estilo"],
                "elemento": combo["elemento"],
                "posicion": combo["posicion"],
            }
            dataset.append(registro)

        return dataset


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from config_loader import ConfigLoader
    from combination_generator import CombinationGenerator

    _BASE = Path(__file__).resolve().parents[2] / "dataset" / "config"

    loader = ConfigLoader(
        labels_path=_BASE / "labels.yaml",
        templates_path=_BASE / "templates.yaml",
    )
    labels, templates = loader.load_all()

    combinaciones = CombinationGenerator(
        labels=labels, target_size=2_000, seed=42
    ).generar()

    pg = PromptGenerator(templates=templates, seed=42)
    dataset = pg.generate_dataset(combinaciones)

    print(f"Total prompts: {len(dataset)}")
    print("\nPrimeros 5 registros:")
    for registro in dataset[:5]:
        print(f"  id={registro['id']} | {registro['prompt']}")
        print(
            f"         color={registro['color']}  estilo={registro['estilo']}"
            f"  elemento={registro['elemento']}  posicion={registro['posicion']}"
        )
