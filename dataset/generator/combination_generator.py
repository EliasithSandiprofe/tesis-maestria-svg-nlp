"""
combination_generator.py
Genera todas las combinaciones posibles de etiquetas (color × estilo ×
elemento × posicion) y las completa hasta alcanzar el tamaño de dataset
deseado de forma reproducible.
"""
import itertools
import random
from pathlib import Path

# Categorías esperadas en el diccionario de labels
_CATEGORIAS: tuple[str, ...] = ("color", "estilo", "elemento", "posicion")


class CombinationGenerator:
    """Produce combinaciones balanceadas de etiquetas para el dataset.

    Pasos internos
    --------------
    1. ``generar_base()``  →  producto cartesiano completo (1 200 entradas).
    2. ``completar()``     →  añade entradas extra de forma reproducible hasta
                              alcanzar ``target_size``.
    3. ``generar()``       →  ejecuta ambos pasos y devuelve la lista final.
    """

    def __init__(
        self,
        labels: dict[str, list[str]],
        target_size: int = 2_000,
        seed: int = 42,
    ) -> None:
        """
        Parámetros
        ----------
        labels : dict[str, list[str]]
            Diccionario con las listas de clases por categoría.
            Debe contener las claves: color, estilo, elemento, posicion.
        target_size : int, optional
            Número total de combinaciones a producir. Por defecto 2 000.
        seed : int, optional
            Semilla para reproducibilidad al seleccionar las entradas extra.
            Por defecto 42.

        Lanza
        -----
        ValueError
            Si ``labels`` no contiene alguna de las categorías requeridas o
            si ``target_size`` es menor que el número de combinaciones base.
        """
        faltantes = [c for c in _CATEGORIAS if c not in labels]
        if faltantes:
            raise ValueError(
                f"El diccionario de labels no contiene las categorías: {faltantes}"
            )

        self.labels = labels
        self.target_size = target_size
        self.seed = seed

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def generar_base(self) -> list[dict[str, str]]:
        """Genera el producto cartesiano completo de todas las etiquetas.

        Retorna
        -------
        list[dict[str, str]]
            Lista de dicts con claves color, estilo, elemento, posicion.
            Contiene exactamente ``len(color) × len(estilo) × len(elemento)
            × len(posicion)`` entradas (1 200 con la configuración estándar).
        """
        combinaciones = [
            {
                "color": color,
                "estilo": estilo,
                "elemento": elemento,
                "posicion": posicion,
            }
            for color, estilo, elemento, posicion in itertools.product(
                self.labels["color"],
                self.labels["estilo"],
                self.labels["elemento"],
                self.labels["posicion"],
            )
        ]
        return combinaciones

    def completar(
        self, base: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Añade entradas adicionales para alcanzar ``target_size``.

        Las entradas extra se seleccionan mediante muestreo aleatorio con
        reemplazo sobre la lista base, usando ``self.seed`` para garantizar
        reproducibilidad.

        Parámetros
        ----------
        base : list[dict[str, str]]
            Lista generada por ``generar_base()``.

        Retorna
        -------
        list[dict[str, str]]
            Lista de exactamente ``target_size`` dicts.

        Lanza
        -----
        ValueError
            Si ``target_size`` es menor que el tamaño de la lista base.
        """
        n_base = len(base)
        if self.target_size < n_base:
            raise ValueError(
                f"target_size ({self.target_size}) no puede ser menor que el "
                f"número de combinaciones base ({n_base})."
            )

        faltan = self.target_size - n_base
        if faltan == 0:
            return list(base)

        rng = random.Random(self.seed)
        extra = rng.choices(base, k=faltan)
        return list(base) + extra

    def generar(self) -> list[dict[str, str]]:
        """Ejecuta el pipeline completo: base → completar → resultado final.

        Retorna
        -------
        list[dict[str, str]]
            Lista de ``target_size`` dicts con claves
            color, estilo, elemento, posicion.
        """
        base = self.generar_base()
        return self.completar(base)


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from config_loader import ConfigLoader

    _BASE = Path(__file__).resolve().parents[2] / "dataset" / "config"

    loader = ConfigLoader(
        labels_path=_BASE / "labels.yaml",
        templates_path=_BASE / "templates.yaml",
    )
    labels = loader.load_labels()

    gen = CombinationGenerator(labels=labels, target_size=2_000, seed=42)

    base = gen.generar_base()
    combinaciones = gen.generar()

    print(f"Combinaciones base : {len(base)}")
    print(f"Total generado     : {len(combinaciones)}")
    print("\nPrimeras 5 combinaciones:")
    for i, c in enumerate(combinaciones[:5], start=1):
        print(f"  {i}. {c}")
