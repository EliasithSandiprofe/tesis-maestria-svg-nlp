"""
model_config.py
---------------
Configuración centralizada del modelo Multi-Head DistilBERT SVG-NLP.

El número de clases de cada tarea se carga automáticamente desde los
``LabelEncoder`` serializados en ``modelos/label_encoders/``, evitando
valores hardcodeados que puedan quedar desincronizados con los datos reales.

Clases
------
SVGModelConfig
    Contiene todos los hiperparámetros y rutas necesarias para instanciar,
    entrenar y serializar el modelo.
"""

from __future__ import annotations

from pathlib import Path

import joblib


# Orden canónico de las tareas (consistente con el pipeline de preprocesamiento)
TASK_NAMES: list[str] = ["color", "estilo", "elemento", "posicion"]


class SVGModelConfig:
    """Configuración del modelo MultiTaskDistilBERT SVG-NLP.

    Attributes
    ----------
    model_name : str
        Identificador del modelo base en Hugging Face Hub.
        Por defecto ``"distilbert-base-uncased"``.
    dropout : float
        Probabilidad de dropout aplicado sobre el embedding [CLS] antes de
        las cabezas de clasificación.
    num_classes : dict[str, int]
        Número de clases por tarea.  Cargado automáticamente desde los
        ``LabelEncoder`` serializados al usar :meth:`from_project`.
    task_names : list[str]
        Nombres de las tareas en el mismo orden que ``num_classes``.
    encoders_dir : Path
        Directorio con los encoders ``.pkl``.
    tokenizer_dir : Path
        Directorio donde está serializado ``DistilBertTokenizerFast``.
    model_save_dir : Path | None
        Directorio reservado para guardar los checkpoints del modelo
        durante la Fase D de entrenamiento.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float,
        num_classes: dict[str, int],
        encoders_dir: Path,
        tokenizer_dir: Path,
        model_save_dir: Path | None = None,
    ) -> None:
        self.model_name    = model_name
        self.dropout       = dropout
        self.num_classes   = num_classes
        self.task_names    = list(num_classes.keys())
        self.encoders_dir  = Path(encoders_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.model_save_dir = Path(model_save_dir) if model_save_dir else None

    # ── Constructor alternativo ───────────────────────────────────────────────

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.3,
    ) -> "SVGModelConfig":
        """Crea la configuración cargando el número de clases desde disco.

        Lee automáticamente los ``LabelEncoder`` serializados en
        ``modelos/label_encoders/`` y extrae el atributo ``classes_`` de
        cada uno para obtener el número de clases de forma dinámica.

        Parameters
        ----------
        project_root : str | Path
            Ruta raíz del proyecto (contiene ``dataset/`` y ``modelos/``).
        model_name : str
            Identificador del modelo base en Hugging Face.
        dropout : float
            Probabilidad de dropout (por defecto 0.3).

        Returns
        -------
        SVGModelConfig
            Instancia completamente configurada.

        Raises
        ------
        FileNotFoundError
            Si algún archivo ``.pkl`` de encoder no existe en ``encoders_dir``.
        """
        project_root  = Path(project_root)
        encoders_dir  = project_root / "modelos" / "label_encoders"
        tokenizer_dir = project_root / "modelos" / "tokenizer"
        model_save_dir = project_root / "modelos" / "checkpoints"

        num_classes: dict[str, int] = {}
        for task in TASK_NAMES:
            encoder_path = encoders_dir / f"{task}_encoder.pkl"
            if not encoder_path.exists():
                raise FileNotFoundError(
                    f"Encoder no encontrado: {encoder_path}\n"
                    "Ejecuta primero run_preprocessing.py (Fase B)."
                )
            le = joblib.load(encoder_path)
            num_classes[task] = int(len(le.classes_))

        return cls(
            model_name=model_name,
            dropout=dropout,
            num_classes=num_classes,
            encoders_dir=encoders_dir,
            tokenizer_dir=tokenizer_dir,
            model_save_dir=model_save_dir,
        )

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        nc_str = ", ".join(f"{k}: {v}" for k, v in self.num_classes.items())
        lines = [
            "SVGModelConfig(",
            f"  model_name     = '{self.model_name}'",
            f"  dropout        = {self.dropout}",
            f"  num_classes    = {{{nc_str}}}",
            f"  encoders_dir   = '{self.encoders_dir}'",
            f"  tokenizer_dir  = '{self.tokenizer_dir}'",
            f"  model_save_dir = '{self.model_save_dir}'",
            ")",
        ]
        return "\n".join(lines)
