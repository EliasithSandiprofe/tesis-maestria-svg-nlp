import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.inference import predict_attributes
from svg_engine.engine import SVGEngine

def generar_diseno(prompt: str):

    resultado = predict_attributes(prompt)

    atributos_nlp = resultado["atributos"]

    atributos_svg = {
        "color_camiseta": atributos_nlp["color"],
        "figura": atributos_nlp["elemento"],
        "color_figura": atributos_nlp["color"],
        "texto": "",
        "estilo": atributos_nlp["estilo"]
    }

    engine = SVGEngine()

    svg = engine.generar_svg(atributos_svg)

    engine.guardar_archivo(
        svg,
        "output/camiseta_generada.svg"
    )

    return resultado