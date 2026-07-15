
import sys
import os

# agregar raíz del proyecto al path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


from mvp.nlp_svg_mapper import NLPSVGMapper
from svg_engine.engine import SVGEngine

#resultado_nlp = {

    #"sustantivos":[
    #    "camiseta",
    #    "dragón",
    #    "estilo"
    #],

    #"adjetivos":[
    #    "negra",
   #     "rojo",
  #      "minimalista"
  #  ]

#}

NLPProcessor.process(prompt)

mapper = NLPSVGMapper()


atributos = mapper.convertir(resultado_nlp)


print("\nATRIBUTOS SVG")
print(atributos)



engine = SVGEngine()


svg = engine.generar_svg(
    atributos
)


engine.guardar_archivo(
    svg,
    "mvp/resultado.svg"
)