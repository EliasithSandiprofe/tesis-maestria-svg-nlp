class NLPSVGMapper:


    def __init__(self):

        self.colores = {
            "negro":"black",
            "negra":"black",
            "rojo":"red",
            "azul":"blue",
            "blanco":"white"
        }


        self.figuras = {

            "estrella":"estrella",
            "guitarra":"guitarra",
            "circulo":"circulo",

            # temporal hasta crear dragon.svg
            "dragón":"defecto",
            "dragon":"defecto"
        }



    def convertir(self, resultado_nlp):

        atributos = {

            "color_camiseta":"white",
            "figura":"defecto",
            "color_figura":"black",
            "texto":"",
            "estilo":"moderno"

        }


        sustantivos = resultado_nlp["sustantivos"]
        adjetivos = resultado_nlp["adjetivos"]


        for adj in adjetivos:

            if adj in ["negro","negra"]:
                atributos["color_camiseta"]="black"



        for palabra in sustantivos:

            if palabra in self.figuras:

                atributos["figura"]=self.figuras[palabra]
                atributos["texto"]=palabra



        for adj in adjetivos:

            if adj in self.colores:

                atributos["color_figura"]=self.colores[adj]



        if "minimalista" in adjetivos:

            atributos["estilo"]="moderno"



        return atributos