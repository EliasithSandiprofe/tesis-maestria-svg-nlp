# -*- coding: utf-8 -*-
"""
Módulo: svg_engine/engine.py
------------------------------------------------------------
Proyecto de Integración - Maestría en Inteligencia Artificial
Autor de la Tarea: Iván

Este módulo contiene la clase SVGEngine, encargada de consolidar 
el lienzo vectorial (plantilla de una camiseta) con los atributos 
de diseño extraídos del modelo NLP.

Diseñado para ser importado directamente en Flask:
    from svg_engine.engine import SVGEngine
"""

import os

class SVGEngine:
    """
    Motor gráfico especializado en la generación de archivos SVG 
    mediante inyección dinámica de propiedades vectoriales en plantillas base.
    """
    
    def __init__(self, ancho=500, alto=500):
        self.ancho = ancho
        self.alto = alto
        
        # Diccionario interno de iconos vectoriales predefinidos (Paths SVG simplificados)
        # Todos los paths se escalan para encajar en el centro de la camiseta sin depender de la "posición"
        self._figuras_disponibles = {
            "guitarra": (
                '<path d="M230 180 L230 300 M225 300 A15 15 0 1 0 255 300 A15 15 0 1 0 225 300 '
                'M240 180 L240 220" stroke="{color}" stroke-width="4" fill="none" />'
                '<ellipse cx="240" cy="300" rx="18" ry="25" fill="{color}" />'
                '<rect x="237" y="160" width="6" height="60" fill="#8B5A2B" />'
                '<path d="M234 150 L246 150 L243 160 L237 160 Z" fill="#D2B48C" />'
            ),
            "estrella": (
                '<polygon points="250,190 265,225 300,225 270,245 285,280 250,255 215,280 230,245 200,225 235,225" '
                'fill="{color}" stroke="#000000" stroke-width="2" />'
            ),
            "circulo": (
                '<circle cx="250" cy="240" r="40" fill="{color}" stroke="#000000" stroke-width="2" />'
            ),
            "defecto": (
                '<rect x="225" y="215" width="50" height="50" fill="{color}" stroke="#000000" stroke-width="2" />'
            )
        }

    def _obtener_plantilla_camiseta(self):
        """
        Retorna la base XML del SVG que dibuja el contorno de una camiseta de manera proporcional.
        Utiliza marcadores de posición ({variable}) para la inyección dinámica.
        """
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" width="{ancho}" height="{alto}">
    <style>
        .texto-diseno {{
            font-family: 'Impact', 'Arial Black', sans-serif;
            font-size: 28px;
            letter-spacing: 2px;
            text-anchor: middle;
            fill: {color_texto};
        }}
        .estilo-vintage {{
            filter: sepia(0.4) contrast(0.9);
            opacity: 0.85;
        }}
    </style>

    <g class="{clase_estilo}">
        <path d="M 150 100 
                 L 200 60 
                 L 250 80 
                 L 300 60 
                 L 350 100 
                 L 390 150 
                 L 340 190 
                 L 320 170 
                 L 320 420 
                 L 180 420 
                 L 180 170 
                 L 160 190 
                 Z" 
              fill="{color_camiseta}" 
              stroke="#2D3142" 
              stroke-width="6" 
              stroke-linejoin="round" />

        {capa_figura}

        <text x="250" y="360" class="texto-diseno">{texto}</text>
    </g>
</svg>
"""

    def generar_svg(self, atributos):
        """
        Genera una cadena de texto SVG válida sustituyendo dinámicamente
        los atributos validados por el motor en la plantilla base.
        """
        # Extraer parámetros seguros con valores de contingencia (fallbacks)
        color_camiseta = atributos.get("color_camiseta", "white")
        figura_solicitada = atributos.get("figura", "defecto").lower()
        color_figura = atributos.get("color_figura", "black")
        texto = atributos.get("texto", "").upper()
        estilo = atributos.get("estilo", "moderno").lower()

        # Validación interna de figuras disponibles
        render_figura = self._figuras_disponibles.get(figura_solicitada, self._figuras_disponibles["defecto"])
        # Inyección dinámica del color del elemento dentro de su propio path
        capa_figura_formateada = render_figura.format(color=color_figura)

        # Aplicación de estilos visuales condicionales basados en el atributo 'estilo'
        clase_estilo = "estilo-vintage" if estilo == "vintage" else ""
        color_texto = "#333333" if color_camiseta in ["white", "yellow", "#FFFFFF"] else "#FFFFFF"

        # Cargar plantilla base e inyectar las variables procesadas
        plantilla = self._obtener_plantilla_camiseta()
        svg_final = plantilla.format(
            ancho=self.ancho,
            alto=self.alto,
            color_camiseta=color_camiseta,
            capa_figura=capa_figura_formateada,
            texto=texto,
            color_texto=color_texto,
            clase_estilo=clase_estilo
        )

        return svg_final

    def guardar_archivo(self, svg_contenido, ruta_salida="camiseta.svg"):
        """
        Método utilitario para escribir el contenido string a un archivo físico.
        """
        try:
            # Asegurar la creación de la ruta de salida si incluye subcarpetas
            directorio = os.path.dirname(ruta_salida)
            if directorio and not os.path.exists(directorio):
                os.makedirs(directorio)

            with open(ruta_salida, "w", encoding="utf-8") as archivo:
                archivo.write(svg_contenido)
            print(f"[SVGEngine] Guardado con éxito en: {ruta_salida}")
            return True
        except Exception as e:
            print(f"[SVGEngine] Error al guardar archivo: {e}")
            return False