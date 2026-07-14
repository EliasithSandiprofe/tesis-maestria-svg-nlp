"""
main.py
Punto de entrada principal del sistema NLP para generación de SVG.
"""
import sys
import os

# Asegura que src/ esté en el path cuando se ejecuta desde la raíz
sys.path.insert(0, os.path.dirname(__file__))

from nlp.processor import NLPProcessor


def main():
    prompt = "Quiero una camiseta negra con un dragón rojo en estilo minimalista"

    print("=" * 60)
    print("SISTEMA NLP - Generador de diseños SVG de camisetas")
    print("=" * 60)

    procesador = NLPProcessor()
    resultado = procesador.procesar(prompt)

    print(f"\nPrompt original : {resultado['prompt_original']}")
    print(f"\nTokens          : {resultado['tokens']}")
    print(f"\nLemas           : {resultado['lemas']}")
    print(f"\nEntidades       : {resultado['entidades']}")
    print(f"\nSustantivos     : {resultado['sustantivos']}")
    print(f"\nAdjetivos       : {resultado['adjetivos']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
