"""
test_prompt.py
Prueba simple del NLPProcessor con un prompt de ejemplo.
"""
import sys
import os

# Permite ejecutar este archivo directamente desde src/nlp/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp.processor import NLPProcessor


def main():
    prompt = "Quiero una camiseta negra con un dragón rojo en estilo minimalista"

    print("=" * 60)
    print("PRUEBA NLPProcessor")
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
