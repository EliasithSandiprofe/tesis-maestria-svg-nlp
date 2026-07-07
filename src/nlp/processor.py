"""
processor.py
Módulo de procesamiento NLP para prompts en español.
Utiliza el modelo es_core_news_sm de spaCy.
"""
import spacy


class NLPProcessor:
    """Procesa prompts en español y extrae información lingüística."""

    def __init__(self, modelo: str = "es_core_news_sm"):
        self.nlp = spacy.load(modelo)

    def procesar(self, prompt: str) -> dict:
        """
        Procesa un prompt en español y devuelve un diccionario con:
        - prompt_original: texto de entrada sin modificar
        - tokens: lista de todas las palabras/símbolos
        - lemas: forma base de cada token (sin stopwords ni puntuación)
        - entidades: entidades nombradas detectadas (texto + etiqueta)
        - sustantivos: tokens cuya categoría gramatical es sustantivo
        - adjetivos: tokens cuya categoría gramatical es adjetivo
        """
        doc = self.nlp(prompt)

        tokens = [token.text for token in doc]

        lemas = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]

        entidades = [
            {"texto": ent.text, "etiqueta": ent.label_}
            for ent in doc.ents
        ]

        sustantivos = [
            token.text for token in doc if token.pos_ == "NOUN"
        ]

        adjetivos = [
            token.text for token in doc if token.pos_ == "ADJ"
        ]

        return {
            "prompt_original": prompt,
            "tokens": tokens,
            "lemas": lemas,
            "entidades": entidades,
            "sustantivos": sustantivos,
            "adjetivos": adjetivos,
        }
