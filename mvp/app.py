import os
import sys

# 1. 🛠️ TRUCOS DE ENTORNO (Antes de los imports pesados)
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from flask import Flask, request, jsonify, render_template

# 🔌 IMPORTAMOS TU GENERADOR DE SVG
# Esto conecta directamente el archivo svg_generator.py con Flask
try:
    from svg_generator import generate_and_save_svg
except ImportError:
    print("❌ Error: No se encontró svg_generator.py en la misma carpeta.")

# 2. 📦 INTENTO DE CARGA DE PYTORCH Y TRANSFORMERS
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    PYTORCH_DISPONIBLE = True
except ImportError as e:
    print(f"⚠️ Alerta de dependencias: {e}")
    PYTORCH_DISPONIBLE = False

app = Flask(__name__, template_folder='templates')

# Configuración de rutas locales del modelo (.pt)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pt')

# Variables globales para el modelo NLP
model = None
tokenizer = None
device = "cpu"

def cargar_modelo_nlp():
    global model, tokenizer
    if not PYTORCH_DISPONIBLE:
        print("❌ No se puede cargar DistilRoBERTa porque PyTorch no está enlazado.")
        return

    print(f"🔄 Cargando DistilRoBERTa en memoria RAM (CPU)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
        
        if os.path.exists(MODEL_PATH):
            pesos = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if isinstance(pesos, dict) and "state_dict" in pesos:
                model.load_state_dict(pesos["state_dict"])
            elif isinstance(pesos, dict):
                model.load_state_dict(pesos)
            else:
                model = pesos
                
            model.to(device)
            model.eval()
            print("✅ ¡Éxito! DistilRoBERTa de 800 MB cargado correctamente.")
        else:
            print(f"⚠️ Archivo no encontrado en: {MODEL_PATH}. Corriendo en contingencia.")
            
    except Exception as e:
        print(f"❌ Error crítico al cargar DistilRoBERTa: {str(e)}")
        model = None

cargar_modelo_nlp()


# 🏠 RUTA PRINCIPAL
@app.route('/')
def index():
    return render_template('index.html')


# 🚀 ENDPOINT AUTOMATIZADO CON EL GENERADOR DE SVG
@app.route('/api/generate', methods=['POST'])
def generate_api():
    data = request.get_json()
    if not data or 'prompt' not in data or not data['prompt'].strip():
        return jsonify({"status": "error", "error": "El prompt de texto no puede estar vacío."}), 400

    user_prompt = data['prompt'].strip()

    # --- CASO A: MODO CONTINGENCIA (Si falla PyTorch) ---
    if model is None or tokenizer is None:
        print("⚠️ Inferencia simulada + Generación real de SVG...")
        
        # Lógica simple por palabras clave para pruebas rápidas
        clase_predicha = 1 if "cuadrado" in user_prompt.lower() else 0
        
        # 🎨 LLAMADA AL GENERADOR DE IVÁN: Aquí se crea físicamente el archivo .svg
        exito, figura, ruta = generate_and_save_svg(clase_predicha, verbose=False)
        
        return jsonify({
            "status": "success",
            "message": "Procesado por motor de contingencia",
            "atributos": {
                "clase_id": clase_predicha,
                "figura_clasificada": figura,
                "prompt_original": user_prompt
            },
            "archivo": "resultado_tesis.svg" # Tu frontend buscará este archivo generado en tiempo real
        }), 200

    # --- CASO B: INFERENCIA REAL CON DISTILROBERTA ---
    try:
        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            clase_predicha = torch.argmax(predictions, dim=-1).item()

        # 🎨 LLAMADA AL GENERADOR DE IVÁN: Pasa la clase matemática real para reescribir el SVG
        exito, figura, ruta = generate_and_save_svg(clase_predicha, verbose=False)

        return jsonify({
            "status": "success",
            "message": "Inferencia de texto procesada por DistilRoBERTa",
            "atributos": {
                "clase_id": clase_predicha,
                "figura_clasificada": figura,
                "prompt_original": user_prompt
            },
            "archivo": "resultado_tesis.svg"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": f"Falla en la inferencia: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
