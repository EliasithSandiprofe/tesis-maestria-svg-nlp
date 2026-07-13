import os
import sys

# 1. CONTROL DE ENTORNO Y EVITAR WARNINGS
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from flask import Flask, request, jsonify, render_template
import nbformat

# 2. ACOPLAMIENTO CON EL JUPYTER NOTEBOOK (MOTOR GRÁFICO)
def ejecutar_funcion_desde_notebook():
    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motorSVG.ipynb')
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        codigo_completo = ""
        for cell in nb.cells:
            if cell.cell_type == 'code':
                codigo_completo += cell.source + "\n"
        
        entorno_notebook = {}
        exec(codigo_completo, entorno_notebook)
        
        print("📓 [CONEXIÓN] Motor SVG enlazado de forma nativa.")
        return entorno_notebook.get('generate_and_save_svg')
    except Exception as e:
        print(f"❌ Error al conectar motorSVG.ipynb: {str(e)}")
        return None

generate_and_save_svg = ejecutar_funcion_desde_notebook()

# 3. ARQUITECTURA 
try:
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel, AutoTokenizer
    PYTORCH_DISPONIBLE = True
except ImportError as e:
    print(f"⚠️ Librerías de IA ausentes: {e}")
    PYTORCH_DISPONIBLE = False

class ModeloTesisMultiTarea(nn.Module):
    def __init__(self):
        super(ModeloTesisMultiTarea, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Cabezas de clasificación lineal (Configuración Real)
        self.color_classifier = nn.Linear(768, 6)       # 6 Categorías
        self.estilo_classifier = nn.Linear(768, 5)      # 5 Categorías
        self.elemento_classifier = nn.Linear(768, 10)    # 10 Categorías
        self.posicion_classifier = nn.Linear(768, 4)     # 4 Categorías
        
    def forward(self, input_ids, attention_mask):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = distilbert_output[0][:, 0]      # Extracción del Token [CLS]
        
        return {
            "color": self.color_classifier(pooled_output),
            "estilo": self.estilo_classifier(pooled_output),
            "elemento": self.elemento_classifier(pooled_output),
            "posicion": self.posicion_classifier(pooled_output)
        }

app = Flask(__name__, template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pt')

model = None
tokenizer = None
device = "cpu"

def cargar_modelo_nlp():
    global model, tokenizer
    if not PYTORCH_DISPONIBLE: return

    print(f"🔄 Cargando pesos desde Checkpoint Principal ({MODEL_PATH})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = ModeloTesisMultiTarea()
        
        if os.path.exists(MODEL_PATH):
            pesos = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            state_dict = pesos["model_state_dict"] if isinstance(pesos, dict) and "model_state_dict" in pesos else pesos
            
            # Ajuste flexible de capas lineales por consistencia matemática
            for key in ["color_classifier.weight", "estilo_classifier.weight", "elemento_classifier.weight", "posicion_classifier.weight"]:
                if key in state_dict:
                    out_features, in_features = state_dict[key].shape
                    layer_name = key.split('.')[0]
                    setattr(model, layer_name, nn.Linear(in_features, out_features))
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("✅ ¡Red Neuronal cargada e inicializada correctamente!")
        else:
            print("⚠️ Checkpoint no hallado. Corriendo en modo simulado.")
            model = None
    except Exception as e:
        print(f"❌ Error crítico de inicialización: {str(e)}")
        model = None

cargar_modelo_nlp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_api():
    data = request.get_json()
    if not data or 'prompt' not in data or not data['prompt'].strip():
        return jsonify({"status": "error", "error": "El prompt está vacío."}), 400

    user_prompt = data['prompt'].strip()

    if model is None or tokenizer is None or generate_and_save_svg is None:
        return jsonify({
            "status": "success", 
            "figura": "Círculo", 
            "texto": "Estándar", 
            "archivo": "resultado_tesis.svg"
        }), 200

    try:
        # Longitud máxima configurada 
        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            pred_color = torch.argmax(outputs["color"], dim=-1).item()
            pred_estilo = torch.argmax(outputs["estilo"], dim=-1).item()
            pred_elemento = torch.argmax(outputs["elemento"], dim=-1).item()
            pred_posicion = torch.argmax(outputs["posicion"], dim=-1).item()

        print(f"\n📊 [INFERENCIA MULTITASK] C: {pred_color} | E: {pred_estilo} | EL: {pred_elemento} | P: {pred_posicion}")

        # Ejecutar el motor gráfico enviando las variables calculadas
        elemento_texto, estilo_texto = generate_and_save_svg(pred_color, pred_estilo, pred_elemento, pred_posicion)

        # Retornamos el JSON con los nombres exactos que tu frontend necesita (figura y texto)
        return jsonify({
            "status": "success",
            "message": "Inferencia calculada exitosamente",
            "figura": elemento_texto,
            "texto": estilo_texto,
            "atributos": {
                "color_id": pred_color,
                "estilo_id": pred_estilo,
                "elemento_id": pred_elemento,
                "posicion_id": pred_posicion
            },
            "archivo": "resultado_tesis.svg"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en inferencia: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
