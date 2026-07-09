import os
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__, template_folder='template')

# 1. Ruta absoluta automática hacia tu modelo de 800 MB
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pt')

print("Cargando modelo YOLO en memoria (esto puede tardar unos segundos)...")
try:
    # Se carga una sola vez al encender el servidor para no saturar la RAM
    model = YOLO(MODEL_PATH)
    print("¡Éxito! El modelo best_model.pt se cargó correctamente.")
except Exception as e:
    print(f"❌ Error crítico al cargar el archivo del modelo: {e}")
    model = None


# 2. Ruta para renderizar tu interfaz de usuario (Frontend)
@app.route('/')
def index():
    return render_template('index.html')


# 3. Endpoint API que procesará el Prompt e interactuará con el modelo
@app.route('/api/generate', methods=['POST'])
def generate_api():
    data = request.get_json()
    
    # Validaciones básicas de seguridad
    if not data or 'prompt' not in data or not data['prompt'].strip():
        return jsonify({
            "status": "error",
            "error": "El prompt no puede estar vacío."
        }), 400

    user_prompt = data['prompt'].strip()

    if model is None:
        return jsonify({
            "status": "error",
            "error": "El modelo de IA no está disponible o no se cargó correctamente en el servidor."
        }), 500

    try:
        # 🚀 INFERENCIA REAL CON TU MODELO
        # Nota: YOLO por defecto procesa imágenes/video. Si tu modelo fue entrenado 
        # para otra tarea, asegúrate de pasarle el tipo de dato que espera.
        results = model(user_prompt)
        
        # Variables para estructurar lo que tu modelo detecte
        figura_detectada = "No identificada"
        
        # Procesamos los resultados del objeto YOLO
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                # Extraemos el ID de la primera clase detectada por la red
                clase_id = int(result.boxes[0].cls[0])
                # Mapeamos el ID al nombre real configurado en tu entrenamiento
                figura_detectada = model.names[clase_id]

        # 🤝 AQUÍ SE CONECTARÁ CON EL MOTOR DE IVÁN POSTERIORMENTE
        # Por ahora dejamos un nombre de archivo estándar de salida
        nombre_archivo_svg = "resultado_tesis.svg"

        return jsonify({
            "status": "success",
            "message": "Inferencia procesada con éxito por YOLO",
            "atributos": {
                "figura": figura_detectada,
                "prompt_original": user_prompt
            },
            "archivo": nombre_archivo_svg
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Falla durante la inferencia del modelo: {str(e)}"
        }), 500


# 4. Arranque del servidor local
if __name__ == '__main__':
    # Usamos el puerto 5000 estándar de Flask
    app.run(debug=True, port=5000)
