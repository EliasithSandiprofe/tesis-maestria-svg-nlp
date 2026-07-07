import os
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

app = Flask(__name__)

# Configuración de rutas para archivos estáticos/mocks
MOCKS_DIR = os.path.join(app.root_path, 'static', 'mocks')
os.makedirs(MOCKS_DIR, exist_ok=True)

# Crear un SVG de prueba en la carpeta de mocks si no existe
SVG_MOCK_PATH = os.path.join(MOCKS_DIR, 'camiseta.svg')
if not os.path.exists(SVG_MOCK_PATH):
    with open(SVG_MOCK_PATH, 'w', encoding='utf-8') as f:
        f.write('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="300" height="300">
            <rect width="100" height="100" fill="#23272A"/>
            <circle cx="50" cy="45" r="25" fill="#7289DA"/>
            <text x="50" y="85" font-family="Arial" font-size="8" fill="white" text-anchor="middle">Mock: Samurai Dragon</text>
        </svg>''')


# 1. Navegación / Ruta Principal
@app.route('/')
def index():
    return render_template('index.html')


# 2. API - Procesamiento de Texto (Simulado)
@app.route('/api/generate', methods=['POST'])
def generate_svg():
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        # ---- SECCIÓN DE MOCKS (Simulación de NLP y SVG Engine) ----
        # Aquí es donde el NLP extraerá atributos en el futuro
        atributos_simulados = {
            "figura": "dragón",
            "texto": "Samurai",
            "prompt_original": prompt
        }
        
        # Nombre del archivo que se "generó"
        archivo_simulado = "camiseta.svg"
        # -----------------------------------------------------------

        return jsonify({
            "status": "success",
            "message": "SVG generado exitosamente (Mock)",
            "atributos": atributos_simulados,
            "archivo": archivo_simulado
        }), 200

    except Exception as e:
        # Manejo de errores interno
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500


# 3. Descarga del archivo generado
@app.route('/download/<filename>')
def download_file(filename):
    # Validación básica de seguridad para evitar Path Traversal
    if filename != "camiseta.svg":
        abort(404, description="Archivo no encontrado")
        
    try:
        return send_from_directory(MOCKS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="El archivo solicitado no existe en el servidor")


# 4. Manejadores de Errores Globales (HTML)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error=e.description), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index.html', error="Ocurrió un error inesperado en el servidor."), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
