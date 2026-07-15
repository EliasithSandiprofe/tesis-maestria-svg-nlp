import os
import sys
import shutil 
from pathlib import Path

# ==========================================
# 1. RESOLVER RUTAS 
# ==========================================
dir_mvp = Path(__file__).resolve().parent
dir_raiz = dir_mvp.parent

# Prioridad a la raíz en el PATH para importar tus módulos locales de la tesis
sys.path = [str(dir_raiz), str(dir_mvp)] + [p for p in sys.path if p not in (str(dir_raiz), str(dir_mvp))]

from flask import Flask, request, jsonify, render_template, send_from_directory

# ==========================================
# 2. IMPORTACIONES 
# ==========================================
try:
    from src.nlp.inference import predict_attributes
    import svg_generator 
    print("✅ Módulos de tesis vinculados con éxito.")
except ImportError as e:
    print(f"❌ Error al cargar los módulos de tesis: {e}")
    sys.exit(1)

app = Flask(__name__, 
            template_folder=os.path.join(dir_mvp, 'templates'),
            static_folder=os.path.join(dir_mvp, 'static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_api():
    data = request.get_json()
    if not data or 'prompt' not in data or not data['prompt'].strip():
        return jsonify({"status": "error", "error": "El prompt está vacío."}), 400

    user_prompt = data['prompt'].strip()

    try:
        print(f"\n📥 [PETICIÓN] Analizando prompt en MVP: '{user_prompt}'")

        # ---------------------------------------------------------
        # PASO 1: Inferencia con tu modelo de Inteligencia Artificial
        # ---------------------------------------------------------
        resultado_nlp = predict_attributes(user_prompt)
        print(f"📊 Atributos extraídos con éxito: {resultado_nlp}")

        # ---------------------------------------------------------
        # PASO 2: Ejecución de tu generador gráfico
        # ---------------------------------------------------------
        if hasattr(svg_generator, 'generar_svg'):
            svg_generator.generar_svg(resultado_nlp)
        elif hasattr(svg_generator, 'generate_svg'):
            svg_generator.generate_svg(resultado_nlp)

        # ---------------------------------------------------------
        # PASO 3:Forzar la creación del Archivo SVG
        # ---------------------------------------------------------
        nombre_archivo_fijo = "resultado_tesis.svg"
        ruta_destino_static = os.path.join(dir_mvp, 'static', nombre_archivo_fijo)
        
        # ELIMINAR ANTERIOR: Nos aseguramos de borrar el archivo viejo en static para evitar bloqueos de lectura/caché
        if os.path.exists(ruta_destino_static):
            try:
                os.remove(ruta_destino_static)
            except Exception as e:
                print(f"⚠️ [SISTEMA] No se pudo borrar el SVG anterior temporalmente: {e}")
        
        # Intentamos rastrear si el script original guardó algo en disco
        rutas_origen_posibles = [
            os.path.join(dir_mvp, nombre_archivo_fijo),
            os.path.join(dir_raiz, nombre_archivo_fijo)
        ]

        archivo_encontrado_y_movido = False
        for ruta_origen in rutas_origen_posibles:
            if os.path.exists(ruta_origen) and ruta_origen != ruta_destino_static:
                print(f"📦 [SISTEMA] Archivo detectado en {ruta_origen}. Moviendo a static...")
                shutil.move(ruta_origen, ruta_destino_static)
                archivo_encontrado_y_movido = True
                break

        # 🛑 Si no lo guardó tu script, lo creamos nosotros usando la inferencia de la IA
        if not archivo_encontrado_y_movido and not os.path.exists(ruta_destino_static):
            print(f"⚠️ [SISTEMA] 'svg_generator' no guardó el archivo en disco. Forzando escritura desde app.py...")
            
            atributos = resultado_nlp.get("atributos", {})
            color_ia = atributos.get("color", "rojo").lower()
            elemento_ia = atributos.get("elemento", "guitarra").lower()
            
            # 1. Diccionario para traducir colores al estándar web CSS/SVG
            mapeo_colores = {
                "rojo": "#dc2626", 
                "azul": "#2563eb", 
                "verde": "#16a34a", 
                "negro": "#111827", 
                "blanco": "#f3f4f6",
                "amarillo": "#ca8a04",
                "rosa": "#db2777"
            }
            color_hex = mapeo_colores.get(color_ia, "#4f46e5") # Indigo por defecto

            # 2. Diccionario de emojis dinámicos para los elementos de tu Tesis
            MAPA_ICONOS = {
                "guitarra": "🎸",
                "estrella": "⭐",
                "aguila": "🦅",
                "águila": "🦅",
                "corazon": "❤️",
                "corazón": "❤️",
                "rayo": "⚡",
                "calavera": "💀"
            }
            # Si no coincide con ninguno del mapa, colocamos un destello por defecto
            emoji_icono = MAPA_ICONOS.get(elemento_ia, "✨")

            elemento_str = str(elemento_ia).upper()

            # Construimos un lienzo SVG vectorizado con el fondo oscuro y la camiseta de color dinámico
            codigo_svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
    <!-- Fondo del lienzo (Fijo y oscuro para dar contraste elegante) -->
    <rect width="400" height="400" fill="#0f172a" rx="24"/>
    <circle cx="200" cy="200" r="140" fill="#000000" opacity="0.15"/>
    
    <!-- Representación de la Camisa - Se pinta dinámicamente con el color de tu modelo -->
    <path d="M 120,80 L 160,40 L 240,40 L 280,80 L 340,110 L 300,160 L 260,140 L 260,360 L 140,360 L 140,140 L 100,160 L 60,110 Z" 
          fill="{color_hex}" 
          stroke="white" 
          stroke-width="3"/>

    <!-- Atributo 'elemento' renderizado en el centro con su respectivo emoji dinámico -->
    <text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" 
          font-family="system-ui, -apple-system, sans-serif" font-weight="bold" font-size="28" fill="white">
        {emoji_icono} {elemento_str}
    </text>

    <!-- Atributo 'estilo' -->
    <text x="50%" y="58%" dominant-baseline="middle" text-anchor="middle" 
          font-family="system-ui, -apple-system, sans-serif" font-style="italic" font-size="16" fill="white" opacity="0.9">
        Estilo: {atributos.get("estilo", "N/A")}
    </text>

    <!-- Metadatos de Posición y Tesis -->
    <text x="50%" y="68%" dominant-baseline="middle" text-anchor="middle" 
          font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="white" opacity="0.7">
        Posición: {atributos.get("posicion", "centrado")}
    </text>

    <text x="50%" y="90%" dominant-baseline="middle" text-anchor="middle" 
          font-family="monospace" font-size="10" fill="white" opacity="0.4">
        Tesis MVP — resultado_tesis.svg
    </text>
</svg>
"""
            # Escribimos físicamente el archivo en mvp/static/
            with open(ruta_destino_static, "w", encoding="utf-8") as f:
                f.write(codigo_svg)
            print(f"✅ [SISTEMA] Archivo SVG generado y guardado con éxito en: {ruta_destino_static}")

        # ---------------------------------------------------------
        # PASO 4: Responder al Frontend (Mapeado exacto de campos)
        # ---------------------------------------------------------
        atributos = resultado_nlp.get("atributos", {})

        return jsonify({
            "status": "success",
            "message": "Inferencia completada.",
            "figura": atributos.get("elemento", "guitarra").upper(),
            "texto": atributos.get("estilo", "vintage").upper(),
            "color": atributos.get("color", "rojo").upper(),
            "posicion": atributos.get("posicion", "centrado").upper(),
            "archivo": nombre_archivo_fijo
        }), 200

    except Exception as e:
        import traceback
        print(f"❌ [ERROR CRÍTICO EN FLUJO]: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Endpoint dedicado para forzar la descarga del archivo al ordenador."""
    try:
        return send_from_directory(
            os.path.join(dir_mvp, 'static'), 
            filename, 
            as_attachment=True
        )
    except Exception as e:
        return f"Error: {e}", 404

if __name__ == '__main__':
    print(f"🚀 Servidor MVP listo. Directorio de tesis: {dir_raiz}")
    static_path = os.path.join(dir_mvp, 'static')
    if not os.path.exists(static_path):
        os.makedirs(static_path)
    app.run(debug=True, port=5000)
