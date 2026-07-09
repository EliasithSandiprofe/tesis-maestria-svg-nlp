# svg_generator.py
import os

# --- CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
# Detecta la carpeta exacta donde está corriendo este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Asegura la ruta directa hacia static/mocks dentro de tu proyecto Flask
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'mocks')
FILENAME = 'resultado_tesis.svg'

# Forzar la creación de la carpeta mocks si no existe en el Finder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- PLANTILLAS GRÁFICAS SVG ---

# Plantilla para Círculo (Clase 0)
SVG_CIRCLE_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="300" height="300">
  <!-- Círculo Verde Suave (#A5D6A7) con borde verde oscuro -->
  <circle cx="50" cy="50" r="45" fill="#A5D6A7" stroke="#2E7D32" stroke-width="3" />
</svg>"""

# Plantilla para Cuadrado (Clase 1)
SVG_SQUARE_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="300" height="300">
  <!-- Cuadrado Azul Suave (#90CAF9) con borde azul oscuro -->
  <rect x="5" y="5" width="90" height="90" rx="10" fill="#90CAF9" stroke="#1565C0" stroke-width="3" />
</svg>"""

# Plantilla de Contingencia / Error (Si llega un ID desconocido)
SVG_ERROR_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="300" height="300">
  <rect x="5" y="5" width="90" height="90" rx="10" fill="#EF9A9A" stroke="#B71C1C" stroke-width="2" />
  <text x="50" y="55" font-family="Arial" font-size="12" fill="#B71C1C" text-anchor="middle">ERROR</text>
  <text x="50" y="70" font-family="Arial" font-size="9" fill="#B71C1C" text-anchor="middle">Clase Inválida</text>
</svg>"""


# --- FUNCIÓN PRINCIPAL DE ESCRITURA ---
def generate_and_save_svg(class_id, verbose=True):
    """
    Toma el ID de la clase predicha (0 o 1), selecciona el SVG correspondiente
    y sobreescribe el archivo físico para actualizar la interfaz del frontend.
    """
    full_path = os.path.join(OUTPUT_FOLDER, FILENAME)

    # 1. Emparejar el ID con su respectivo bloque de diseño
    if class_id == 0:
        svg_content = SVG_CIRCLE_TEMPLATE
        figure_name = "Círculo"
    elif class_id == 1:
        svg_content = SVG_SQUARE_TEMPLATE
        figure_name = "Cuadrado"
    else:
        svg_content = SVG_ERROR_TEMPLATE
        figure_name = f"Desconocida (ID: {class_id})"

    # 2. Intentar escribir/sobreescribir el archivo en disco de forma atómica
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(svg_content.strip())
        
        if verbose:
            print(f"🎨 Motor SVG: Se sobreescribió con un '{figure_name}'")
            print(f"📌 Destino: {full_path}")
            
        return True, figure_name, full_path
    except Exception as e:
        if verbose:
            print(f"❌ Error crítico en motor SVG al intentar guardar: {e}")
        return False, str(e), None


# --- MODO DE PRUEBA INDEPENDIENTE ---
# Puedes correr únicamente este archivo ejecutando: python3 svg_generator.py
if __name__ == "__main__":
    print("⚡ Probando la generación directa de archivos SVG para la Tesis...")
    # Creamos un círculo por defecto para verificar que el Finder cree las rutas bien
    generate_and_save_svg(0, verbose=True)
