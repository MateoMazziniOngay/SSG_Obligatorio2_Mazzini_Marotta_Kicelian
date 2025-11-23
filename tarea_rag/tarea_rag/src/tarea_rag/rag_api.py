"""
API Flask para el sistema RAG de análisis de ventas
"""
import sys
from pathlib import Path

# Agregar el directorio src al path para imports
# rag_api.py está en: src/tarea_rag/rag_api.py
# Necesitamos: src/
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback

from tarea_rag.config import STATIC_FOLDER
from tarea_rag.data_loader import DataLoader
from tarea_rag.vectorstore import VectorStoreManager
from tarea_rag.query_processor import QueryProcessor

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)  # Permitir CORS para el frontend

# Inicializar sistema RAG
print("\n" + "="*60)
print("🚀 Inicializando Sistema RAG")
print("="*60)

# 1. Cargar datos
data_loader = DataLoader()
df_ventas_full = data_loader.load_data()

# 2. Crear documentos
documents = data_loader.create_documents()

# 3. Crear vectorstore con embeddings
vectorstore_manager = VectorStoreManager()
vectorstore_manager.create_vectorstore(documents)

# 4. Inicializar procesador de consultas
query_processor = QueryProcessor(vectorstore_manager)

print("="*60)
print("✅ Sistema RAG listo para recibir consultas")
print("="*60 + "\n")

@app.route('/')
def index():
    """Servir página principal"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Servir archivos estáticos (CSS, JS, etc.)"""
    return send_from_directory(app.static_folder, filename)

@app.route('/api/health', methods=['GET'])
def health():
    """Verificar que el servicio está funcionando"""
    return jsonify({"status": "ok", "message": "API funcionando correctamente"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas generales"""
    return jsonify(data_loader.get_stats())

@app.route('/api/query', methods=['POST'])
def query():
    """Procesar pregunta del usuario"""
    try:
        data = request.get_json()
        pregunta = data.get('pregunta', '')
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        # Procesar consulta usando QueryProcessor
        resultado = query_processor.process_query(pregunta)
        
        # Si hay error, devolver con status 500
        if "error" in resultado:
            return jsonify(resultado), 500
        
        return jsonify(resultado)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Servidor RAG iniciado en http://localhost:5001")
    print("="*60 + "\n")
    app.run(debug=True, port=5001, host='0.0.0.0')
