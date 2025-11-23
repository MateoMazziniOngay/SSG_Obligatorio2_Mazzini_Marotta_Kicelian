"""
Configuración del sistema RAG
"""
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
STATIC_FOLDER = "static"

# Configuración del modelo LLM
LLM_MODEL = "deepseek-v3.1:671b-cloud"
LLM_BASE_URL = "http://127.0.0.1:11434"

# Configuración de embeddings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Archivos de datos
CLIENTES_FILE = "Clientes-Tabla 1.csv"
PRODUCTOS_FILE = "Productos-Tabla 1.csv"
VENTAS_FILE = "Ventas-Tabla 1.csv"

# Separador CSV
CSV_SEPARATOR = ';'
CSV_SKIPROWS = 1

# Configuración RAG
RAG_TOP_K = 5  # Número de documentos a recuperar
RAG_TEMPERATURE = 0.1  # Temperatura para el LLM
