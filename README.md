# Sistema RAG - Análisis de Ventas

Sistema de análisis de datos de ventas utilizando **RAG (Retrieval-Augmented Generation)** con embeddings, vectorstore FAISS y LLM local (Ollama).

## 📁 Estructura del Proyecto

```
src/tarea_rag/
├── config.py              # Configuración del sistema (rutas, modelos, constantes)
├── data_loader.py         # Carga de datos CSV y creación de documentos
├── vectorstore.py         # Gestión del vectorstore FAISS con embeddings
├── query_processor.py     # Procesamiento de consultas usando RAG
├── rag_api.py             # API Flask (endpoints y rutas)
├── data/                  # Archivos CSV de datos
│   ├── Clientes-Tabla 1.csv
│   ├── Productos-Tabla 1.csv
│   └── Ventas-Tabla 1.csv
└── static/                # Frontend (HTML, CSS, JS)
    ├── index.html
    ├── styles.css
    └── script.js
```

## 🏗️ Arquitectura RAG

### Flujo del Sistema

1. **Carga de Datos** (`data_loader.py`)
   - Lee archivos CSV de clientes, productos y ventas
   - Enriquece datos con JOINs
   - Crea documentos estructurados con metadata

2. **Vectorización** (`vectorstore.py`)
   - Genera embeddings usando HuggingFace Transformers
   - Almacena en vectorstore FAISS
   - Indexa para búsqueda por similaridad

3. **Consulta RAG** (`query_processor.py`)
   - Búsqueda semántica de documentos relevantes
   - Construcción de contexto
   - Generación de respuesta con LLM

4. **API REST** (`rag_api.py`)
   - Endpoints Flask
   - Integración con frontend

### Módulos Principales

#### 1. **config.py**
- Configuración centralizada
- Parámetros del modelo LLM (Ollama)
- Configuración de embeddings (HuggingFace)
- Constantes RAG (top_k, temperature)

#### 2. **data_loader.py**
- Clase `DataLoader`: Carga y procesa datos
- Método `create_documents()`: Convierte datos en documentos LangChain
- Crea documentos para:
  - Ventas individuales con detalles completos
  - Resúmenes por cliente
  - Resúmenes por producto
  - Resúmenes por categoría

#### 3. **vectorstore.py**
- Clase `VectorStoreManager`: Gestiona FAISS
- Inicializa modelo de embeddings multilingüe
- Métodos:
  - `create_vectorstore()`: Crea índice vectorial
  - `similarity_search()`: Búsqueda semántica
  - `similarity_search_with_score()`: Con scores de similaridad

#### 4. **query_processor.py**
- Clase `QueryProcessor`: Procesa consultas con RAG
- Usa `RetrievalQA` de LangChain
- Pipeline:
  1. Usuario hace pregunta
  2. Búsqueda vectorial de documentos relevantes (top_k)
  3. Construcción de contexto
  4. Generación de respuesta con LLM
  5. Respuesta en lenguaje natural

#### 5. **rag_api.py**
- API Flask con endpoints REST
- Endpoints:
  - `GET /`: Página principal
  - `GET /<filename>`: Archivos estáticos
  - `GET /api/health`: Estado del servicio
  - `GET /api/stats`: Estadísticas generales
  - `POST /api/query`: Procesar consultas RAG

#### 6. **Frontend (static/)**
- **index.html**: Estructura de la interfaz
- **styles.css**: Estilos y diseño visual
- **script.js**: Lógica del cliente

## 🚀 Ejecución

```bash
cd tarea_rag/tarea_rag
poetry run python src/tarea_rag/rag_api.py
```

El servidor estará disponible en: `http://localhost:5001`

## 🎯 Flujo de Trabajo RAG

1. **Usuario hace pregunta** → Frontend envía POST a `/api/query`
2. **Búsqueda vectorial** → Sistema busca los 5 documentos más relevantes por similaridad semántica
3. **Construcción de contexto** → Documentos recuperados se usan como contexto
4. **Generación LLM** → El LLM genera respuesta basándose en el contexto
5. **Respuesta al usuario** → Frontend muestra respuesta + fuentes utilizadas

## 📊 Datos

El sistema convierte datos CSV en documentos vectorizados:
- **Clientes**: Información y resúmenes de compras
- **Productos**: Catálogo con estadísticas de ventas
- **Ventas**: Transacciones individuales con detalles completos
- **Agregaciones**: Resúmenes por cliente, producto y categoría

## 🛠️ Tecnologías

### Backend
- **Flask**: API REST
- **Pandas**: Procesamiento de datos
- **LangChain**: Framework RAG
- **FAISS**: Vectorstore para búsqueda por similaridad
- **HuggingFace Transformers**: Generación de embeddings
- **Ollama**: LLM local (DeepSeek v3.1)

### Frontend
- **HTML5, CSS3, JavaScript**: Interfaz de usuario

### Modelos
- **LLM**: DeepSeek v3.1 (vía Ollama)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2

## 📝 Ventajas del Sistema RAG

✅ **No genera código**: Respuestas directas basadas en información recuperada
✅ **Búsqueda semántica**: Encuentra información relevante aunque use palabras diferentes
✅ **Contexto preciso**: Solo usa información relevante de la base de datos
✅ **Escalable**: Fácil agregar más datos sin cambiar código
✅ **Transparente**: Muestra las fuentes usadas para cada respuesta
✅ **Multilingüe**: Embeddings optimizados para español
