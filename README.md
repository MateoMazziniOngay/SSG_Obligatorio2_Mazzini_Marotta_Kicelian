# Sistema RAG - Análisis de Ventas

Sistema de análisis de datos de ventas utilizando RAG (Retrieval-Augmented Generation) con LLM local.

## 📁 Estructura del Proyecto

```
src/tarea_rag/
├── config.py              # Configuración del sistema (rutas, modelo LLM, constantes)
├── data_loader.py         # Carga y procesamiento de datos CSV
├── prompts.py             # Plantillas de prompts para el LLM
├── query_processor.py     # Lógica de procesamiento de consultas
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

## 🏗️ Arquitectura

### Módulos Principales

#### 1. **config.py**
- Centraliza toda la configuración del sistema
- Define rutas de archivos y carpetas
- Configuración del modelo LLM
- Constantes globales

#### 2. **data_loader.py**
- Clase `DataLoader`: Maneja la carga de datos CSV
- Métodos:
  - `load_data()`: Carga todos los CSVs y crea DataFrames
  - `get_schema_info()`: Genera descripción del esquema para el LLM
  - `get_stats()`: Calcula estadísticas generales

#### 3. **prompts.py**
- Clase `PromptTemplates`: Contiene todas las plantillas de prompts
- Prompts para:
  - Análisis de datos (generación de código pandas)
  - Clasificación de preguntas
  - Respuestas conversacionales
  - Manejo de preguntas fuera del dominio
  - Generación de respuestas en lenguaje natural

#### 4. **query_processor.py**
- Clase `QueryProcessor`: Procesa consultas del usuario
- Métodos principales:
  - `classify_question()`: Clasifica el tipo de pregunta
  - `handle_out_of_domain()`: Maneja preguntas no relacionadas
  - `handle_conversation()`: Maneja interacciones conversacionales
  - `handle_data_query()`: Procesa consultas de datos
  - `process_query()`: Método principal que orquesta el flujo

#### 5. **rag_api.py**
- API Flask con endpoints REST
- Endpoints:
  - `GET /`: Página principal
  - `GET /<filename>`: Archivos estáticos
  - `GET /api/health`: Estado del servicio
  - `GET /api/stats`: Estadísticas generales
  - `POST /api/query`: Procesar consultas

#### 6. **Frontend (static/)**
- **index.html**: Estructura de la interfaz
- **styles.css**: Estilos y diseño visual
- **script.js**: Lógica del cliente (fetch, DOM, eventos)

## 🚀 Ejecución

```bash
cd tarea_rag/tarea_rag
poetry run python src/tarea_rag/rag_api.py
```

El servidor estará disponible en: `http://localhost:5001`

## 🎯 Flujo de Trabajo

1. **Usuario hace una pregunta** → Frontend envía POST a `/api/query`
2. **QueryProcessor clasifica** → ¿Datos, conversación o fuera del dominio?
3. **Procesamiento según tipo**:
   - **Datos**: Genera código pandas → Ejecuta → Formatea resultado → Genera respuesta natural
   - **Conversación**: Responde directamente
   - **Fuera dominio**: Explica limitaciones
4. **Respuesta al usuario** → Frontend muestra la respuesta

## 📊 Datos

El sistema analiza tres tipos de datos:
- **Clientes**: Información de clientes
- **Productos**: Catálogo de productos con categorías y precios
- **Ventas**: Transacciones con fechas, cantidades y totales

## 🛠️ Tecnologías

- **Backend**: Flask, Pandas, LangChain, Ollama
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **LLM**: DeepSeek v3.1 (local via Ollama)

## 📝 Buenas Prácticas Implementadas

✅ **Separación de responsabilidades**: Cada módulo tiene una única responsabilidad
✅ **Configuración centralizada**: Fácil modificación de parámetros
✅ **Reutilización de código**: Clases y métodos bien definidos
✅ **Frontend modular**: HTML, CSS y JS separados
✅ **Manejo de errores**: Try-catch y validaciones
✅ **Documentación**: Docstrings y comentarios claros
