"""
API Flask para el sistema RAG de análisis de ventas
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pathlib import Path
import pandas as pd
import re
import traceback
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Permitir CORS para el frontend

# Cargar datos al iniciar
print("Cargando datos...")
data_folder = Path(__file__).resolve().parent / "data"

# Cargar CSVs
df_clientes = pd.read_csv(data_folder / "Clientes-Tabla 1.csv", sep=';', skiprows=1)
df_productos = pd.read_csv(data_folder / "Productos-Tabla 1.csv", sep=';', skiprows=1)
df_ventas = pd.read_csv(data_folder / "Ventas-Tabla 1.csv", sep=';', skiprows=1)
df_ventas['FechaVenta'] = pd.to_datetime(df_ventas['FechaVenta'])
df_ventas['Año'] = df_ventas['FechaVenta'].dt.year
df_ventas['Mes'] = df_ventas['FechaVenta'].dt.month

# Crear datos enriquecidos
df_ventas_full = df_ventas.merge(df_productos, on='IdProducto', how='left')
df_ventas_full = df_ventas_full.merge(df_clientes, on='IdCliente', how='left')
df_ventas_full['Total'] = df_ventas_full['Cantidad'] * df_ventas_full['Precio']

print(f"✓ Cargados {len(df_clientes)} clientes, {len(df_productos)} productos, {len(df_ventas)} ventas")

# Configurar LLM
llm = OllamaLLM(
    model="deepseek-v3.1:671b-cloud",
    base_url="http://127.0.0.1:11434"
)

# Crear esquema de datos
esquema_datos = f"""
ESTRUCTURA DE DATOS DISPONIBLE:

1. df_clientes: {len(df_clientes)} registros
   Columnas: {list(df_clientes.columns)}

2. df_productos: {len(df_productos)} registros
   Columnas: {list(df_productos.columns)}

3. df_ventas: {len(df_ventas)} registros
   Columnas: {list(df_ventas.columns)}

4. df_ventas_full: {len(df_ventas_full)} registros (ventas con JOIN)
   Columnas: {list(df_ventas_full.columns)}
   Incluye: Año, Mes, Total (Cantidad * Precio)

ESTADÍSTICAS:
- Total ventas: {len(df_ventas_full)}
- Rango de fechas: {df_ventas_full['FechaVenta'].min()} a {df_ventas_full['FechaVenta'].max()}
- Total ingresos: ${df_ventas_full['Total'].sum():.2f}
- Categorías: {df_ventas_full['Categoria'].unique().tolist()}
"""

prompt_analisis = ChatPromptTemplate.from_template("""Eres un asistente experto en análisis de datos con pandas.

{esquema}

Pregunta: {input}

INSTRUCCIONES CRÍTICAS:
1. Genera código pandas VÁLIDO en UNA SOLA LÍNEA
2. Usa: df_ventas_full, df_ventas, df_productos, df_clientes
3. NO uses print, return, ni múltiples líneas
4. Para encontrar el TOP 1, usa: .groupby().agg().sort_values(ascending=False).head(1)
5. Para convertir a dict simple usa: .to_dict('records')[0] si es DataFrame con 1 fila
6. NUNCA uses .iloc[0] sin verificar que hay datos primero
7. Asegúrate de que todos los paréntesis estén balanceados
8. Para suma de totales usa: .sum() no .agg()

EJEMPLOS DE CÓDIGO CORRECTO:
- Cliente que más compró: df_ventas_full.groupby('NombreCliente')['Total'].sum().sort_values(ascending=False).head(1).to_dict()
- Categoría con más ingresos: df_ventas_full.groupby('Categoria')['Total'].sum().sort_values(ascending=False).head(1).to_dict()
- Producto más vendido: df_ventas_full.groupby('NombreProducto')['Cantidad'].sum().sort_values(ascending=False).head(1).to_dict()

RESPONDE SOLO CON EL CÓDIGO, SIN TEXTO ADICIONAL:""")

prompt_clasificacion = ChatPromptTemplate.from_template("""Eres un asistente que clasifica si una pregunta requiere consultar datos o es solo conversacional.

Pregunta del usuario: {pregunta}

Clasifica la pregunta en una de estas categorías:
- "datos": Si requiere consultar información de ventas, productos, clientes, estadísticas, números, etc.
- "conversacion": Si es un saludo, agradecimiento, despedida, pregunta sobre qué puedes hacer, etc.
- "fuera_dominio": Si la pregunta es sobre temas que NO están relacionados con ventas, productos, clientes o análisis de negocio (ej: deportes, recetas, historia, geografía, etc.)

Ejemplos:
- "hola" -> conversacion
- "¿cómo estás?" -> conversacion
- "gracias" -> conversacion
- "¿qué puedes hacer?" -> conversacion
- "¿Cuántas ventas hubo en marzo?" -> datos
- "¿Quién es el mejor cliente?" -> datos
- "muéstrame los productos" -> datos
- "¿quién es Messi?" -> fuera_dominio
- "¿cómo se hace una pizza?" -> fuera_dominio
- "¿cuál es la capital de Francia?" -> fuera_dominio
- "¿qué tiempo hace hoy?" -> fuera_dominio

RESPONDE SOLO CON UNA PALABRA: "datos", "conversacion" o "fuera_dominio":""")

prompt_fuera_dominio = ChatPromptTemplate.from_template("""El usuario te hizo una pregunta que no está relacionada con tu especialidad.

Pregunta: {pregunta}

Responde amablemente explicando que eres un asistente especializado en análisis de datos de ventas, productos y clientes, y que solo puedes ayudar con preguntas relacionadas a ese dominio. Sugiere que te hagan preguntas sobre ventas, productos o clientes.

Respuesta:""")

prompt_conversacion = ChatPromptTemplate.from_template("""Eres un asistente amigable de análisis de datos de ventas. El usuario te está escribiendo de forma conversacional.

Usuario: {pregunta}

Responde de forma amigable y breve. Si es un saludo, preséntate y ofrece ayuda. Si te preguntan qué puedes hacer, explica brevemente que puedes responder preguntas sobre ventas, productos y clientes.

Respuesta:""")

prompt_respuesta_natural = ChatPromptTemplate.from_template("""Eres un asistente amigable de análisis de datos. Tu trabajo es convertir resultados de consultas en respuestas naturales y fáciles de entender.

Pregunta del usuario: {pregunta}

Resultado obtenido: {resultado}

INSTRUCCIONES:
1. Genera una respuesta en lenguaje natural, clara y directa
2. Si el resultado es un número, incluye el número en la respuesta
3. Si el resultado es un diccionario con un solo valor, extrae y presenta ese valor de forma clara
4. Si el resultado es una tabla/lista, presenta un resumen de los datos más importantes
5. Usa emojis apropiados para hacer la respuesta más amigable (📊, 💰, 🏆, 📈, etc.)
6. Sé conciso pero informativo

Respuesta:""")

def ejecutar_consulta_pandas(codigo_pandas):
    """Ejecuta código pandas de forma segura"""
    try:
        # Limpiar y validar el código
        codigo_pandas = codigo_pandas.strip()
        
        # Verificar que los paréntesis estén balanceados
        if codigo_pandas.count('(') != codigo_pandas.count(')'):
            return "Error: código generado tiene paréntesis desbalanceados"
        
        namespace = {
            'df_ventas': df_ventas,
            'df_productos': df_productos,
            'df_clientes': df_clientes,
            'df_ventas_full': df_ventas_full,
            'pd': pd,
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'int': int,
            'float': float,
            'str': str
        }
        resultado = eval(codigo_pandas, {"__builtins__": {}}, namespace)
        
        # Manejar resultados vacíos
        if isinstance(resultado, pd.DataFrame) and len(resultado) == 0:
            return "No se encontraron resultados para esta consulta"
        if isinstance(resultado, pd.Series) and len(resultado) == 0:
            return "No se encontraron resultados para esta consulta"
            
        return resultado
    except IndexError:
        return "No se encontraron resultados que coincidan con los criterios de búsqueda"
    except SyntaxError as e:
        return f"Error de sintaxis en el código generado: {str(e)}"
    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}"

@app.route('/')
def index():
    """Servir página principal"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Verificar que el servicio está funcionando"""
    return jsonify({"status": "ok", "message": "API funcionando correctamente"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas generales"""
    return jsonify({
        "total_clientes": int(len(df_clientes)),
        "total_productos": int(len(df_productos)),
        "total_ventas": int(len(df_ventas)),
        "ingresos_totales": float(df_ventas_full['Total'].sum()),
        "fecha_min": str(df_ventas_full['FechaVenta'].min()),
        "fecha_max": str(df_ventas_full['FechaVenta'].max()),
        "categorias": df_ventas_full['Categoria'].unique().tolist()
    })

@app.route('/api/query', methods=['POST'])
def query():
    """Procesar pregunta del usuario"""
    try:
        data = request.get_json()
        pregunta = data.get('pregunta', '')
        
        if not pregunta:
            return jsonify({"error": "Pregunta vacía"}), 400
        
        print(f"\n📊 Pregunta: {pregunta}")
        
        # Paso 1: Clasificar si requiere datos o es conversacional
        print(f"🔍 Clasificando pregunta...")
        clasificacion = llm.invoke(prompt_clasificacion.format(pregunta=pregunta)).strip().lower()
        
        print(f"📋 Clasificación: {clasificacion}")
        
        # Si está fuera del dominio, responder con mensaje de limitación
        if "fuera_dominio" in clasificacion or "fuera" in clasificacion:
            print(f"🚫 Pregunta fuera del dominio...")
            respuesta = llm.invoke(prompt_fuera_dominio.format(pregunta=pregunta))
            
            return jsonify({
                "pregunta": pregunta,
                "respuesta": respuesta.strip(),
                "tipo": "fuera_dominio"
            })
        
        # Si es conversacional, responder directamente sin consultar datos
        if "conversacion" in clasificacion or "conversación" in clasificacion:
            print(f"💬 Respondiendo de forma conversacional...")
            respuesta = llm.invoke(prompt_conversacion.format(pregunta=pregunta))
            
            return jsonify({
                "pregunta": pregunta,
                "respuesta": respuesta.strip(),
                "tipo": "conversacion"
            })
        
        # Si requiere datos, continuar con el flujo normal
        print(f"🤖 Generando código para consulta...")
        
        # Generar código con LLM
        respuesta_llm = llm.invoke(prompt_analisis.format(esquema=esquema_datos, input=pregunta))
        
        print(f"🤖 LLM generó: {respuesta_llm[:200]}...")
        
        # Limpiar código
        codigo = respuesta_llm.strip()
        # Remover markdown si existe
        codigo = re.sub(r'```(?:python)?\s*', '', codigo)
        codigo = re.sub(r'```\s*', '', codigo)
        # Tomar solo la primera línea si hay múltiples
        lineas = [l.strip() for l in codigo.split('\n') if l.strip() and not l.strip().startswith('#')]
        if lineas:
            codigo = lineas[0]
        else:
            return jsonify({"error": "El LLM no generó código válido"}), 500
        
        print(f"💻 Ejecutando: {codigo}")
        
        # Ejecutar código
        resultado = ejecutar_consulta_pandas(codigo)
        
        # Formatear resultado
        if isinstance(resultado, pd.DataFrame):
            resultado_formateado = resultado.to_dict('records')
        elif isinstance(resultado, pd.Series):
            resultado_formateado = resultado.to_dict()
        elif isinstance(resultado, (int, float, str)):
            resultado_formateado = resultado
        else:
            resultado_formateado = str(resultado)
        
        print(f"✅ Resultado: {resultado_formateado}")
        
        # Generar respuesta en lenguaje natural
        print(f"🤖 Generando respuesta natural...")
        respuesta_natural = llm.invoke(prompt_respuesta_natural.format(
            pregunta=pregunta,
            resultado=resultado_formateado
        ))
        
        print(f"💬 Respuesta: {respuesta_natural[:100]}...")
        
        return jsonify({
            "pregunta": pregunta,
            "respuesta": respuesta_natural.strip(),
            "resultado_raw": resultado_formateado,
            "tipo": type(resultado).__name__
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Servidor RAG iniciado en http://localhost:5001")
    print("="*60 + "\n")
    app.run(debug=True, port=5001, host='0.0.0.0')
