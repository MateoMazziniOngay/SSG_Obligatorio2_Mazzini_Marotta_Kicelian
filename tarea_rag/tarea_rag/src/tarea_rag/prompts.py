"""
Plantillas de prompts para el sistema RAG
"""
from langchain_core.prompts import ChatPromptTemplate


class PromptTemplates:
    """Clase que contiene todas las plantillas de prompts"""
    
    @staticmethod
    def get_analysis_prompt() -> ChatPromptTemplate:
        """Prompt para generar código pandas a partir de preguntas"""
        return ChatPromptTemplate.from_template("""Eres un asistente experto en análisis de datos con pandas.

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
    
    @staticmethod
    def get_classification_prompt() -> ChatPromptTemplate:
        """Prompt para clasificar el tipo de pregunta"""
        return ChatPromptTemplate.from_template("""Eres un asistente que clasifica si una pregunta requiere consultar datos o es solo conversacional.

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
    
    @staticmethod
    def get_out_of_domain_prompt() -> ChatPromptTemplate:
        """Prompt para responder preguntas fuera del dominio"""
        return ChatPromptTemplate.from_template("""El usuario te hizo una pregunta que no está relacionada con tu especialidad.

Pregunta: {pregunta}

Responde amablemente explicando que eres un asistente especializado en análisis de datos de ventas, productos y clientes, y que solo puedes ayudar con preguntas relacionadas a ese dominio. Sugiere que te hagan preguntas sobre ventas, productos o clientes.

Respuesta:""")
    
    @staticmethod
    def get_conversation_prompt() -> ChatPromptTemplate:
        """Prompt para respuestas conversacionales"""
        return ChatPromptTemplate.from_template("""Eres un asistente amigable de análisis de datos de ventas. El usuario te está escribiendo de forma conversacional.

Usuario: {pregunta}

Responde de forma amigable y breve. Si es un saludo, preséntate y ofrece ayuda. Si te preguntan qué puedes hacer, explica brevemente que puedes responder preguntas sobre ventas, productos y clientes.

Respuesta:""")
    
    @staticmethod
    def get_natural_response_prompt() -> ChatPromptTemplate:
        """Prompt para generar respuestas en lenguaje natural"""
        return ChatPromptTemplate.from_template("""Eres un asistente amigable de análisis de datos. Tu trabajo es convertir resultados de consultas en respuestas naturales y fáciles de entender.

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
