"""
Módulo para procesamiento de consultas y ejecución de código pandas
"""
import pandas as pd
import re
from typing import Any, Dict
from langchain_ollama import OllamaLLM
from tarea_rag.prompts import PromptTemplates
from tarea_rag.config import LLM_MODEL, LLM_BASE_URL


class QueryProcessor:
    """Clase para procesar consultas del usuario usando LLM y pandas"""
    
    def __init__(self, df_clientes: pd.DataFrame, df_productos: pd.DataFrame, 
                 df_ventas: pd.DataFrame, df_ventas_full: pd.DataFrame, 
                 esquema_datos: str):
        self.df_clientes = df_clientes
        self.df_productos = df_productos
        self.df_ventas = df_ventas
        self.df_ventas_full = df_ventas_full
        self.esquema_datos = esquema_datos
        
        # Inicializar LLM
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL
        )
        
        # Cargar prompts
        self.prompts = PromptTemplates()
    
    def classify_question(self, pregunta: str) -> str:
        """Clasifica si la pregunta es de datos, conversacional o fuera del dominio"""
        print(f"🔍 Clasificando pregunta...")
        prompt = self.prompts.get_classification_prompt()
        clasificacion = self.llm.invoke(prompt.format(pregunta=pregunta)).strip().lower()
        print(f"📋 Clasificación: {clasificacion}")
        return clasificacion
    
    def handle_out_of_domain(self, pregunta: str) -> Dict[str, Any]:
        """Maneja preguntas fuera del dominio"""
        print(f"🚫 Pregunta fuera del dominio...")
        prompt = self.prompts.get_out_of_domain_prompt()
        respuesta = self.llm.invoke(prompt.format(pregunta=pregunta))
        
        return {
            "pregunta": pregunta,
            "respuesta": respuesta.strip(),
            "tipo": "fuera_dominio"
        }
    
    def handle_conversation(self, pregunta: str) -> Dict[str, Any]:
        """Maneja preguntas conversacionales"""
        print(f"💬 Respondiendo de forma conversacional...")
        prompt = self.prompts.get_conversation_prompt()
        respuesta = self.llm.invoke(prompt.format(pregunta=pregunta))
        
        return {
            "pregunta": pregunta,
            "respuesta": respuesta.strip(),
            "tipo": "conversacion"
        }
    
    def handle_data_query(self, pregunta: str) -> Dict[str, Any]:
        """Procesa consultas que requieren análisis de datos"""
        print(f"🤖 Generando código para consulta...")
        
        # Generar código con LLM
        prompt = self.prompts.get_analysis_prompt()
        respuesta_llm = self.llm.invoke(prompt.format(
            esquema=self.esquema_datos, 
            input=pregunta
        ))
        
        print(f"🤖 LLM generó: {respuesta_llm[:200]}...")
        
        # Limpiar código
        codigo = self._clean_code(respuesta_llm)
        
        if not codigo:
            return {"error": "El LLM no generó código válido"}
        
        print(f"💻 Ejecutando: {codigo}")
        
        # Ejecutar código
        resultado = self._execute_pandas_code(codigo)
        
        # Formatear resultado
        resultado_formateado = self._format_result(resultado)
        
        print(f"✅ Resultado: {resultado_formateado}")
        
        # Generar respuesta en lenguaje natural
        print(f"🤖 Generando respuesta natural...")
        respuesta_natural = self._generate_natural_response(pregunta, resultado_formateado)
        
        print(f"💬 Respuesta: {respuesta_natural[:100]}...")
        
        return {
            "pregunta": pregunta,
            "respuesta": respuesta_natural.strip(),
            "resultado_raw": resultado_formateado,
            "tipo": type(resultado).__name__
        }
    
    def _clean_code(self, codigo_llm: str) -> str:
        """Limpia el código generado por el LLM"""
        codigo = codigo_llm.strip()
        # Remover markdown si existe
        codigo = re.sub(r'```(?:python)?\s*', '', codigo)
        codigo = re.sub(r'```\s*', '', codigo)
        # Tomar solo la primera línea si hay múltiples
        lineas = [l.strip() for l in codigo.split('\n') 
                  if l.strip() and not l.strip().startswith('#')]
        return lineas[0] if lineas else ""
    
    def _execute_pandas_code(self, codigo_pandas: str) -> Any:
        """Ejecuta código pandas de forma segura"""
        try:
            # Verificar que los paréntesis estén balanceados
            if codigo_pandas.count('(') != codigo_pandas.count(')'):
                return "Error: código generado tiene paréntesis desbalanceados"
            
            namespace = {
                'df_ventas': self.df_ventas,
                'df_productos': self.df_productos,
                'df_clientes': self.df_clientes,
                'df_ventas_full': self.df_ventas_full,
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
    
    def _format_result(self, resultado: Any) -> Any:
        """Formatea el resultado para enviarlo como JSON"""
        if isinstance(resultado, pd.DataFrame):
            return resultado.to_dict('records')
        elif isinstance(resultado, pd.Series):
            return resultado.to_dict()
        elif isinstance(resultado, (int, float, str)):
            return resultado
        else:
            return str(resultado)
    
    def _generate_natural_response(self, pregunta: str, resultado: Any) -> str:
        """Genera respuesta en lenguaje natural"""
        prompt = self.prompts.get_natural_response_prompt()
        respuesta = self.llm.invoke(prompt.format(
            pregunta=pregunta,
            resultado=resultado
        ))
        return respuesta
    
    def process_query(self, pregunta: str) -> Dict[str, Any]:
        """Procesa una consulta completa del usuario"""
        print(f"\n📊 Pregunta: {pregunta}")
        
        if not pregunta:
            return {"error": "Pregunta vacía"}
        
        # Clasificar pregunta
        clasificacion = self.classify_question(pregunta)
        
        # Manejar según clasificación
        if "fuera_dominio" in clasificacion or "fuera" in clasificacion:
            return self.handle_out_of_domain(pregunta)
        elif "conversacion" in clasificacion or "conversación" in clasificacion:
            return self.handle_conversation(pregunta)
        else:
            return self.handle_data_query(pregunta)
