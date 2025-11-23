"""
Módulo para procesamiento de consultas usando RAG
"""
from typing import Any, Dict
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from tarea_rag.vectorstore import VectorStoreManager
from tarea_rag.config import LLM_MODEL, LLM_BASE_URL, RAG_TOP_K, RAG_TEMPERATURE


class QueryProcessor:
    """Clase para procesar consultas del usuario usando RAG"""
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._create_qa_chain()
    
    def _initialize_llm(self):
        """Inicializa el modelo de lenguaje"""
        print("Inicializando LLM...")
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            temperature=RAG_TEMPERATURE
        )
        print(f"✓ LLM inicializado: {LLM_MODEL}")
    
    def _create_qa_chain(self):
        """Crea la cadena de pregunta-respuesta"""
        print("Creando cadena RAG...")
        
        # Template para el prompt
        template = """Eres un asistente experto en análisis de datos de ventas. Usa la siguiente información para responder la pregunta del usuario de manera clara, precisa y amigable.

Contexto recuperado de la base de datos:
{context}

Pregunta del usuario: {question}

INSTRUCCIONES:
1. Analiza cuidadosamente el contexto proporcionado
2. Responde de forma directa y específica basándote SOLO en la información del contexto
3. Si necesitas mencionar números, cálculos o estadísticas, asegúrate de que estén en el contexto
4. Si el contexto no contiene información suficiente para responder, indícalo claramente
5. Usa emojis apropiados para hacer la respuesta más amigable (📊, 💰, 🏆, 📈, etc.)
6. Sé conciso pero informativo
7. Si hay múltiples datos relevantes, preséntalos de forma organizada

Respuesta:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Crear la cadena RetrievalQA
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore_manager.vectorstore.as_retriever(
                search_kwargs={"k": RAG_TOP_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("✓ Cadena RAG creada exitosamente")
    
    def process_query(self, pregunta: str) -> Dict[str, Any]:
        """Procesa una consulta usando RAG"""
        print(f"\n📊 Pregunta: {pregunta}")
        
        if not pregunta:
            return {"error": "Pregunta vacía"}
        
        try:
            # Ejecutar la consulta RAG
            print("🔍 Buscando documentos relevantes...")
            result = self.qa_chain.invoke({"query": pregunta})
            
            # Extraer resultado y documentos fuente
            respuesta = result['result']
            source_docs = result['source_documents']
            
            print(f"✓ Encontrados {len(source_docs)} documentos relevantes")
            print(f"💬 Respuesta generada")
            
            # Preparar metadata de los documentos recuperados
            sources = []
            for doc in source_docs:
                sources.append({
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                })
            
            return {
                "pregunta": pregunta,
                "respuesta": respuesta,
                "num_documentos_recuperados": len(source_docs),
                "sources": sources,
                "tipo": "rag"
            }
            
        except Exception as e:
            print(f"❌ Error procesando consulta: {str(e)}")
            return {
                "error": f"Error al procesar la consulta: {str(e)}",
                "pregunta": pregunta
            }
