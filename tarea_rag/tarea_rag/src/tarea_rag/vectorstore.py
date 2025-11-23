"""
Módulo para gestión del vectorstore FAISS
"""
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tarea_rag.config import EMBEDDING_MODEL


class VectorStoreManager:
    """Clase para gestionar el vectorstore FAISS"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Inicializa el modelo de embeddings"""
        print("Inicializando modelo de embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✓ Modelo de embeddings cargado: {EMBEDDING_MODEL}")
    
    def create_vectorstore(self, documents: List[Document]):
        """Crea el vectorstore a partir de documentos"""
        print(f"Creando vectorstore con {len(documents)} documentos...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("✓ Vectorstore creado exitosamente")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Busca documentos similares a la consulta"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore no inicializado. Ejecuta create_vectorstore() primero.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Busca documentos similares con score de similaridad"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore no inicializado. Ejecuta create_vectorstore() primero.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
