"""
Módulo para carga y procesamiento de datos
"""
import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from tarea_rag.config import DATA_FOLDER, CLIENTES_FILE, PRODUCTOS_FILE, VENTAS_FILE, CSV_SEPARATOR, CSV_SKIPROWS


class DataLoader:
    """Clase para cargar y preparar los datos de ventas para RAG"""
    
    def __init__(self, data_folder: Path = DATA_FOLDER):
        self.data_folder = data_folder
        self.df_clientes = None
        self.df_productos = None
        self.df_ventas = None
        self.df_ventas_full = None
    
    def load_data(self) -> pd.DataFrame:
        """Carga todos los archivos CSV y crea el DataFrame enriquecido"""
        print("Cargando datos...")
        
        # Cargar CSVs
        self.df_clientes = pd.read_csv(
            self.data_folder / CLIENTES_FILE, 
            sep=CSV_SEPARATOR, 
            skiprows=CSV_SKIPROWS
        )
        self.df_productos = pd.read_csv(
            self.data_folder / PRODUCTOS_FILE, 
            sep=CSV_SEPARATOR, 
            skiprows=CSV_SKIPROWS
        )
        self.df_ventas = pd.read_csv(
            self.data_folder / VENTAS_FILE, 
            sep=CSV_SEPARATOR, 
            skiprows=CSV_SKIPROWS
        )
        
        # Procesar fechas
        self.df_ventas['FechaVenta'] = pd.to_datetime(self.df_ventas['FechaVenta'])
        self.df_ventas['Año'] = self.df_ventas['FechaVenta'].dt.year
        self.df_ventas['Mes'] = self.df_ventas['FechaVenta'].dt.month
        self.df_ventas['Mes_Nombre'] = self.df_ventas['FechaVenta'].dt.strftime('%B')
        
        # Crear datos enriquecidos con JOINs
        self.df_ventas_full = self._create_enriched_data()
        
        print(f"✓ Cargados {len(self.df_clientes)} clientes, "
              f"{len(self.df_productos)} productos, "
              f"{len(self.df_ventas)} ventas")
        
        return self.df_ventas_full
    
    def _create_enriched_data(self) -> pd.DataFrame:
        """Crea DataFrame enriquecido con información de productos y clientes"""
        df_full = self.df_ventas.merge(self.df_productos, on='IdProducto', how='left')
        df_full = df_full.merge(self.df_clientes, on='IdCliente', how='left')
        df_full['Total'] = df_full['Cantidad'] * df_full['Precio']
        return df_full
    
    def create_documents(self) -> List[Document]:
        """Convierte los datos en documentos para el vectorstore"""
        if self.df_ventas_full is None:
            raise ValueError("Datos no cargados. Ejecuta load_data() primero.")
        
        documents = []
        
        # Crear documentos para cada venta con información completa
        for _, row in self.df_ventas_full.iterrows():
            content = f"""
Venta ID: {row['IdVenta']}
Cliente: {row['NombreCliente']} (ID: {row['IdCliente']})
Producto: {row['NombreProducto']} (ID: {row['IdProducto']})
Categoría: {row['Categoria']}
Fecha: {row['FechaVenta'].strftime('%Y-%m-%d')}
Año: {row['Año']}
Mes: {row['Mes']} ({row['Mes_Nombre']})
Cantidad: {row['Cantidad']} unidades
Precio unitario: ${row['Precio']:.2f}
Total: ${row['Total']:.2f}
"""
            metadata = {
                'id_venta': int(row['IdVenta']),
                'id_cliente': int(row['IdCliente']),
                'nombre_cliente': str(row['NombreCliente']),
                'id_producto': int(row['IdProducto']),
                'nombre_producto': str(row['NombreProducto']),
                'categoria': str(row['Categoria']),
                'fecha': str(row['FechaVenta'].strftime('%Y-%m-%d')),
                'año': int(row['Año']),
                'mes': int(row['Mes']),
                'mes_nombre': str(row['Mes_Nombre']),
                'cantidad': int(row['Cantidad']),
                'precio': float(row['Precio']),
                'total': float(row['Total'])
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Agregar documentos resumen por cliente
        clientes_resumen = self.df_ventas_full.groupby('NombreCliente').agg({
            'Total': 'sum',
            'IdVenta': 'count'
        }).reset_index()
        
        for _, row in clientes_resumen.iterrows():
            content = f"""
Resumen del Cliente: {row['NombreCliente']}
Total de compras: {row['IdVenta']} transacciones
Monto total gastado: ${row['Total']:.2f}
Promedio por compra: ${row['Total']/row['IdVenta']:.2f}
"""
            metadata = {
                'tipo': 'resumen_cliente',
                'nombre_cliente': str(row['NombreCliente']),
                'total_compras': int(row['IdVenta']),
                'monto_total': float(row['Total'])
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Agregar documentos resumen por producto
        productos_resumen = self.df_ventas_full.groupby(['NombreProducto', 'Categoria']).agg({
            'Cantidad': 'sum',
            'Total': 'sum',
            'IdVenta': 'count'
        }).reset_index()
        
        for _, row in productos_resumen.iterrows():
            content = f"""
Resumen del Producto: {row['NombreProducto']}
Categoría: {row['Categoria']}
Unidades vendidas: {row['Cantidad']}
Número de ventas: {row['IdVenta']}
Ingresos totales: ${row['Total']:.2f}
"""
            metadata = {
                'tipo': 'resumen_producto',
                'nombre_producto': str(row['NombreProducto']),
                'categoria': str(row['Categoria']),
                'unidades_vendidas': int(row['Cantidad']),
                'ingresos_totales': float(row['Total'])
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Agregar documentos resumen por categoría
        categorias_resumen = self.df_ventas_full.groupby('Categoria').agg({
            'Total': 'sum',
            'Cantidad': 'sum',
            'IdVenta': 'count'
        }).reset_index()
        
        for _, row in categorias_resumen.iterrows():
            content = f"""
Resumen de Categoría: {row['Categoria']}
Ventas totales: {row['IdVenta']} transacciones
Unidades vendidas: {row['Cantidad']}
Ingresos totales: ${row['Total']:.2f}
"""
            metadata = {
                'tipo': 'resumen_categoria',
                'categoria': str(row['Categoria']),
                'ventas_totales': int(row['IdVenta']),
                'ingresos_totales': float(row['Total'])
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        print(f"✓ Creados {len(documents)} documentos para el vectorstore")
        return documents
        
        print(f"✓ Creados {len(documents)} documentos para el vectorstore")
        return documents
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas generales de los datos"""
        if self.df_ventas_full is None:
            raise ValueError("Datos no cargados. Ejecuta load_data() primero.")
        
        return {
            "total_clientes": int(len(self.df_clientes)),
            "total_productos": int(len(self.df_productos)),
            "total_ventas": int(len(self.df_ventas)),
            "ingresos_totales": float(self.df_ventas_full['Total'].sum()),
            "fecha_min": str(self.df_ventas_full['FechaVenta'].min()),
            "fecha_max": str(self.df_ventas_full['FechaVenta'].max()),
            "categorias": self.df_ventas_full['Categoria'].unique().tolist()
        }
