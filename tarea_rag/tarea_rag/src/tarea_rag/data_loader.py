"""
Módulo para carga y procesamiento de datos
"""
import pandas as pd
from pathlib import Path
from typing import Tuple
from tarea_rag.config import DATA_FOLDER, CLIENTES_FILE, PRODUCTOS_FILE, VENTAS_FILE, CSV_SEPARATOR, CSV_SKIPROWS


class DataLoader:
    """Clase para cargar y preparar los datos de ventas"""
    
    def __init__(self, data_folder: Path = DATA_FOLDER):
        self.data_folder = data_folder
        self.df_clientes = None
        self.df_productos = None
        self.df_ventas = None
        self.df_ventas_full = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        
        # Crear datos enriquecidos con JOINs
        self.df_ventas_full = self._create_enriched_data()
        
        print(f"✓ Cargados {len(self.df_clientes)} clientes, "
              f"{len(self.df_productos)} productos, "
              f"{len(self.df_ventas)} ventas")
        
        return self.df_clientes, self.df_productos, self.df_ventas, self.df_ventas_full
    
    def _create_enriched_data(self) -> pd.DataFrame:
        """Crea DataFrame enriquecido con información de productos y clientes"""
        df_full = self.df_ventas.merge(self.df_productos, on='IdProducto', how='left')
        df_full = df_full.merge(self.df_clientes, on='IdCliente', how='left')
        df_full['Total'] = df_full['Cantidad'] * df_full['Precio']
        return df_full
    
    def get_schema_info(self) -> str:
        """Genera descripción del esquema de datos para el LLM"""
        if self.df_ventas_full is None:
            raise ValueError("Datos no cargados. Ejecuta load_data() primero.")
        
        return f"""
ESTRUCTURA DE DATOS DISPONIBLE:

1. df_clientes: {len(self.df_clientes)} registros
   Columnas: {list(self.df_clientes.columns)}

2. df_productos: {len(self.df_productos)} registros
   Columnas: {list(self.df_productos.columns)}

3. df_ventas: {len(self.df_ventas)} registros
   Columnas: {list(self.df_ventas.columns)}

4. df_ventas_full: {len(self.df_ventas_full)} registros (ventas con JOIN)
   Columnas: {list(self.df_ventas_full.columns)}
   Incluye: Año, Mes, Total (Cantidad * Precio)

ESTADÍSTICAS:
- Total ventas: {len(self.df_ventas_full)}
- Rango de fechas: {self.df_ventas_full['FechaVenta'].min()} a {self.df_ventas_full['FechaVenta'].max()}
- Total ingresos: ${self.df_ventas_full['Total'].sum():.2f}
- Categorías: {self.df_ventas_full['Categoria'].unique().tolist()}
"""
    
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
