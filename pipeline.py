# pipeline.py
from src.scraper_facebook import obtener_datos_facebook
from src.preprocessing import preprocess_dataframe
from src.embedding import generar_embeddings
from src.classification import predecir_clases
from src.clustering import agrupar_post
from src import analysis
from dotenv import load_dotenv
import pandas as pd
import os


# Cargar variables de entorno
load_dotenv("/dbfs/FileStore/dreammaker/.env")
FACEBOOK_TOKEN = os.getenv("FACEBOOK_TOKEN")  

# Rutas locales dentro del proyecto
DATALAKE_ROOT = "/dbfs/FileStore/dreammaker/output"
RAW_PATH = os.path.join(DATALAKE_ROOT, "raw")
PROCESSED_PATH = os.path.join(DATALAKE_ROOT, "processed")
REPORTS_PATH = os.path.join(DATALAKE_ROOT, "reports")
IMAGES_PATH = os.path.join(DATALAKE_ROOT, "images")

# Crear carpetas si no existen
for path in [RAW_PATH, PROCESSED_PATH, REPORTS_PATH, IMAGES_PATH]:
    os.makedirs(path, exist_ok=True)

def main():
    # 1. Scraping
    print("Obteniendo datos desde Facebook...")
    df = obtener_datos_facebook(FACEBOOK_TOKEN, datalake_path=RAW_PATH)  # guarda en /dbfs/FileStore/dreammaker/output/raw

    # 2. Preprocesamiento
    print("Limpiando texto...")
    df = preprocess_dataframe(df, output_path=PROCESSED_PATH)  # guarda en /dbfs/FileStore/dreammaker/output/processed

    # 3. Generar embeddings
    print("Generando embeddings...")
    df = generar_embeddings(df, output_path=PROCESSED_PATH)  # guarda en /dbfs/FileStore/dreammaker/output/processed

    # 4. Clasificación
    print("Clasificando comentarios...")
    df = predecir_clases(df, output_path=PROCESSED_PATH)  # guarda en /dbfs/FileStore/dreammaker/output/processed

    # 5. Clustering
    print("Agrupando posts...")
    df_agrupados = agrupar_post(df=df, n_clusters=11, plot=False, processed_path=PROCESSED_PATH, images_path=IMAGES_PATH) # guarda en /dbfs/FileStore/dreammaker/output/processed e imágenes en output/images

    # 6. Columnas de clasificación y colores
    df_agrupados = analysis.crear_columnas_clasificacion(df_agrupados)
    df_agrupados = analysis.asignar_colores_y_bordes(df_agrupados)
    
    # ** Funciones posteriores (Generar gráficos y Reporte estadístico) no son necesarias. **
    #    Es utilizados únicamente como información preliminar rápida antes de reportes en PowerBI.

    # 7. Generar gráficos
    analysis.plot_proporcion(df_agrupados, folder=IMAGES_PATH)
    analysis.plot_volumen(df_agrupados,
                          x_col='Comentario político emocional',
                          y_col='Pregunta sobre el producto',
                          log_scale=True,
                          folder=IMAGES_PATH)

    # 8. Reporte estadístico
    analysis.generar_reporte_estadistico(df_agrupados, folder=REPORTS_PATH)

    # 9. Guardar resultados finales
    df.to_parquet(os.path.join(PROCESSED_PATH, "pipeline_resultados.parquet"), index=False)
    df_agrupados.to_parquet(os.path.join(PROCESSED_PATH, "pipeline_agrupados.parquet"), index=False)

if __name__ == "__main__":
    main()

