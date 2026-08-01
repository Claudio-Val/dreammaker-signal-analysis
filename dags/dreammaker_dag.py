"""
dags/dreammaker_dag.py
======================
DAG de Airflow para el pipeline incremental de análisis de comentarios
de Facebook de DreamMaker.

Arquitectura Medallion:
    Bronze  → Cloud Storage  (gs://<BUCKET_NAME>/bronze/, CSV crudo de la API)
    Silver  → BigQuery       (dataset dreammaker_silver: limpio + embeddings BETO)
    Gold    → BigQuery       (dataset dreammaker_gold: predicciones + métricas por post)

Ver src/config.py para los identificadores de proyecto/bucket/datasets.

Cada task es atómica e idempotente:
    - Lee su input desde la capa anterior (GCS o BigQuery)
    - Detecta el delta respecto a lo ya procesado
    - Escribe su output en la capa actual (GCS o BigQuery)
    - Retorna el path/table_id del output vía XCom

Credenciales requeridas
------------------------
Airflow Variables:
    FACEBOOK_TOKEN   token de acceso Facebook Graph API
    PAGE_ID          ID numérico de la página de Facebook

GCP: Application Default Credentials — cuenta de servicio adjunta al
entorno de ejecución de Airflow (sin configuración adicional en el DAG).

Schedule
--------
    Por defecto corre diariamente a las 06:00 hora local.
    Ajustar `schedule_interval` según necesidad del negocio.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.scraper_facebook import task_scraping
from src.preprocessing import task_preprocessing
from src.embedding import task_embeddings
from src.classification import task_clasificacion
from src.aggregation import task_agregacion


# ── Argumentos por defecto ────────────────────────────────────────────────────
default_args = {
    "owner":            "dreammaker",
    "depends_on_past":  False,
    "email_on_failure": False,       # Cambiar a True y configurar email en producción
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id          = "dreammaker_comments_pipeline",
    description     = "Pipeline incremental NLP de comentarios Facebook — DreamMaker (GCS + BigQuery)",
    default_args    = default_args,
    start_date      = datetime(2024, 1, 1),
    schedule = "0 6 * * *",   # Diario a las 06:00
    catchup         = False,            # No ejecutar corridas históricas pendientes
    max_active_runs = 1,                # Evitar corridas paralelas sobre los mismos datos
    tags            = ["dreammaker", "nlp", "facebook", "medallion", "gcp"],
) as dag:

    # ── Task 1: Scraping → Bronze (GCS) ───────────────────────────────────────
    t1_scraping = PythonOperator(
        task_id         = "scraping_bronze",
        python_callable = task_scraping,
        doc_md          = """
        **Scraping → Bronze (Cloud Storage)**

        Extrae posts y comentarios desde la Facebook Graph API.
        Detecta comment_id ya presentes en el CSV de Bronze en GCS y
        sube solo el delta. Descarta comentarios sin texto (stickers,
        GIFs, imágenes).

        Output: `gs://<BUCKET_NAME>/bronze/facebook_comments_raw.csv`
        """,
    )

    # ── Task 2: Preprocesamiento → Silver clean (BigQuery) ────────────────────
    t2_preprocessing = PythonOperator(
        task_id         = "preprocessing_silver_csv",
        python_callable = task_preprocessing,
        doc_md          = """
        **Preprocesamiento → Silver (BigQuery)**

        Lee Bronze desde GCS, detecta comentarios no presentes en la
        tabla Silver clean de BigQuery y aplica limpieza de texto:
        lowercase, eliminación de tildes, colapso de espacios y signos
        especiales, y tipado de timestamps/contadores.

        Output: `dreammaker_silver.facebook_comments_clean`
        """,
    )

    # ── Task 3: Embeddings → Silver embeddings (BigQuery) ─────────────────────
    t3_embeddings = PythonOperator(
        task_id         = "embeddings_silver_parquet",
        python_callable = task_embeddings,
        doc_md          = """
        **Embeddings → Silver (BigQuery)**

        Lee Silver clean desde BigQuery, detecta comentarios sin
        embedding y los vectoriza con BETO
        (dccuchile/bert-base-spanish-wwm-cased), mean pooling sobre
        última capa oculta → vector 768-dim.

        Output: `dreammaker_silver.facebook_comments_embeddings`
        """,
        execution_timeout = timedelta(hours=2),   # BETO sobre miles de comentarios puede demorar
    )

    # ── Task 4: Clasificación → Gold (BigQuery) ────────────────────────────────
    t4_clasificacion = PythonOperator(
        task_id         = "clasificacion_gold",
        python_callable = task_clasificacion,
        doc_md          = """
        **Clasificación jerárquica → Gold (BigQuery)**

        Descarga los modelos entrenados desde GCS si aún no están en
        disco local. Capa 1: Logística OvR sobre embeddings → 8
        etiquetas multilabel. Capa 2: XGBoost sobre embeddings + Capa 1
        → sarcasmo binario. Reglas de negocio: interes_real,
        elogio_real, critica_genuina, solicitud_real,
        polarizacion_politica, fanatismo, fanatismo_no_comercial.

        Outputs:
            `dreammaker_gold.facebook_comments_classified`
            `dreammaker_gold.facebook_comments_probabilities`
            `dreammaker_gold.facebook_comments_gold_enriched`
        """,
    )

    # ── Task 5: Agregación → Gold post metrics (BigQuery) ──────────────────────
    t5_agregacion = PythonOperator(
        task_id         = "agregacion_gold_post_metrics",
        python_callable = task_agregacion,
        doc_md          = """
        **Agregación → Gold (métricas por publicación, BigQuery)**

        Opera sobre el Gold classified completo acumulado en BigQuery.
        Construye señales compuestas (senal_comercial, ruido_social),
        agrega por post_id con suavizado bayesiano y enriquece con
        clasificación temática por regex. Reemplaza la tabla completa
        en cada corrida (WRITE_TRUNCATE).

        Output: `dreammaker_gold.facebook_post_metrics`
        """,
    )

    # ── Dependencias ──────────────────────────────────────────────────────────
    t1_scraping >> t2_preprocessing >> t3_embeddings >> t4_clasificacion >> t5_agregacion
