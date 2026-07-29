"""
dags/dreammaker_dag.py
======================
DAG de Airflow para el pipeline incremental de análisis de comentarios
de Facebook de DreamMaker.

Arquitectura Medallion:
    Bronze  → data/bronze/   (raw CSV de la API)
    Silver  → data/silver/   (limpio + embeddings BETO)
    Gold    → data/gold/     (predicciones + métricas por post)

Cada task es atómica e idempotente:
    - Lee su input desde disco (capa anterior)
    - Detecta el delta respecto a lo ya procesado
    - Escribe su output a disco (capa actual)
    - Retorna el path del output via XCom

Credenciales requeridas en Airflow Variables:
    FACEBOOK_TOKEN   token de acceso Facebook Graph API
    PAGE_ID          ID numérico de la página de Facebook

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
    description     = "Pipeline incremental NLP de comentarios Facebook — DreamMaker",
    default_args    = default_args,
    start_date      = datetime(2024, 1, 1),
    schedule = "0 6 * * *",   # Diario a las 06:00
    catchup         = False,            # No ejecutar corridas históricas pendientes
    max_active_runs = 1,                # Evitar corridas paralelas sobre los mismos datos
    tags            = ["dreammaker", "nlp", "facebook", "medallion"],
) as dag:

    # ── Task 1: Scraping → Bronze ─────────────────────────────────────────────
    t1_scraping = PythonOperator(
        task_id         = "scraping_bronze",
        python_callable = task_scraping,
        doc_md          = """
        **Scraping → Bronze**

        Extrae posts y comentarios desde la Facebook Graph API.
        Detecta comment_id ya presentes en Bronze y persiste solo el delta.
        Descarta comentarios sin texto (stickers, GIFs, imágenes).

        Output: `data/bronze/facebook_comments_raw.csv`
        """,
    )

    # ── Task 2: Preprocesamiento → Silver CSV ─────────────────────────────────
    t2_preprocessing = PythonOperator(
        task_id         = "preprocessing_silver_csv",
        python_callable = task_preprocessing,
        doc_md          = """
        **Preprocesamiento → Silver CSV**

        Lee Bronze, detecta comentarios no presentes en Silver y aplica
        limpieza de texto: lowercase, eliminación de tildes, colapso
        de espacios y signos especiales.

        Output: `data/silver/facebook_comments_clean.csv`
        """,
    )

    # ── Task 3: Embeddings → Silver Parquet ───────────────────────────────────
    t3_embeddings = PythonOperator(
        task_id         = "embeddings_silver_parquet",
        python_callable = task_embeddings,
        doc_md          = """
        **Embeddings → Silver Parquet**

        Lee Silver CSV, detecta comentarios sin embedding y los vectoriza
        con BETO (dccuchile/bert-base-spanish-wwm-cased), mean pooling
        sobre última capa oculta → vector 768-dim.

        Output: `data/silver/facebook_comments_embeddings.parquet`
        """,
        execution_timeout = timedelta(hours=2),   # BETO sobre miles de comentarios puede demorar
    )

    # ── Task 4: Clasificación → Gold ──────────────────────────────────────────
    t4_clasificacion = PythonOperator(
        task_id         = "clasificacion_gold",
        python_callable = task_clasificacion,
        doc_md          = """
        **Clasificación jerárquica → Gold**

        Capa 1: Logística OvR sobre embeddings → 8 etiquetas multilabel.
        Capa 2: XGBoost sobre embeddings + Capa 1 → sarcasmo binario.
        Reglas de negocio: interes_real, elogio_real, critica_genuina,
        solicitud_real, polarizacion_politica, fanatismo, fanatismo_no_comercial.

        Outputs:
            `data/gold/facebook_comments_classified.parquet`
            `data/gold/facebook_comments_probabilities.parquet`
            `data/gold/facebook_comments_gold_enriched.parquet`
        """,
    )

    # ── Task 5: Agregación → Gold (post metrics) ──────────────────────────────
    t5_agregacion = PythonOperator(
        task_id         = "agregacion_gold_post_metrics",
        python_callable = task_agregacion,
        doc_md          = """
        **Agregación → Gold (métricas por publicación)**

        Opera sobre el Gold classified completo acumulado.
        Construye señales compuestas (senal_comercial, ruido_social),
        agrega por post_id con suavizado bayesiano y enriquece con
        clasificación temática por regex.

        Output: `data/gold/facebook_post_metrics.parquet`
        """,
    )

    # ── Dependencias ──────────────────────────────────────────────────────────
    t1_scraping >> t2_preprocessing >> t3_embeddings >> t4_clasificacion >> t5_agregacion
