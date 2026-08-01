"""
src/config.py
==============
Configuración central del proyecto: identificadores de GCP, datasets,
tablas de BigQuery y rutas del bucket de Cloud Storage. Ningún otro
módulo debe hardcodear estos valores — todos se importan desde acá.

Arquitectura de almacenamiento
-------------------------------
    Bronze   → Cloud Storage   (CSV crudo, tal como llega de la API)
    Silver   → BigQuery        (dataset dreammaker_silver)
    Gold     → BigQuery        (dataset dreammaker_gold)
    Modelos  → Cloud Storage   (artefactos .pkl entrenados, versionados)
    Reportes → Cloud Storage   (gráficos / CSV de análisis exploratorio)

Autenticación
--------------
Application Default Credentials (ADC) — no se hardcodea ninguna ruta de
credenciales en el código. La resuelve la propia librería de Google Cloud,
en este orden:
  1. Variable de entorno GOOGLE_APPLICATION_CREDENTIALS, apuntando a un
     JSON de cuenta de servicio — definida en my_env.env para modo local.
  2. Cuenta de servicio adjunta al entorno de ejecución (Airflow / Cloud
     Run / VM) — sin configuración adicional en producción.
  3. `gcloud auth application-default login` — para desarrollo interactivo.
"""

import os

# ── Proyecto GCP ────────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "learned-surge-481419-p1")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")

# ── Cloud Storage ────────────────────────────────────────────────────────────
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "dreammaker-mlops")

# Bronze vive en GCS como CSV crudo (no en BigQuery).
DATASET_BRONZE  = "bronze"
GCS_BRONZE_BLOB = f"{DATASET_BRONZE}/facebook_comments_raw.csv"

# Artefactos de modelo y reportes, también en GCS (prefijos dentro del bucket).
MODELS_PATH  = "models"
REPORTS_PATH = "reports"

# ── BigQuery ──────────────────────────────────────────────────────────────────
DATASET_SILVER = "dreammaker_silver"
DATASET_GOLD   = "dreammaker_gold"

TABLE_SILVER_CLEAN      = f"{PROJECT_ID}.{DATASET_SILVER}.facebook_comments_clean"
TABLE_SILVER_EMBEDDINGS = f"{PROJECT_ID}.{DATASET_SILVER}.facebook_comments_embeddings"

TABLE_GOLD_CLASSIFIED    = f"{PROJECT_ID}.{DATASET_GOLD}.facebook_comments_classified"
TABLE_GOLD_PROBABILITIES = f"{PROJECT_ID}.{DATASET_GOLD}.facebook_comments_probabilities"
TABLE_GOLD_ENRICHED      = f"{PROJECT_ID}.{DATASET_GOLD}.facebook_comments_gold_enriched"
TABLE_GOLD_POST_METRICS  = f"{PROJECT_ID}.{DATASET_GOLD}.facebook_post_metrics"
