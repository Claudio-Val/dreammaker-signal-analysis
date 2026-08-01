"""
pipeline.py
===========
Pipeline incremental de análisis de comentarios de Facebook para
e-commerce de figuras coleccionables chilenas.

Implementa una arquitectura Medallion (Bronze → Silver → Gold) y un
clasificador jerárquico en dos capas para detección multilabel de
intenciones y sarcasmo condicionado en comentarios en español chileno.

Almacenamiento
--------------
    Bronze  → Cloud Storage  (CSV crudo)
    Silver  → BigQuery       (dataset dreammaker_silver)
    Gold    → BigQuery       (dataset dreammaker_gold)

Ver src/config.py para los identificadores de proyecto/bucket/datasets,
y el README para el detalle de la migración desde almacenamiento local.

Comportamiento incremental
--------------------------
En cada ejecución el pipeline detecta automáticamente qué comentarios
son nuevos respecto al estado anterior (comparando IDs contra la capa
siguiente en GCS/BigQuery) y procesa únicamente esos. Si es la primera
ejecución, procesa y persiste todo desde cero.

Etapas
------
1. Scraping        : extrae desde Facebook Graph API             → Bronze (GCS)
2. Preprocesamiento: limpieza de texto, filtrado de vacíos       → Silver (BigQuery)
3. Embeddings      : vectorización con BETO (768-dim)            → Silver (BigQuery)
4. Clasificación   : pipeline jerárquico OvR + XGBoost           → Gold (BigQuery)
5. Agregación      : métricas por post con suavizado bayesiano   → Gold (BigQuery)

Uso
---
    python pipeline.py

Requisitos
----------
    Archivo my_env.env con:
        FACEBOOK_TOKEN=<token>
        PAGE_ID=<page_id>
        GOOGLE_APPLICATION_CREDENTIALS=<ruta al JSON de la cuenta de servicio>

    Modelos entrenados, subidos a gs://<BUCKET_NAME>/models/:
        logistic_ovr_model.pkl
        logistic_ovr_thresholds.pkl
        xgb_sarcasm_model.pkl
        xgb_sarcasm_threshold.pkl
    (se descargan automáticamente a ./models/ la primera vez que se usan)
"""

import os
import shutil
import sys

# ── Limpiar caché de módulos src ──────────────────────────────────────────────
# Garantiza que se usen los .py actuales y no versiones compiladas en caché.
# Debe ejecutarse antes de cualquier import de src/.
_cache = os.path.join("src", "__pycache__")
if os.path.exists(_cache):
    shutil.rmtree(_cache)

for _mod in list(sys.modules.keys()):
    if _mod.startswith("src"):
        del sys.modules[_mod]

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv

from src.aggregation import generar_metricas_post
from src.classification import predecir_labels
from src.embedding import generar_embeddings
from src.preprocessing import preprocess_dataframe
from src.scraper_facebook import obtener_datos_facebook


# ── Variables de entorno ──────────────────────────────────────────────────────
load_dotenv("my_env.env", override=True)

ACCESS_TOKEN = os.getenv("FACEBOOK_TOKEN")
PAGE_ID      = os.getenv("PAGE_ID")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _crear_estructura_directorios(base_path: str) -> None:
    """
    Crea las carpetas locales todavía necesarias: modelos descargados
    desde GCS y outputs de análisis exploratorio. Bronze/Silver/Gold ya
    no viven en disco local (ver GCS/BigQuery en src/config.py), así que
    no se crean data/bronze, data/silver ni data/gold.
    """
    for subdir in ["models", "outputs"]:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


# ── Pipeline principal ────────────────────────────────────────────────────────
def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline incremental completo de extremo a extremo.

    Detecta automáticamente los comentarios nuevos en cada etapa y
    procesa únicamente esos, haciendo append en Bronze (GCS) y en
    Silver/Gold (BigQuery). Las métricas por post (paso 5) siempre se
    recalculan sobre el Gold completo acumulado para mantener el prior
    bayesiano consistente.

    Retorna
    -------
    df_gold          : pd.DataFrame  predicciones binarias (solo nuevos)
    df_probas        : pd.DataFrame  probabilidades por etiqueta (solo nuevos)
    df_gold_enriched : pd.DataFrame  etiquetas de negocio (solo nuevos)
    df_post_metrics  : pd.DataFrame  métricas agregadas por publicación (completo)
    """
    _crear_estructura_directorios(os.getcwd())

    # ──────────────────────────────────────────────────────────────────
    # 1. SCRAPING  →  Bronze (GCS)
    # ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Extrayendo comentarios nuevos desde Facebook...")
    df = obtener_datos_facebook(
        access_token=ACCESS_TOKEN,
        page_id=PAGE_ID,
    )

    # ──────────────────────────────────────────────────────────────────
    # 2. PREPROCESAMIENTO  →  Silver (BigQuery)
    # ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Preprocesando comentarios nuevos...")
    df = preprocess_dataframe(df)

    # ──────────────────────────────────────────────────────────────────
    # 3. EMBEDDINGS  →  Silver (BigQuery)
    # ──────────────────────────────────────────────────────────────────
    print("\n[3/5] Generando embeddings BETO...")
    df = generar_embeddings(df)

    # ──────────────────────────────────────────────────────────────────
    # 4. CLASIFICACIÓN JERÁRQUICA + REGLAS DE NEGOCIO  →  Gold (BigQuery)
    # ──────────────────────────────────────────────────────────────────
    print("\n[4/5] Clasificando comentarios nuevos (pipeline jerárquico)...")
    df_gold, df_probas, df_gold_enriched = predecir_labels(df)

    # ──────────────────────────────────────────────────────────────────
    # 5. AGREGACIÓN POR POST  →  Gold (BigQuery)
    # ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Calculando métricas por publicación...")
    df_post_metrics = generar_metricas_post(df_gold_enriched)

    if df_gold.empty:
        print("\n✅ Pipeline finalizado. No había comentarios nuevos.")
        print(f"   Métricas recalculadas para {len(df_post_metrics)} publicaciones.")
    else:
        print(f"\n✅ Pipeline incremental finalizado.")
        print(f"   Comentarios nuevos procesados : {len(df_gold)}")
        print(f"   Publicaciones con métricas    : {len(df_post_metrics)}")

    return df_gold, df_probas, df_gold_enriched, df_post_metrics


if __name__ == "__main__":
    main()
