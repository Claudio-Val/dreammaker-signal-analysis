"""
pipeline.py
===========
Pipeline incremental de análisis de comentarios de Facebook para
e-commerce de figuras coleccionables chilenas.

Implementa una arquitectura Medallion (Bronze → Silver → Gold) y un
clasificador jerárquico en dos capas para detección multilabel de
intenciones y sarcasmo condicionado en comentarios en español chileno.

Comportamiento incremental
--------------------------
En cada ejecución el pipeline detecta automáticamente qué comentarios
son nuevos respecto al estado anterior y procesa únicamente esos,
haciendo append en cada capa. Si es la primera ejecución, procesa
y persiste todo desde cero.

Etapas
------
1. Scraping        : extrae desde Facebook Graph API             → Bronze
2. Preprocesamiento: limpieza de texto, filtrado de vacíos       → Silver (CSV)
3. Embeddings      : vectorización con BETO (768-dim)            → Silver (Parquet)
4. Clasificación   : pipeline jerárquico OvR + XGBoost           → Gold (comentarios)
5. Agregación      : métricas por post con suavizado bayesiano   → Gold (posts)

Arquitectura del clasificador
-----------------------------
    Capa 1 — Regresión Logística OvR (multilabel)
              Etiquetas: interes, elogio_producto, critica_producto,
                         fanatismo_emocional, conflictivo, otro,
                         solicitud, cuidado
              Thresholds optimizados por etiqueta.

    Capa 2 — XGBoost (binario)
              Etiqueta : sarcasmo
              Input    : embedding comentario + predicciones Capa 1

    Reglas de negocio:
              solicitud_real, interes_real, elogio_real,
              critica_genuina, polarizacion_politica,
              fanatismo, fanatismo_no_comercial

Salidas en Gold
---------------
    facebook_comments_classified.parquet    predicciones binarias por comentario
    facebook_comments_probabilities.parquet probabilidades por comentario
    facebook_comments_gold_enriched.parquet reglas de negocio por comentario
    facebook_post_metrics.parquet           métricas agregadas por publicación

Uso
---
    python pipeline.py

Requisitos
----------
    Archivo my_env.env con:
        FACEBOOK_TOKEN=<token>
        PAGE_ID=<page_id>

    Modelos entrenados en models/:
        logistic_ovr_model.pkl
        logistic_ovr_thresholds.pkl
        xgb_sarcasm_model.pkl
        xgb_sarcasm_threshold.pkl
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
    """Crea la estructura Medallion de carpetas si no existe."""
    for subdir in ["data/bronze", "data/silver", "data/gold", "models", "outputs"]:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


# ── Pipeline principal ────────────────────────────────────────────────────────
def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline incremental completo de extremo a extremo.

    Detecta automáticamente los comentarios nuevos en cada etapa
    y procesa únicamente esos, haciendo append en Bronze, Silver y Gold.
    Las métricas por post (paso 5) siempre se recalculan sobre el Gold
    completo acumulado para mantener el prior bayesiano consistente.

    Retorna
    -------
    df_gold          : pd.DataFrame  predicciones binarias (solo nuevos)
    df_probas        : pd.DataFrame  probabilidades por etiqueta (solo nuevos)
    df_gold_enriched : pd.DataFrame  etiquetas de negocio (solo nuevos)
    df_post_metrics  : pd.DataFrame  métricas agregadas por publicación (completo)
    """
    _crear_estructura_directorios(os.getcwd())

    # ──────────────────────────────────────────────────────────────────
    # 1. SCRAPING  →  Bronze
    #    Extrae desde la API y retorna solo los comment_id nuevos
    #    con texto válido (excluye stickers, GIFs e imágenes).
    #    Si Bronze no existe, retorna todo. Si ya está al día, df vacío.
    # ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Extrayendo comentarios nuevos desde Facebook...")
    df = obtener_datos_facebook(
        access_token=ACCESS_TOKEN,
        page_id=PAGE_ID,
    )

    # ──────────────────────────────────────────────────────────────────
    # 2. PREPROCESAMIENTO  →  Silver CSV
    #    Limpia solo el delta recibido y hace append al Silver CSV.
    # ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Preprocesando comentarios nuevos...")
    df = preprocess_dataframe(df)

    # ──────────────────────────────────────────────────────────────────
    # 3. EMBEDDINGS  →  Silver Parquet
    #    Genera embeddings BETO solo para el delta y hace append.
    # ──────────────────────────────────────────────────────────────────
    print("\n[3/5] Generando embeddings BETO...")
    df = generar_embeddings(df)

    # ──────────────────────────────────────────────────────────────────
    # 4. CLASIFICACIÓN JERÁRQUICA + REGLAS DE NEGOCIO  →  Gold
    #    Clasifica solo el delta y hace append a los tres Parquet Gold.
    # ──────────────────────────────────────────────────────────────────
    print("\n[4/5] Clasificando comentarios nuevos (pipeline jerárquico)...")
    df_gold, df_probas, df_gold_enriched = predecir_labels(df)

    # ──────────────────────────────────────────────────────────────────
    # 5. AGREGACIÓN POR POST  →  Gold (métricas)
    #    Siempre opera sobre el Gold completo acumulado para que el
    #    prior bayesiano sea consistente con todos los datos disponibles.
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