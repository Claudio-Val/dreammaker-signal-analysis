"""
src/classification.py
=====================
Pipeline jerárquico de clasificación en dos capas con reglas de negocio.

    Capa 1 — Regresión Logística OvR (multilabel)
              Predice: interes, elogio_producto, critica_producto,
                       fanatismo_emocional, conflictivo, otro,
                       solicitud, cuidado

    Capa 2 — XGBoost (binario)
              Predice: sarcasmo
              Input  : embeddings + predicciones Capa 1

Lee embeddings desde Silver (BigQuery) y escribe en Gold (BigQuery):
    facebook_comments_classified, facebook_comments_probabilities,
    facebook_comments_gold_enriched.

Modelos entrenados
-------------------
Los artefactos .pkl NO se versionan en el repo. Se descargan desde
Cloud Storage (gs://<BUCKET_NAME>/<MODELS_PATH>/) a disco local la
primera vez que se necesitan (ver gcp_io.descargar_blob_si_falta), y se
reutilizan en corridas siguientes si ya están presentes en ./models/.

Compatibilidad
--------------
- Modo local   : predecir_labels(df) → tuple[pd.DataFrame, ...]
- Modo Airflow : task_clasificacion(**context) → str (table_id de Gold classified)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.gcp_io import (
    cargar_dataframe_bq,
    descargar_blob_si_falta,
    ids_existentes_bq,
    leer_tabla_bq,
)
from src.schemas import SCHEMA_GOLD_CLASSIFIED, SCHEMA_GOLD_ENRICHED, SCHEMA_GOLD_PROBABILITIES


# ── Constantes ────────────────────────────────────────────────────────────────
_FEATURE_LABELS = [
    "interes",
    "elogio_producto",
    "critica_producto",
    "fanatismo_emocional",
    "conflictivo",
    "otro",
    "solicitud",
    "cuidado",
]

_MODEL_FILES = {
    "capa1_model":      "logistic_ovr_model.pkl",
    "capa1_thresholds": "logistic_ovr_thresholds.pkl",
    "capa2_model":      "xgb_sarcasm_model.pkl",
    "capa2_threshold":  "xgb_sarcasm_threshold.pkl",
}
_MODEL_PATHS = {nombre: Path("models") / archivo for nombre, archivo in _MODEL_FILES.items()}


# ── Carga de modelos ──────────────────────────────────────────────────────────
def _cargar_modelos() -> tuple:
    """
    Descarga los 4 artefactos del pipeline jerárquico desde GCS a disco
    local si aún no están presentes, y los carga con joblib.
    """
    for nombre, archivo in _MODEL_FILES.items():
        descargar_blob_si_falta(f"{config.MODELS_PATH}/{archivo}", _MODEL_PATHS[nombre])

    return (
        joblib.load(_MODEL_PATHS["capa1_model"]),
        joblib.load(_MODEL_PATHS["capa1_thresholds"]),
        joblib.load(_MODEL_PATHS["capa2_model"]),
        joblib.load(_MODEL_PATHS["capa2_threshold"]),
    )


# ── Capa 1 ────────────────────────────────────────────────────────────────────
def _predecir_capa1(model, thresholds, X_emb, df_res, df_probas):
    y_proba = model.predict_proba(X_emb)
    for i, label in enumerate(_FEATURE_LABELS):
        probs = y_proba[i][:, 1]
        df_res[f"pred_{label}"]     = (probs >= thresholds[label]).astype(int)
        df_probas[f"proba_{label}"] = probs
    return df_res, df_probas


# ── Capa 2 ────────────────────────────────────────────────────────────────────
def _predecir_capa2(model, threshold, X_emb, df_res, df_probas):
    X_labels   = df_res[[f"pred_{l}" for l in _FEATURE_LABELS]].astype(int).values
    X_sarcasmo = np.hstack([X_emb, X_labels])
    proba      = model.predict_proba(X_sarcasmo)[:, 1]
    df_res["pred_sarcasmo"]     = (proba >= threshold).astype(int)
    df_probas["proba_sarcasmo"] = proba
    return df_res, df_probas


# ── Reglas de negocio ─────────────────────────────────────────────────────────
def _aplicar_reglas_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva etiquetas interpretables a partir de las predicciones binarias.

    Reglas
    ------
    solicitud_real        : solicitud=1 AND sarcasmo=0
    interes_real          : interes=1 AND sarcasmo=0
    elogio_real           : elogio_producto=1 AND sarcasmo=0
    critica_genuina       : critica_producto=1 AND sarcasmo=0 AND conflictivo=0
    polarizacion_politica : conflictivo=1
    fanatismo              : fanatismo_emocional=1 AND sarcasmo=0
    fanatismo_no_comercial: fanatismo_emocional=1 AND sarcasmo=0 AND interes=0
    """
    d = df.copy()

    d["solicitud_real"] = (
        (d["pred_solicitud"] == 1) &
        (d["pred_sarcasmo"]  == 0)
    ).astype(int)

    d["interes_real"] = (
        (d["pred_interes"]  == 1) &
        (d["pred_sarcasmo"] == 0)
    ).astype(int)

    d["elogio_real"] = (
        (d["pred_elogio_producto"] == 1) &
        (d["pred_sarcasmo"]        == 0)
    ).astype(int)

    d["critica_genuina"] = (
        (d["pred_critica_producto"] == 1) &
        (d["pred_sarcasmo"]         == 0) &
        (d["pred_conflictivo"]      == 0)
    ).astype(int)

    d["polarizacion_politica"] = (
        d["pred_conflictivo"] == 1
    ).astype(int)

    d["fanatismo"] = (
        (d["pred_fanatismo_emocional"] == 1) &
        (d["pred_sarcasmo"]            == 0)
    ).astype(int)

    d["fanatismo_no_comercial"] = (
        (d["pred_fanatismo_emocional"] == 1) &
        (d["pred_sarcasmo"]            == 0) &
        (d["pred_interes"]             == 0)
    ).astype(int)

    return d


def _clasificar_y_persistir(df: pd.DataFrame) -> str:
    """
    Lógica central de clasificación. Retorna el table_id de Gold classified.
    Usada tanto en modo local como en modo Airflow.
    """
    if df.empty:
        print("   Sin datos nuevos para clasificar.")
        return config.TABLE_GOLD_CLASSIFIED

    model_c1, thr_c1, model_c2, thr_c2 = _cargar_modelos()
    print("   ✔ Modelos cargados.")

    # .apply(np.array) normaliza tanto listas (recién generadas en embedding.py)
    # como arrays devueltos por BigQuery para columnas REPEATED FLOAT64.
    X_emb        = np.vstack(df["embedding"].apply(np.array).values)
    df_resultado = df.copy()
    df_probas    = pd.DataFrame()

    if "comment_id" in df.columns:
        df_probas["comment_id"] = df["comment_id"].values

    print("   Ejecutando Capa 1 (multilabel OvR)...")
    df_resultado, df_probas = _predecir_capa1(model_c1, thr_c1, X_emb, df_resultado, df_probas)

    print("   Ejecutando Capa 2 (sarcasmo XGBoost)...")
    df_resultado, df_probas = _predecir_capa2(model_c2, thr_c2, X_emb, df_resultado, df_probas)

    print("   Aplicando reglas de negocio...")
    df_gold_enriched = _aplicar_reglas_negocio(df_resultado)

    cargar_dataframe_bq(df_resultado,     config.TABLE_GOLD_CLASSIFIED,    SCHEMA_GOLD_CLASSIFIED,    config.DATASET_GOLD)
    cargar_dataframe_bq(df_probas,        config.TABLE_GOLD_PROBABILITIES, SCHEMA_GOLD_PROBABILITIES, config.DATASET_GOLD)
    cargar_dataframe_bq(df_gold_enriched, config.TABLE_GOLD_ENRICHED,      SCHEMA_GOLD_ENRICHED,      config.DATASET_GOLD)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos clasificados e insertados en Gold (BigQuery).")
    print(f"{'─'*45}")

    return config.TABLE_GOLD_CLASSIFIED


# ── Modo local ────────────────────────────────────────────────────────────────
def predecir_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clasifica el delta, lo inserta en Gold (BigQuery) y retorna tupla de DataFrames."""
    if df.empty:
        print("   Sin datos nuevos para clasificar.")
        return df, pd.DataFrame(), pd.DataFrame()

    model_c1, thr_c1, model_c2, thr_c2 = _cargar_modelos()
    print("   ✔ Modelos cargados.")

    X_emb        = np.vstack(df["embedding"].apply(np.array).values)
    df_resultado = df.copy()
    df_probas    = pd.DataFrame()

    if "comment_id" in df.columns:
        df_probas["comment_id"] = df["comment_id"].values

    print("   Ejecutando Capa 1 (multilabel OvR)...")
    df_resultado, df_probas = _predecir_capa1(model_c1, thr_c1, X_emb, df_resultado, df_probas)

    print("   Ejecutando Capa 2 (sarcasmo XGBoost)...")
    df_resultado, df_probas = _predecir_capa2(model_c2, thr_c2, X_emb, df_resultado, df_probas)

    print("   Aplicando reglas de negocio...")
    df_gold_enriched = _aplicar_reglas_negocio(df_resultado)

    cargar_dataframe_bq(df_resultado,     config.TABLE_GOLD_CLASSIFIED,    SCHEMA_GOLD_CLASSIFIED,    config.DATASET_GOLD)
    cargar_dataframe_bq(df_probas,        config.TABLE_GOLD_PROBABILITIES, SCHEMA_GOLD_PROBABILITIES, config.DATASET_GOLD)
    cargar_dataframe_bq(df_gold_enriched, config.TABLE_GOLD_ENRICHED,      SCHEMA_GOLD_ENRICHED,      config.DATASET_GOLD)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos clasificados e insertados en Gold (BigQuery).")
    print(f"{'─'*45}")

    return df_resultado, df_probas, df_gold_enriched


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_clasificacion(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee Silver embeddings desde BigQuery, detecta el delta vs Gold
    classified, clasifica solo los nuevos y hace append.
    """
    ids_clasificados = ids_existentes_bq(config.TABLE_GOLD_CLASSIFIED, "comment_id")

    df_silver = leer_tabla_bq(config.TABLE_SILVER_EMBEDDINGS)
    if df_silver.empty:
        print("   Silver embeddings vacío en BigQuery.")
        return config.TABLE_GOLD_CLASSIFIED

    df_delta = df_silver[~df_silver["comment_id"].astype(str).isin(ids_clasificados)].copy()

    print(f"Delta para clasificar: {len(df_delta)} comentarios.")
    return _clasificar_y_persistir(df_delta)
