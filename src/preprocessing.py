"""
src/preprocessing.py
====================
Limpieza de texto, tipado de columnas y filtrado de comentarios vacíos.
Escribe en la capa Silver, en BigQuery (dataset dreammaker_silver,
tabla facebook_comments_clean).

Comportamiento incremental
--------------------------
- Detecta comment_id ya presentes en la tabla Silver de BigQuery y
  procesa/inserta solo los nuevos (WRITE_APPEND vía load job).

Compatibilidad
--------------
- Modo local   : preprocess_dataframe(df) → pd.DataFrame
- Modo Airflow : task_preprocessing(**context) → str (table_id de Silver clean)
"""

import re
import unicodedata

import pandas as pd

from src import config
from src.gcp_io import cargar_dataframe_bq, ids_existentes_bq, leer_csv_gcs
from src.schemas import SCHEMA_SILVER_CLEAN


# ── Limpieza de texto ─────────────────────────────────────────────────────────
def clean_text(text: str | None, inchar: str = " ") -> str | None:
    """
    Limpieza básica de texto:
    - Elimina signos especiales
    - Lowercase + strip
    - Elimina tildes (NFD)
    - Colapsa saltos de línea y espacios múltiples
    - Colapsa múltiples '?' en uno solo

    Retorna None si la entrada es None.
    """
    if text is None:
        return None

    signos = "()[]{}<>^*#@=+#¡!"
    for k in signos:
        text = text.replace(k, " ")

    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = text.replace("\n", " ")
    text = re.sub(r"\?{2,}", "?", text)
    text = text.replace(" ", inchar)
    text = re.sub(f"{re.escape(inchar)}+", inchar, text)

    return text


# ── Filtrado de vacíos ────────────────────────────────────────────────────────
def filtrar_comentarios_vacios(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas donde comment_message sea NaN o string vacío."""
    n_antes = len(df)

    mask = (
        df["comment_message"].notna()
        & df["comment_message"].astype(str).str.strip().ne("")
    )
    df = df[mask].copy()

    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        print(f"   Filas eliminadas (comentario vacío): {n_eliminados}")

    return df.reset_index(drop=True)


# ── Tipado ─────────────────────────────────────────────────────────────────────
def _tipar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Castea timestamps y contadores al tipo esperado por SCHEMA_SILVER_CLEAN."""
    df = df.copy()
    df["post_created"]      = pd.to_datetime(df["post_created"], utc=True, errors="coerce")
    df["comment_created"]   = pd.to_datetime(df["comment_created"], utc=True, errors="coerce")
    df["post_reactions"]    = pd.to_numeric(df["post_reactions"], errors="coerce").fillna(0).astype(int)
    df["comment_reactions"] = pd.to_numeric(df["comment_reactions"], errors="coerce").fillna(0).astype(int)
    return df


def _limpiar_y_persistir(df: pd.DataFrame) -> str:
    """
    Lógica central de preprocesamiento. Retorna el table_id de Silver clean.
    Usada tanto en modo local como en modo Airflow.
    """
    if df.empty:
        print("   Sin datos nuevos para preprocesar.")
        return config.TABLE_SILVER_CLEAN

    df = filtrar_comentarios_vacios(df)

    _clean_col = lambda x: clean_text(str(x), " ")
    df["clean_comment_message"] = df["comment_message"].apply(_clean_col)
    df["clean_post_message"]    = df["post_message"].apply(_clean_col)

    df = _tipar_columnas(df)

    cargar_dataframe_bq(df, config.TABLE_SILVER_CLEAN, SCHEMA_SILVER_CLEAN, config.DATASET_SILVER)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos insertados en Silver (BigQuery).")
    print(f"  Tabla: {config.TABLE_SILVER_CLEAN}")
    print(f"{'─'*45}")

    return config.TABLE_SILVER_CLEAN


# ── Modo local ────────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el delta recibido, lo inserta en BigQuery y retorna el DataFrame limpio."""
    if df.empty:
        print("   Sin datos nuevos para preprocesar.")
        return df

    df = filtrar_comentarios_vacios(df)

    _clean_col = lambda x: clean_text(str(x), " ")
    df["clean_comment_message"] = df["comment_message"].apply(_clean_col)
    df["clean_post_message"]    = df["post_message"].apply(_clean_col)

    df = _tipar_columnas(df)

    cargar_dataframe_bq(df, config.TABLE_SILVER_CLEAN, SCHEMA_SILVER_CLEAN, config.DATASET_SILVER)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos insertados en Silver (BigQuery).")
    print(f"  Tabla: {config.TABLE_SILVER_CLEAN}")
    print(f"{'─'*45}")

    return df


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_preprocessing(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee Bronze desde GCS, detecta el delta contra Silver clean en
    BigQuery, procesa y hace append.
    """
    ids_silver = ids_existentes_bq(config.TABLE_SILVER_CLEAN, "comment_id")

    df_bronze = leer_csv_gcs(config.GCS_BRONZE_BLOB)
    if df_bronze.empty:
        print("   Bronze vacío en GCS.")
        return config.TABLE_SILVER_CLEAN

    df_delta = df_bronze[~df_bronze["comment_id"].astype(str).isin(ids_silver)].copy()

    print(f"Delta para preprocesar: {len(df_delta)} comentarios.")
    return _limpiar_y_persistir(df_delta)
