"""
src/embedding.py
================
Genera embeddings BETO para clean_comment_message y clean_post_message,
y los escribe en la capa Silver de BigQuery (tabla
facebook_comments_embeddings, dataset dreammaker_silver).

El modelo se carga una sola vez al importar el módulo.

Comportamiento incremental
--------------------------
- Detecta comment_id ya presentes en la tabla de embeddings y vectoriza
  solo los nuevos.

Compatibilidad
--------------
- Modo local   : generar_embeddings(df) → pd.DataFrame
- Modo Airflow : task_embeddings(**context) → str (table_id de Silver embeddings)
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src import config
from src.gcp_io import cargar_dataframe_bq, ids_existentes_bq, leer_tabla_bq
from src.schemas import SCHEMA_SILVER_EMBEDDINGS


# ── Carga del modelo (una sola vez al importar) ───────────────────────────────
_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

print(f"Cargando modelo BETO ({_MODEL_NAME})...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model     = AutoModel.from_pretrained(_MODEL_NAME)
_model.eval()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
print(f"✔ Modelo cargado en: {_device}")


# ── Embeddings por lotes ──────────────────────────────────────────────────────
def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Genera embeddings BETO por lotes usando mean pooling sobre la
    última capa oculta.
    """
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            encoded = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(_device) for k, v in encoded.items()}
            outputs = _model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings).numpy()


def _vectorizar_y_persistir(df: pd.DataFrame) -> str:
    """
    Lógica central de embeddings. Retorna el table_id de Silver embeddings.
    Usada tanto en modo local como en modo Airflow.
    """
    if df.empty:
        print("   Sin datos nuevos para generar embeddings.")
        return config.TABLE_SILVER_EMBEDDINGS

    df = df.copy()
    df["clean_comment_message"] = df["clean_comment_message"].fillna("").astype(str)
    df["clean_post_message"]    = df["clean_post_message"].fillna("").astype(str)

    print(f"   Generando embeddings de comentarios ({len(df)} filas)...")
    comment_emb = get_embeddings_batch(df["clean_comment_message"].tolist())
    df["embedding"] = [row.tolist() for row in comment_emb]  # listas planas → BQ REPEATED FLOAT64

    print(f"   Generando embeddings de publicaciones ({len(df)} filas)...")
    post_emb = get_embeddings_batch(df["clean_post_message"].tolist())
    df["post_embedding"] = [row.tolist() for row in post_emb]

    cargar_dataframe_bq(
        df, config.TABLE_SILVER_EMBEDDINGS, SCHEMA_SILVER_EMBEDDINGS, config.DATASET_SILVER
    )

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos con embeddings insertados en Silver.")
    print(f"  Tabla: {config.TABLE_SILVER_EMBEDDINGS}")
    print(f"{'─'*45}")

    return config.TABLE_SILVER_EMBEDDINGS


# ── Modo local ────────────────────────────────────────────────────────────────
def generar_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Genera embeddings, los inserta en BigQuery y retorna el DataFrame con columnas nuevas."""
    if df.empty:
        print("   Sin datos nuevos para generar embeddings.")
        return df

    df = df.copy()
    df["clean_comment_message"] = df["clean_comment_message"].fillna("").astype(str)
    df["clean_post_message"]    = df["clean_post_message"].fillna("").astype(str)

    print(f"   Generando embeddings de comentarios ({len(df)} filas)...")
    comment_emb = get_embeddings_batch(df["clean_comment_message"].tolist())
    df["embedding"] = [row.tolist() for row in comment_emb]

    print(f"   Generando embeddings de publicaciones ({len(df)} filas)...")
    post_emb = get_embeddings_batch(df["clean_post_message"].tolist())
    df["post_embedding"] = [row.tolist() for row in post_emb]

    cargar_dataframe_bq(
        df, config.TABLE_SILVER_EMBEDDINGS, SCHEMA_SILVER_EMBEDDINGS, config.DATASET_SILVER
    )

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos con embeddings insertados en Silver.")
    print(f"  Tabla: {config.TABLE_SILVER_EMBEDDINGS}")
    print(f"{'─'*45}")

    return df


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_embeddings(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee Silver clean desde BigQuery, detecta el delta vs Silver
    embeddings, genera embeddings solo para los nuevos y hace append.
    """
    ids_embebidos = ids_existentes_bq(config.TABLE_SILVER_EMBEDDINGS, "comment_id")

    df_silver = leer_tabla_bq(config.TABLE_SILVER_CLEAN)
    if df_silver.empty:
        print("   Silver clean vacío en BigQuery.")
        return config.TABLE_SILVER_EMBEDDINGS

    df_delta = df_silver[~df_silver["comment_id"].astype(str).isin(ids_embebidos)].copy()

    print(f"Delta para embeddings: {len(df_delta)} comentarios.")
    return _vectorizar_y_persistir(df_delta)
