"""
src/embedding.py
================
Genera embeddings BETO para clean_comment_message y clean_post_message.

El modelo se carga una sola vez al importar el módulo.

Comportamiento incremental
--------------------------
- Primera ejecución : crea el Parquet Silver desde cero.
- Ejecuciones siguientes: genera embeddings solo para comentarios nuevos
  y hace append al Parquet existente.

Compatibilidad
--------------
- Modo local   : generar_embeddings(df) → pd.DataFrame
- Modo Airflow : task_embeddings(**context) → str (path Silver Parquet)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Carga del modelo (una sola vez al importar) ───────────────────────────────
_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

print(f"Cargando modelo BETO ({_MODEL_NAME})...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model     = AutoModel.from_pretrained(_MODEL_NAME)
_model.eval()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
print(f"✔ Modelo cargado en: {_device}")

_SILVER_CSV     = Path("data/silver/facebook_comments_clean.csv")
_SILVER_PARQUET = Path("data/silver/facebook_comments_embeddings.parquet")


# ── Embeddings por lotes ──────────────────────────────────────────────────────
def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Genera embeddings BETO por lotes usando mean pooling."""
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            encoded = _tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            encoded  = {k: v.to(_device) for k, v in encoded.items()}
            outputs  = _model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings).numpy()


def _vectorizar_y_persistir(df: pd.DataFrame) -> str:
    """
    Lógica central de embeddings. Retorna el path del Silver Parquet.
    Usada tanto en modo local como en modo Airflow.
    """
    if df.empty:
        print("   Sin datos nuevos para generar embeddings.")
        return str(_SILVER_PARQUET)

    df = df.copy()
    df["clean_comment_message"] = df["clean_comment_message"].fillna("").astype(str)
    df["clean_post_message"]    = df["clean_post_message"].fillna("").astype(str)

    print(f"   Generando embeddings de comentarios ({len(df)} filas)...")
    df["embedding"] = list(get_embeddings_batch(df["clean_comment_message"].tolist()))

    print(f"   Generando embeddings de publicaciones ({len(df)} filas)...")
    df["post_embedding"] = list(get_embeddings_batch(df["clean_post_message"].tolist()))

    _SILVER_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if _SILVER_PARQUET.exists():
        df_total = pd.concat([pd.read_parquet(_SILVER_PARQUET), df], ignore_index=True)
    else:
        df_total = df

    df_total.to_parquet(_SILVER_PARQUET, index=False)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos con embeddings agregados.")
    print(f"  Total acumulado en Silver: {len(df_total)} registros.")
    print(f"  Archivo: {_SILVER_PARQUET}")
    print(f"{'─'*45}")

    return str(_SILVER_PARQUET)


# ── Modo local ────────────────────────────────────────────────────────────────
def generar_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Genera embeddings y retorna DataFrame con columnas nuevas. Uso: pipeline.py local."""
    if df.empty:
        print("   Sin datos nuevos para generar embeddings.")
        return df

    df = df.copy()
    df["clean_comment_message"] = df["clean_comment_message"].fillna("").astype(str)
    df["clean_post_message"]    = df["clean_post_message"].fillna("").astype(str)

    print(f"   Generando embeddings de comentarios ({len(df)} filas)...")
    df["embedding"] = list(get_embeddings_batch(df["clean_comment_message"].tolist()))

    print(f"   Generando embeddings de publicaciones ({len(df)} filas)...")
    df["post_embedding"] = list(get_embeddings_batch(df["clean_post_message"].tolist()))

    _SILVER_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if _SILVER_PARQUET.exists():
        df_total = pd.concat([pd.read_parquet(_SILVER_PARQUET), df], ignore_index=True)
    else:
        df_total = df

    df_total.to_parquet(_SILVER_PARQUET, index=False)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos con embeddings agregados.")
    print(f"  Total acumulado en Silver: {len(df_total)} registros.")
    print(f"  Archivo: {_SILVER_PARQUET}")
    print(f"{'─'*45}")

    return df


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_embeddings(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee desde Silver CSV, detecta delta vs Parquet existente,
    genera embeddings solo para los nuevos y hace append.
    Retorna el path del Silver Parquet para XCom.
    """
    # IDs ya embebidos
    ids_embebidos: set = set()
    if _SILVER_PARQUET.exists():
        df_emb = pd.read_parquet(_SILVER_PARQUET, columns=["comment_id"])
        ids_embebidos = set(df_emb["comment_id"].astype(str).unique())

    df_silver = pd.read_csv(str(_SILVER_CSV))

    df_delta = df_silver[
        ~df_silver["comment_id"].astype(str).isin(ids_embebidos)
    ].copy()

    print(f"Delta para embeddings: {len(df_delta)} comentarios.")
    return _vectorizar_y_persistir(df_delta)
