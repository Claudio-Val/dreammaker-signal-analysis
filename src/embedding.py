"""
src/embedding.py
================
Genera embeddings BETO para clean_comment_message y clean_post_message.

El modelo se carga una sola vez al importar el módulo.

Comportamiento incremental
--------------------------
- Primera ejecución : crea el Parquet Silver desde cero.
- Ejecuciones siguientes: carga el Parquet existente, genera embeddings
  solo para los comentarios nuevos y hace append.
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

_SILVER_PARQUET = Path("data/silver/facebook_comments_embeddings.parquet")


# ── Embeddings por lotes ──────────────────────────────────────────────────────
def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Genera embeddings BETO por lotes usando mean pooling sobre la
    última capa oculta.

    Parámetros
    ----------
    texts : list[str]
        Lista de textos a codificar.
    batch_size : int
        Tamaño del lote. Ajustar según VRAM disponible.

    Retorna
    -------
    np.ndarray  shape (n_textos, 768)
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


# ── Función pública ───────────────────────────────────────────────────────────
def generar_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera embeddings BETO para comentarios y publicaciones nuevos
    y hace append al Parquet Silver (o lo crea en primera ejecución).

    Parámetros
    ----------
    df : pd.DataFrame
        Delta de comentarios nuevos limpios (salida de preprocess_dataframe).
        Si viene vacío, retorna inmediatamente sin hacer nada.

    Retorna
    -------
    pd.DataFrame
        Solo los comentarios nuevos con columnas 'embedding' y 'post_embedding'.
    """
    if df.empty:
        print("   Sin datos nuevos para generar embeddings.")
        return df

    df = df.copy()

    df["clean_comment_message"] = df["clean_comment_message"].fillna("").astype(str)
    df["clean_post_message"]    = df["clean_post_message"].fillna("").astype(str)

    # Embeddings de comentarios
    print(f"   Generando embeddings de comentarios ({len(df)} filas)...")
    comment_emb = get_embeddings_batch(df["clean_comment_message"].tolist())
    df["embedding"] = list(comment_emb)

    # Embeddings de publicaciones
    print(f"   Generando embeddings de publicaciones ({len(df)} filas)...")
    post_emb = get_embeddings_batch(df["clean_post_message"].tolist())
    df["post_embedding"] = list(post_emb)

    # Guardar: append si existe, crear si es primera ejecución
    _SILVER_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if _SILVER_PARQUET.exists():
        df_existente = pd.read_parquet(_SILVER_PARQUET)
        df_total     = pd.concat([df_existente, df], ignore_index=True)
    else:
        df_total = df

    df_total.to_parquet(_SILVER_PARQUET, index=False)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos con embeddings agregados.")
    print(f"  Total acumulado en Silver: {len(df_total)} registros.")
    print(f"  Archivo: {_SILVER_PARQUET}")
    print(f"{'─'*45}")

    return df
