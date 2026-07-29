"""
src/preprocessing.py
====================
Limpieza de texto y filtrado de comentarios vacíos.

Comportamiento incremental
--------------------------
- Primera ejecución : crea el Silver CSV desde cero.
- Ejecuciones siguientes: hace append al Silver CSV existente
  con los comentarios nuevos ya filtrados y limpios.

Compatibilidad
--------------
- Modo local   : preprocess_dataframe(df) → pd.DataFrame
- Modo Airflow : task_preprocessing(**context) → str (path Silver CSV)
"""

import os
import re
import unicodedata

import pandas as pd


_BRONZE_FILE = os.path.join("data", "bronze", "facebook_comments_raw.csv")
_SILVER_PATH = os.path.join("data", "silver")
_SILVER_FILE = os.path.join(_SILVER_PATH, "facebook_comments_clean.csv")


# ── Limpieza de texto ─────────────────────────────────────────────────────────
def clean_text(text: str | None, inchar: str = " ") -> str | None:
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


def filtrar_comentarios_vacios(df: pd.DataFrame) -> pd.DataFrame:
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


def _guardar_silver(df: pd.DataFrame) -> None:
    os.makedirs(_SILVER_PATH, exist_ok=True)
    if os.path.exists(_SILVER_FILE):
        df.to_csv(_SILVER_FILE, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(_SILVER_FILE, index=False, encoding="utf-8")


def _limpiar_y_persistir(df: pd.DataFrame) -> str:
    """
    Lógica central de preprocesamiento. Retorna el path del Silver CSV.
    Usada tanto en modo local como en modo Airflow.
    """
    if df.empty:
        print("   Sin datos nuevos para preprocesar.")
        return _SILVER_FILE

    df = filtrar_comentarios_vacios(df)

    _clean_col = lambda x: clean_text(str(x), " ")
    df["clean_comment_message"] = df["comment_message"].apply(_clean_col)
    df["clean_post_message"]    = df["post_message"].apply(_clean_col)

    _guardar_silver(df)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos guardados en Silver.")
    print(f"  Archivo: {_SILVER_FILE}")
    print(f"{'─'*45}")

    return _SILVER_FILE


# ── Modo local ────────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el delta recibido y retorna DataFrame limpio. Uso: pipeline.py local."""
    if df.empty:
        print("   Sin datos nuevos para preprocesar.")
        return df

    df_orig = df.copy()
    df_orig = filtrar_comentarios_vacios(df_orig)

    _clean_col = lambda x: clean_text(str(x), " ")
    df_orig["clean_comment_message"] = df_orig["comment_message"].apply(_clean_col)
    df_orig["clean_post_message"]    = df_orig["post_message"].apply(_clean_col)

    _guardar_silver(df_orig)

    print(f"{'─'*45}")
    print(f"  {len(df_orig)} comentarios nuevos guardados en Silver.")
    print(f"  Archivo: {_SILVER_FILE}")
    print(f"{'─'*45}")

    return df_orig


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_preprocessing(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee desde Bronze, procesa el delta y escribe en Silver CSV.
    Retorna el path del Silver CSV para XCom.
    """
    # Leer IDs ya procesados en Silver para extraer solo el delta
    ids_silver: set = set()
    if os.path.exists(_SILVER_FILE):
        df_silver = pd.read_csv(_SILVER_FILE, usecols=["comment_id"], dtype=str)
        ids_silver = set(df_silver["comment_id"].dropna().unique())

    df_bronze = pd.read_csv(_BRONZE_FILE)

    # Delta: solo los que no están en Silver
    df_delta = df_bronze[
        ~df_bronze["comment_id"].astype(str).isin(ids_silver)
    ].copy()

    print(f"Delta para preprocesar: {len(df_delta)} comentarios.")
    return _limpiar_y_persistir(df_delta)
