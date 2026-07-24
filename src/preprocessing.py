"""
src/preprocessing.py
====================
Limpieza de texto y filtrado de comentarios vacíos.

Comportamiento incremental
--------------------------
- Primera ejecución : crea el Silver CSV desde cero.
- Ejecuciones siguientes: hace append al Silver CSV existente
  con los comentarios nuevos ya filtrados y limpios.
"""

import os
import re
import unicodedata

import pandas as pd


_SILVER_PATH = os.path.join("data", "silver")
_SILVER_FILE = os.path.join(_SILVER_PATH, "facebook_comments_clean.csv")


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
    """
    Elimina filas donde comment_message sea NaN o string vacío.
    """
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


# ── Guardado incremental ──────────────────────────────────────────────────────
def _guardar_silver(df: pd.DataFrame) -> None:
    """
    Hace append al Silver CSV si existe, o lo crea si es la primera ejecución.
    """
    os.makedirs(_SILVER_PATH, exist_ok=True)

    if os.path.exists(_SILVER_FILE):
        df.to_csv(_SILVER_FILE, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(_SILVER_FILE, index=False, encoding="utf-8")


# ── Función pública ───────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline de preprocesamiento sobre comentarios nuevos:
      1. Filtra filas con comment_message vacío
      2. Aplica clean_text → clean_comment_message y clean_post_message
      3. Hace append al Silver CSV (o lo crea en primera ejecución)

    Parámetros
    ----------
    df : pd.DataFrame
        Delta de comentarios nuevos proveniente de obtener_datos_facebook.
        Si el df viene vacío, retorna inmediatamente sin hacer nada.

    Retorna
    -------
    pd.DataFrame
        Comentarios nuevos limpios. DataFrame vacío si no había novedades.
    """
    if df.empty:
        print("   Sin datos nuevos para preprocesar.")
        return df

    n_entrada = len(df)

    # Filtrar vacíos
    df = filtrar_comentarios_vacios(df)

    # Limpiar texto
    _clean_col = lambda x: clean_text(str(x), " ")
    df["clean_comment_message"] = df["comment_message"].apply(_clean_col)
    df["clean_post_message"]    = df["post_message"].apply(_clean_col)

    # Guardar
    _guardar_silver(df)

    es_primera = not os.path.exists(_SILVER_FILE) or n_entrada == len(df)

    print(f"{'─'*45}")
    print(f"  {len(df)} comentarios nuevos guardados en Silver.")
    print(f"  Archivo: {_SILVER_FILE}")
    print(f"{'─'*45}")

    return df
