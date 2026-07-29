"""
src/aggregation.py
==================
Agrega predicciones a nivel comentario hacia métricas por publicación
y enriquece cada post con variables temáticas y de producto.

Compatibilidad
--------------
- Modo local   : generar_metricas_post(df) → pd.DataFrame
- Modo Airflow : task_agregacion(**context) → str (path Gold post metrics)
"""

import re
from pathlib import Path

import pandas as pd


# ── Constantes ────────────────────────────────────────────────────────────────
_GOLD_PATH         = Path("data/gold")
_GOLD_CLASSIFIED   = _GOLD_PATH / "facebook_comments_classified.parquet"
_GOLD_ENRICHED     = _GOLD_PATH / "facebook_comments_gold_enriched.parquet"
_GOLD_POST_METRICS = _GOLD_PATH / "facebook_post_metrics.parquet"

_KEYWORDS_PRODUCTO = [
    "cm", "dimensiones", "escala", "impreso",
    "fabricados", "material", "pintura", "pintado",
]

_CLASES_TEMATICAS = {
    "post_carabineros":          ["carabinero", "carabineros"],
    "post_ejercito":             ["ejercito"],
    "post_fach":                 ["fach", "fuerza aerea"],
    "post_armada":               ["armada", "marina", "infanteria de marina", "infante de marina"],
    "post_pdi":                  ["pdi", "policia de investigaciones"],
    "post_guerra_pacifico":      ["guerra del pacifico", "gdp"],
    "post_personajes_historicos":["allende", "pinochet", "bernardo o'higgins", "jose miguel carrera",
                                  "manuel rodriguez", "arturo prat", "gabriela mistral",
                                  "pablo neruda", "lautaro"],
}


# ── Señales de negocio ────────────────────────────────────────────────────────
def _agregar_senales_negocio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comentario_senal_comercial"] = (
        (df["interes_real"] == 1) | (df["solicitud_real"] == 1) |
        (df["elogio_real"]  == 1) | (df["critica_genuina"] == 1)
    ).astype(int)
    df["comentario_ruido_social"] = (
        (df["polarizacion_politica"] == 1) | (df["fanatismo_no_comercial"] == 1)
    ).astype(int)
    return df


# ── Agregación por post ───────────────────────────────────────────────────────
def _agregar_por_post(df: pd.DataFrame) -> pd.DataFrame:
    metricas = (
        df.groupby(["post_id", "clean_post_message"], as_index=False)
        .agg(
            volumen_total                  = ("comment_id",                 "count"),
            volumen_interes                = ("interes_real",               "sum"),
            volumen_solicitudes            = ("solicitud_real",             "sum"),
            volumen_elogios                = ("elogio_real",                "sum"),
            volumen_criticas               = ("critica_genuina",            "sum"),
            volumen_polarizacion           = ("polarizacion_politica",      "sum"),
            volumen_fanatismo              = ("fanatismo",                  "sum"),
            volumen_fanatismo_no_comercial = ("fanatismo_no_comercial",     "sum"),
            volumen_senal                  = ("comentario_senal_comercial", "sum"),
            volumen_ruido                  = ("comentario_ruido_social",    "sum"),
        )
    )

    alpha = metricas["volumen_total"].median()
    p = {
        "interes":                df["interes_real"].mean(),
        "solicitud":              df["solicitud_real"].mean(),
        "elogio":                 df["elogio_real"].mean(),
        "critica":                df["critica_genuina"].mean(),
        "polarizacion":           df["polarizacion_politica"].mean(),
        "fanatismo":              df["fanatismo"].mean(),
        "fanatismo_no_comercial": df["fanatismo_no_comercial"].mean(),
        "senal":                  df["comentario_senal_comercial"].mean(),
        "ruido":                  df["comentario_ruido_social"].mean(),
    }
    denom = metricas["volumen_total"] + alpha

    metricas["prop_interes"]                = (metricas["volumen_interes"]                 + alpha * p["interes"])               / denom
    metricas["prop_solicitudes"]            = (metricas["volumen_solicitudes"]              + alpha * p["solicitud"])             / denom
    metricas["prop_elogios"]                = (metricas["volumen_elogios"]                  + alpha * p["elogio"])                / denom
    metricas["prop_criticas"]               = (metricas["volumen_criticas"]                 + alpha * p["critica"])               / denom
    metricas["prop_polarizacion"]           = (metricas["volumen_polarizacion"]             + alpha * p["polarizacion"])          / denom
    metricas["prop_fanatismo"]              = (metricas["volumen_fanatismo"]                + alpha * p["fanatismo"])             / denom
    metricas["prop_fanatismo_no_comercial"] = (metricas["volumen_fanatismo_no_comercial"]   + alpha * p["fanatismo_no_comercial"]) / denom
    metricas["prop_senal_comercial"]        = (metricas["volumen_senal"]                    + alpha * p["senal"])                 / denom
    metricas["prop_ruido_social"]           = (metricas["volumen_ruido"]                    + alpha * p["ruido"])                 / denom

    return metricas


# ── Clasificación temática ────────────────────────────────────────────────────
def _clasificar_tematica(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    texto = df["clean_post_message"].fillna("").str.lower()

    patron_producto = r"\b(?:{})\b".format("|".join(map(re.escape, _KEYWORDS_PRODUCTO)))
    df["post_producto_materializado"] = texto.str.contains(patron_producto, case=False, regex=True).astype(int)

    for columna, palabras in _CLASES_TEMATICAS.items():
        patron     = "|".join(rf"\b{re.escape(p.lower())}\b" for p in palabras)
        df[columna] = texto.str.contains(patron, regex=True).astype(int)

    return df


def _agregar_y_persistir() -> str:
    """
    Lógica central de agregación. Siempre opera sobre el Gold classified
    completo para que el prior bayesiano sea consistente.
    Retorna el path del Gold post metrics.
    """
    from src.classification import _aplicar_reglas_negocio

    if not _GOLD_CLASSIFIED.exists():
        print("   Sin datos para agregar.")
        return str(_GOLD_POST_METRICS)

    df_classified = pd.read_parquet(_GOLD_CLASSIFIED)
    df            = _aplicar_reglas_negocio(df_classified)
    df.to_parquet(_GOLD_ENRICHED, index=False)

    df            = _agregar_senales_negocio(df)
    metricas_post = _agregar_por_post(df)
    metricas_post = _clasificar_tematica(metricas_post)

    _GOLD_PATH.mkdir(parents=True, exist_ok=True)
    metricas_post.to_parquet(_GOLD_POST_METRICS, index=False)

    print(f"{'─'*45}")
    print(f"  Métricas calculadas para {len(metricas_post)} publicaciones.")
    print(f"  Archivo: {_GOLD_POST_METRICS}")
    print(f"{'─'*45}")

    return str(_GOLD_POST_METRICS)


# ── Modo local ────────────────────────────────────────────────────────────────
def generar_metricas_post(df: pd.DataFrame) -> pd.DataFrame:
    """Genera métricas por post y retorna DataFrame. Uso: pipeline.py local."""
    if not _GOLD_CLASSIFIED.exists():
        print("   Sin datos para agregar.")
        return pd.DataFrame()
    if df.empty:
        print("   Sin comentarios nuevos. Recalculando métricas sobre Gold completo...")
    _agregar_y_persistir()
    return pd.read_parquet(_GOLD_POST_METRICS)


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_agregacion(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Agrega Gold classified completo a métricas por post.
    Retorna el path del Gold post metrics para XCom.
    """
    return _agregar_y_persistir()
