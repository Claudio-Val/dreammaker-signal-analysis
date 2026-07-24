"""
src/aggregation.py
==================
Agrega predicciones a nivel comentario hacia métricas por publicación
y enriquece cada post con variables temáticas y de producto.

Este módulo opera sobre el Gold enriquecido (salida de classification.py)
y produce un dataset analítico final a nivel post, listo para consumo
en herramientas de visualización como PowerBI.

Etapas internas
---------------
1. Señales comerciales y de ruido   : variables binarias derivadas de labels
2. Métricas por post                : agregación con suavizado bayesiano
3. Clasificación temática de posts  : instituciones, personajes, producto físico

Suavizado bayesiano
-------------------
Las proporciones se calculan con suavizado para evitar estimaciones
extremas en posts con poco volumen de comentarios:

    prop = (volumen_label + α * p_global) / (volumen_total + α)

donde α = mediana del volumen de comentarios por post (prior adaptativo).

Salida
------
    data/gold/facebook_post_metrics.parquet
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
    "post_carabineros": [
        "carabinero", "carabineros",
    ],
    "post_ejercito": [
        "ejercito",
    ],
    "post_fach": [
        "fach", "fuerza aerea",
    ],
    "post_armada": [
        "armada", "marina", "infanteria de marina", "infante de marina",
    ],
    "post_pdi": [
        "pdi", "policia de investigaciones",
    ],
    "post_guerra_pacifico": [
        "guerra del pacifico", "gdp",
    ],
    "post_personajes_historicos": [
        "allende", "pinochet", "bernardo o'higgins", "jose miguel carrera",
        "manuel rodriguez", "arturo prat", "gabriela mistral",
        "pablo neruda", "lautaro",
    ],
}


# ── Señales de negocio ────────────────────────────────────────────────────────
def _agregar_senales_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva dos señales agregadas a partir de los labels de negocio:

    comentario_senal_comercial : interes_real | solicitud_real |
                                 elogio_real  | critica_genuina
    comentario_ruido_social    : polarizacion_politica | fanatismo_no_comercial
    """
    df = df.copy()

    df["comentario_senal_comercial"] = (
        (df["interes_real"]         == 1) |
        (df["solicitud_real"]       == 1) |
        (df["elogio_real"]          == 1) |
        (df["critica_genuina"]      == 1)
    ).astype(int)

    df["comentario_ruido_social"] = (
        (df["polarizacion_politica"]  == 1) |
        (df["fanatismo_no_comercial"] == 1)
    ).astype(int)

    return df


# ── Agregación por post ───────────────────────────────────────────────────────
def _agregar_por_post(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega comentarios a nivel publicación calculando volúmenes y
    proporciones con suavizado bayesiano (prior adaptativo α = mediana
    del volumen de comentarios por post).
    """
    metricas = (
        df
        .groupby(["post_id", "clean_post_message"], as_index=False)
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

    # Prior adaptativo
    alpha = metricas["volumen_total"].median()

    # Proporciones globales del dataset (estimadores del prior)
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
    """
    Enriquece el dataset de métricas con variables binarias asociadas
    al contenido textual de la publicación.

    Variables generadas
    -------------------
    post_producto_materializado : menciona características físicas del producto
    post_carabineros            : menciona Carabineros de Chile
    post_ejercito               : menciona Ejército de Chile
    post_fach                   : menciona Fuerza Aérea
    post_armada                 : menciona Armada / Infantería de Marina
    post_pdi                    : menciona PDI
    post_guerra_pacifico        : menciona Guerra del Pacífico
    post_personajes_historicos  : menciona personajes históricos chilenos

    Las categorías no son mutuamente excluyentes.
    """
    df = df.copy()
    texto = df["clean_post_message"].fillna("").str.lower()

    patron_producto = r"\b(?:{})\b".format(
        "|".join(map(re.escape, _KEYWORDS_PRODUCTO))
    )
    df["post_producto_materializado"] = (
        texto.str.contains(patron_producto, case=False, regex=True).astype(int)
    )

    for columna, palabras in _CLASES_TEMATICAS.items():
        patron = "|".join(rf"\b{re.escape(p.lower())}\b" for p in palabras)
        df[columna] = texto.str.contains(patron, regex=True).astype(int)

    return df


# ── Función pública ───────────────────────────────────────────────────────────
def generar_metricas_post(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el dataset analítico final a nivel publicación.

    Siempre opera sobre el Gold enriched completo acumulado para que
    el prior bayesiano sea consistente con todos los datos disponibles.
    Re-aplica las reglas de negocio desde classification.py en cada
    ejecución para reflejar siempre la versión actual de las reglas.

    Parámetros
    ----------
    df : pd.DataFrame
        Gold enriquecido del delta (salida de predecir_labels).
        Si viene vacío, recarga desde classified en disco.

    Retorna
    -------
    pd.DataFrame
        Dataset analítico a nivel post guardado en:
        data/gold/facebook_post_metrics.parquet
    """
    from src.classification import _aplicar_reglas_negocio

    if not _GOLD_CLASSIFIED.exists():
        print("   Sin datos para agregar.")
        return pd.DataFrame()

    # Siempre partir desde classified completo y re-aplicar reglas
    # para garantizar que las columnas de negocio estén actualizadas
    if df.empty:
        print("   Sin comentarios nuevos. Recalculando métricas sobre Gold completo...")

    df_classified = pd.read_parquet(_GOLD_CLASSIFIED)
    df = _aplicar_reglas_negocio(df_classified)
    df.to_parquet(_GOLD_ENRICHED, index=False)

    # Pipeline de agregación
    df            = _agregar_senales_negocio(df)
    metricas_post = _agregar_por_post(df)
    metricas_post = _clasificar_tematica(metricas_post)

    # Guardar
    _GOLD_PATH.mkdir(parents=True, exist_ok=True)
    metricas_post.to_parquet(_GOLD_POST_METRICS, index=False)

    print(f"{'─'*45}")
    print(f"  Métricas calculadas para {len(metricas_post)} publicaciones.")
    print(f"  Archivo: {_GOLD_POST_METRICS}")
    print(f"{'─'*45}")

    return metricas_post