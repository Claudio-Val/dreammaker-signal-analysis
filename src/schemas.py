"""
src/schemas.py
===============
Definición explícita de schemas de BigQuery para las tablas Silver y
Gold. Se definen a mano (en vez de dejar que BigQuery infiera tipos al
cargar) para tener control total sobre el tipo de cada columna, evitar
sorpresas de autodetección en cargas futuras, y para que las tablas
queden documentadas en un solo lugar del código.
"""

from google.cloud import bigquery


_LABELS = [
    "interes", "elogio_producto", "critica_producto", "fanatismo_emocional",
    "conflictivo", "otro", "solicitud", "cuidado", "sarcasmo",
]

_REGLAS_NEGOCIO = [
    "solicitud_real", "interes_real", "elogio_real", "critica_genuina",
    "polarizacion_politica", "fanatismo", "fanatismo_no_comercial",
]


def _pred_schema() -> list[bigquery.SchemaField]:
    """Columnas pred_<label> (0/1) — Capa 1 (8 etiquetas) + Capa 2 (sarcasmo)."""
    return [bigquery.SchemaField(f"pred_{l}", "INTEGER") for l in _LABELS]


def _proba_schema() -> list[bigquery.SchemaField]:
    """Columnas proba_<label> (float) — Capa 1 (8 etiquetas) + Capa 2 (sarcasmo)."""
    return [bigquery.SchemaField(f"proba_{l}", "FLOAT64") for l in _LABELS]


def _reglas_negocio_schema() -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(c, "INTEGER") for c in _REGLAS_NEGOCIO]


# ── Silver ─────────────────────────────────────────────────────────────────────
SCHEMA_SILVER_CLEAN = [
    bigquery.SchemaField("post_id",              "STRING"),
    bigquery.SchemaField("post_message",          "STRING"),
    bigquery.SchemaField("post_created",          "TIMESTAMP"),
    bigquery.SchemaField("post_reactions",        "INTEGER"),
    bigquery.SchemaField("post_image_url",        "STRING"),
    bigquery.SchemaField("comment_id",            "STRING", mode="REQUIRED"),
    bigquery.SchemaField("comment_message",       "STRING"),
    bigquery.SchemaField("comment_created",       "TIMESTAMP"),
    bigquery.SchemaField("comment_reactions",     "INTEGER"),
    bigquery.SchemaField("clean_comment_message", "STRING"),
    bigquery.SchemaField("clean_post_message",    "STRING"),
]

SCHEMA_SILVER_EMBEDDINGS = SCHEMA_SILVER_CLEAN + [
    bigquery.SchemaField("embedding",      "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("post_embedding", "FLOAT64", mode="REPEATED"),
]

# ── Gold ───────────────────────────────────────────────────────────────────────
SCHEMA_GOLD_CLASSIFIED = SCHEMA_SILVER_EMBEDDINGS + _pred_schema()

SCHEMA_GOLD_PROBABILITIES = [
    bigquery.SchemaField("comment_id", "STRING", mode="REQUIRED"),
] + _proba_schema()

SCHEMA_GOLD_ENRICHED = SCHEMA_GOLD_CLASSIFIED + _reglas_negocio_schema()

SCHEMA_GOLD_POST_METRICS = [
    bigquery.SchemaField("post_id",                      "STRING", mode="REQUIRED"),
    bigquery.SchemaField("clean_post_message",            "STRING"),
    bigquery.SchemaField("volumen_total",                 "INTEGER"),
    bigquery.SchemaField("volumen_interes",               "INTEGER"),
    bigquery.SchemaField("volumen_solicitudes",           "INTEGER"),
    bigquery.SchemaField("volumen_elogios",                "INTEGER"),
    bigquery.SchemaField("volumen_criticas",               "INTEGER"),
    bigquery.SchemaField("volumen_polarizacion",           "INTEGER"),
    bigquery.SchemaField("volumen_fanatismo",               "INTEGER"),
    bigquery.SchemaField("volumen_fanatismo_no_comercial", "INTEGER"),
    bigquery.SchemaField("volumen_senal",                   "INTEGER"),
    bigquery.SchemaField("volumen_ruido",                   "INTEGER"),
    bigquery.SchemaField("prop_interes",                    "FLOAT64"),
    bigquery.SchemaField("prop_solicitudes",                "FLOAT64"),
    bigquery.SchemaField("prop_elogios",                    "FLOAT64"),
    bigquery.SchemaField("prop_criticas",                   "FLOAT64"),
    bigquery.SchemaField("prop_polarizacion",                "FLOAT64"),
    bigquery.SchemaField("prop_fanatismo",                   "FLOAT64"),
    bigquery.SchemaField("prop_fanatismo_no_comercial",      "FLOAT64"),
    bigquery.SchemaField("prop_senal_comercial",              "FLOAT64"),
    bigquery.SchemaField("prop_ruido_social",                 "FLOAT64"),
    bigquery.SchemaField("post_producto_materializado",       "INTEGER"),
    bigquery.SchemaField("post_carabineros",                  "INTEGER"),
    bigquery.SchemaField("post_ejercito",                     "INTEGER"),
    bigquery.SchemaField("post_fach",                         "INTEGER"),
    bigquery.SchemaField("post_armada",                       "INTEGER"),
    bigquery.SchemaField("post_pdi",                          "INTEGER"),
    bigquery.SchemaField("post_guerra_pacifico",              "INTEGER"),
    bigquery.SchemaField("post_personajes_historicos",        "INTEGER"),
]
