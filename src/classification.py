"""
src/classification.py
=====================
Pipeline jerárquico de clasificación en dos capas con reglas de negocio.

    Capa 1 — Regresión Logística OvR (multilabel)
              Predice: interes, elogio_producto, critica_producto,
                       fanatismo_emocional, conflictivo, otro,
                       solicitud, cuidado
              Modelos: models/logistic_ovr_model.pkl
                       models/logistic_ovr_thresholds.pkl

    Capa 2 — XGBoost (binario)
              Predice: sarcasmo
              Input  : embeddings + predicciones Capa 1
              Modelos: models/xgb_sarcasm_model.pkl
                       models/xgb_sarcasm_threshold.pkl

Comportamiento incremental
--------------------------
- Clasifica únicamente el delta de comentarios nuevos recibido.
- Hace append al Gold Parquet existente (o lo crea en primera ejecución).
- El archivo de probabilidades se trata igual.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

_MODEL_PATHS = {
    "capa1_model":      Path("models/logistic_ovr_model.pkl"),
    "capa1_thresholds": Path("models/logistic_ovr_thresholds.pkl"),
    "capa2_model":      Path("models/xgb_sarcasm_model.pkl"),
    "capa2_threshold":  Path("models/xgb_sarcasm_threshold.pkl"),
}

_GOLD_PATH       = Path("data/gold")
_GOLD_CLASSIFIED = _GOLD_PATH / "facebook_comments_classified.parquet"
_GOLD_PROBS      = _GOLD_PATH / "facebook_comments_probabilities.parquet"
_GOLD_ENRICHED   = _GOLD_PATH / "facebook_comments_gold_enriched.parquet"


# ── Carga de modelos ──────────────────────────────────────────────────────────
def _cargar_modelos() -> tuple:
    """Carga los cuatro artefactos del pipeline jerárquico."""
    for nombre, ruta in _MODEL_PATHS.items():
        if not ruta.exists():
            raise RuntimeError(
                f"Modelo no encontrado: {ruta}\n"
                f"Asegúrate de colocar '{ruta.name}' dentro de models/."
            )
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
    fanatismo             : fanatismo_emocional=1 AND sarcasmo=0
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


# ── Guardado incremental ──────────────────────────────────────────────────────
def _guardar_gold(df_nuevo: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append al Parquet Gold si existe, crear si es primera ejecución."""
    _GOLD_PATH.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_total = pd.concat([pd.read_parquet(path), df_nuevo], ignore_index=True)
    else:
        df_total = df_nuevo

    df_total.to_parquet(path, index=False)
    return df_total


# ── Función pública ───────────────────────────────────────────────────────────
def predecir_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline jerárquico completo sobre el delta de comentarios nuevos.

    Parámetros
    ----------
    df : pd.DataFrame
        Comentarios nuevos con columna 'embedding' (768-dim).
        Si viene vacío, retorna inmediatamente sin hacer nada.

    Retorna
    -------
    df_resultado     : pd.DataFrame  predicciones binarias (solo nuevos)
    df_probas        : pd.DataFrame  probabilidades (solo nuevos)
    df_gold_enriched : pd.DataFrame  reglas de negocio (solo nuevos)
    """
    if df.empty:
        print("   Sin datos nuevos para clasificar.")
        return df, pd.DataFrame(), pd.DataFrame()

    model_c1, thr_c1, model_c2, thr_c2 = _cargar_modelos()
    print("   ✔ Modelos cargados.")

    X_emb        = np.vstack(df["embedding"].values)
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

    # Append a Gold
    total_class    = _guardar_gold(df_resultado,    _GOLD_CLASSIFIED)
    total_probs    = _guardar_gold(df_probas,        _GOLD_PROBS)
    total_enriched = _guardar_gold(df_gold_enriched, _GOLD_ENRICHED)

    n_nuevos = len(df)
    print(f"{'─'*45}")
    print(f"  {n_nuevos} comentarios nuevos clasificados.")
    print(f"  Total acumulado en Gold: {len(total_enriched)} registros.")
    print(f"  Predicciones  : {_GOLD_CLASSIFIED}")
    print(f"  Probabilidades: {_GOLD_PROBS}")
    print(f"  Enriquecido   : {_GOLD_ENRICHED}")
    print(f"{'─'*45}")

    return df_resultado, df_probas, df_gold_enriched