"""
src/scraper_facebook.py
=======================
Extrae posts y comentarios desde la Facebook Graph API y los guarda en
la capa Bronze, en Cloud Storage (gs://<BUCKET_NAME>/bronze/...).

Comportamiento incremental
--------------------------
- Primera ejecución : sube todos los comentarios extraídos.
- Ejecuciones siguientes: descarga el CSV existente desde GCS, detecta
  comment_id ya vistos y sube el archivo completo (existente + nuevos) —
  ver nota sobre inmutabilidad de blobs en gcp_io.escribir_csv_gcs.

Compatibilidad
--------------
- Modo local   : obtener_datos_facebook(access_token, page_id) → pd.DataFrame (solo delta)
- Modo Airflow : task_scraping(**context) → str (blob path en GCS)

Credenciales de Facebook
-------------------------
- Modo local  : FACEBOOK_TOKEN / PAGE_ID leídos desde my_env.env (ver pipeline.py)
- Modo Airflow: FACEBOOK_TOKEN / PAGE_ID leídos desde Airflow Variables
"""

import pandas as pd
import requests

from src import config
from src.gcp_io import escribir_csv_gcs, ids_existentes_gcs


# ── Valores que CSV/pandas interpreta como NaN ────────────────────────────────
_NA_VALUES = {"NA", "N/A", "NULL", "null", "None", "nan", "NaN", ""}


def _clean(val):
    """Devuelve None si val es un string que pandas leería como NaN."""
    if val is None:
        return None
    return None if str(val).strip() in _NA_VALUES else val


def _extraer_y_persistir(access_token: str, page_id: str) -> tuple[str, pd.DataFrame]:
    """
    Lógica central de extracción. Retorna (blob_path, df_nuevos).
    Usada tanto en modo local como en modo Airflow.
    """
    ids_existentes = ids_existentes_gcs(config.GCS_BRONZE_BLOB, "comment_id")
    print(f"Bronze existente en GCS: {len(ids_existentes)} comment_id registrados.")

    print("\nExtrayendo datos desde Facebook...")

    url = f"https://graph.facebook.com/v18.0/{page_id}/posts"
    params = {
        "fields": (
            "id,message,created_time,"
            "attachments{media_type,media,url},"
            "reactions.summary(true),"
            "comments.limit(100){message,created_time,id,reactions.summary(true)}"
        ),
        "access_token": access_token,
    }

    all_posts = []
    seen_post_ids: set = set()
    visited_urls: set = set()

    while url:
        if url in visited_urls:
            print("⚠️  URL de paginación de posts repetida, deteniendo loop.")
            break
        visited_urls.add(url)

        response = requests.get(url, params=params)
        data = response.json()

        if "error" in data:
            raise Exception(
                f"Facebook API Error {data['error']['code']}: {data['error']['message']}"
            )

        for post in data.get("data", []):
            if post["id"] not in seen_post_ids:
                seen_post_ids.add(post["id"])
                all_posts.append(post)

        url = data.get("paging", {}).get("next")
        params = None

    print(f"Posts extraídos desde la API: {len(all_posts)}")

    # ── Comentarios con paginación segura y deduplicación ────────────────────
    all_comments_by_post: dict = {}

    for post in all_posts:
        post_id = post["id"]
        comments = []
        seen_ids: set = set()
        visited_comment_urls: set = set()

        if "comments" in post:
            for c in post["comments"]["data"]:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    comments.append(c)

            next_url = post["comments"].get("paging", {}).get("next")

            while next_url:
                if next_url in visited_comment_urls:
                    print(f"⚠️  URL repetida en post {post_id}, deteniendo loop.")
                    break
                visited_comment_urls.add(next_url)

                resp = requests.get(next_url)
                comment_data = resp.json()

                for c in comment_data.get("data", []):
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        comments.append(c)

                next_url = comment_data.get("paging", {}).get("next")

        all_comments_by_post[post_id] = comments

    total_api = sum(len(v) for v in all_comments_by_post.values())
    print(f"Comentarios únicos extraídos desde la API: {total_api}")

    # ── Construcción del DataFrame ────────────────────────────────────────────
    rows = []

    for post in all_posts:
        post_id        = post.get("id")
        post_message   = _clean(post.get("message"))
        post_created   = post.get("created_time")
        post_reactions = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
        attachments    = post.get("attachments", {}).get("data", [])
        post_image_url = attachments[0].get("url") if attachments else None
        comments       = all_comments_by_post.get(post_id, [])

        if comments:
            for comment in comments:
                rows.append({
                    "post_id":           post_id,
                    "post_message":      post_message,
                    "post_created":      post_created,
                    "post_reactions":    post_reactions,
                    "post_image_url":    post_image_url,
                    "comment_id":        comment.get("id"),
                    "comment_message":   _clean(comment.get("message")),
                    "comment_created":   comment.get("created_time"),
                    "comment_reactions": (
                        comment.get("reactions", {}).get("summary", {}).get("total_count", 0)
                    ),
                })
        else:
            rows.append({
                "post_id":           post_id,
                "post_message":      post_message,
                "post_created":      post_created,
                "post_reactions":    post_reactions,
                "post_image_url":    post_image_url,
                "comment_id":        None,
                "comment_message":   None,
                "comment_created":   None,
                "comment_reactions": 0,
            })

    df_api = pd.DataFrame(rows)

    # Filtrar sin texto (stickers, GIFs, imágenes)
    n_antes = len(df_api)
    mask = (
        df_api["comment_message"].notna()
        & df_api["comment_message"].astype(str).str.strip().ne("")
    )
    df_api = df_api[mask].copy()
    n_sin_texto = n_antes - len(df_api)
    if n_sin_texto > 0:
        print(f"   {n_sin_texto} comentarios sin texto descartados (stickers/GIFs).")

    # Filtrar solo nuevos
    if ids_existentes:
        df_nuevos = df_api[~df_api["comment_id"].astype(str).isin(ids_existentes)].copy()
    else:
        df_nuevos = df_api.copy()

    n_nuevos = len(df_nuevos)

    if n_nuevos == 0:
        print("\n✔ Sin comentarios nuevos con texto. Bronze ya está al día.")
    else:
        escribir_csv_gcs(df_nuevos, config.GCS_BRONZE_BLOB, append=True)
        print(f"\n{'─'*45}")
        print(f"  {n_nuevos} comentarios nuevos subidos a Bronze (GCS).")
        print(f"  Total Bronze acumulado: {len(ids_existentes) + n_nuevos} registros.")
        print(f"  gs://{config.BUCKET_NAME}/{config.GCS_BRONZE_BLOB}")
        print(f"{'─'*45}")

    return config.GCS_BRONZE_BLOB, df_nuevos


# ── Modo local ────────────────────────────────────────────────────────────────
def obtener_datos_facebook(access_token: str, page_id: str) -> pd.DataFrame:
    """
    Extrae datos y retorna SOLO el delta de comentarios nuevos. Uso: pipeline.py local.

    (Antes esta función retornaba el Bronze completo leído desde disco,
    lo que hacía que preprocessing/embedding/classification reprocesaran
    todo en cada corrida. Se corrige para que, igual que en modo Airflow,
    solo se propague el delta real hacia la siguiente etapa.)
    """
    if not access_token:
        raise ValueError("Token no encontrado. Revisa FACEBOOK_TOKEN en tu .env.")
    _, df_nuevos = _extraer_y_persistir(access_token, page_id)
    return df_nuevos


# ── Modo Airflow ──────────────────────────────────────────────────────────────
def task_scraping(**context) -> str:
    """
    Callable para PythonOperator de Airflow.
    Lee credenciales desde Airflow Variables.
    Retorna el blob path de Bronze en GCS para XCom.
    """
    from airflow.sdk import Variable
    access_token = Variable.get("FACEBOOK_TOKEN")
    page_id      = Variable.get("PAGE_ID")
    blob_path, _ = _extraer_y_persistir(access_token, page_id)
    return blob_path
