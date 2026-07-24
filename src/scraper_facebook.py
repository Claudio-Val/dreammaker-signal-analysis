"""
src/scraper_facebook.py
=======================
Extrae posts y comentarios desde la Facebook Graph API y los guarda
en la capa Bronze (data/bronze/facebook_comments_raw.csv).

Comportamiento incremental
--------------------------
- Primera ejecución : guarda todos los comentarios extraídos.
- Ejecuciones siguientes: carga el CSV existente, detecta comment_id
  ya vistos y hace append únicamente con los comentarios nuevos.
"""

import os

import pandas as pd
import requests


# ── Valores que CSV/pandas interpreta como NaN ────────────────────────────────
_NA_VALUES = {"NA", "N/A", "NULL", "null", "None", "nan", "NaN", ""}

_BRONZE_PATH = os.path.join("data", "bronze")
_BRONZE_FILE = os.path.join(_BRONZE_PATH, "facebook_comments_raw.csv")


def _clean(val):
    """Devuelve None si val es un string que pandas leería como NaN."""
    if val is None:
        return None
    return None if str(val).strip() in _NA_VALUES else val


def _cargar_ids_existentes() -> set:
    """
    Carga los comment_id ya presentes en Bronze.
    Retorna un set vacío si el archivo no existe (primera ejecución).
    """
    if not os.path.exists(_BRONZE_FILE):
        print("Bronze no encontrado. Se realizará extracción completa.")
        return set()

    df_existente = pd.read_csv(_BRONZE_FILE, usecols=["comment_id"], dtype=str)
    ids = set(df_existente["comment_id"].dropna().unique())
    print(f"Bronze existente cargado: {len(ids)} comment_id registrados.")
    return ids


def _guardar_bronze(df_nuevos: pd.DataFrame) -> None:
    """
    Hace append al CSV Bronze si existe, o lo crea si es la primera ejecución.
    """
    os.makedirs(_BRONZE_PATH, exist_ok=True)

    if os.path.exists(_BRONZE_FILE):
        df_nuevos.to_csv(_BRONZE_FILE, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df_nuevos.to_csv(_BRONZE_FILE, index=False, encoding="utf-8")


# ── Función pública ───────────────────────────────────────────────────────────
def obtener_datos_facebook(access_token: str, page_id: str) -> pd.DataFrame:
    """
    Extrae posts y comentarios desde la Facebook Graph API.

    Comportamiento incremental: solo retorna y persiste comentarios
    cuyo comment_id no esté ya en Bronze. En la primera ejecución
    persiste y retorna todo.

    Parámetros
    ----------
    access_token : str
        Token de acceso (variable de entorno FACEBOOK_TOKEN).
    page_id : str
        ID de la página de Facebook (variable de entorno PAGE_ID).

    Retorna
    -------
    pd.DataFrame
        Solo los registros nuevos. DataFrame vacío si no hay novedades.
    """
    if not access_token:
        raise ValueError("Token de acceso no encontrado. Revisa FACEBOOK_TOKEN en tu .env.")

    ids_existentes = _cargar_ids_existentes()

    print("\nExtrayendo datos desde Facebook...")

    # ── Posts con paginación segura ───────────────────────────────────────────
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
                f"Facebook API Error {data['error']['code']}: "
                f"{data['error']['message']}"
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
                    print(f"⚠️  URL de paginación repetida en post {post_id}, deteniendo loop.")
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
        post_id      = post.get("id")
        post_message = _clean(post.get("message"))
        post_created = post.get("created_time")
        post_reactions = (
            post.get("reactions", {}).get("summary", {}).get("total_count", 0)
        )
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
                        comment.get("reactions", {})
                               .get("summary", {})
                               .get("total_count", 0)
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

    # ── Filtrar solo comentarios nuevos ───────────────────────────────────────
    if ids_existentes:
        df_nuevos = df_api[
            ~df_api["comment_id"].astype(str).isin(ids_existentes)
        ].copy()
    else:
        df_nuevos = df_api.copy()

    n_nuevos = len(df_nuevos)

    if n_nuevos == 0:
        print("\n✔ Sin comentarios nuevos. Bronze ya está al día.")
        return df_nuevos

    # ── Guardar en Bronze ─────────────────────────────────────────────────────
    _guardar_bronze(df_nuevos)

    print(f"\n{'─'*45}")
    print(f"  {n_nuevos} comentarios nuevos agregados a Bronze.")
    print(f"  Total Bronze acumulado: {len(ids_existentes) + n_nuevos} registros.")
    print(f"  Archivo: {_BRONZE_FILE}")
    print(f"{'─'*45}")

    return df_nuevos
