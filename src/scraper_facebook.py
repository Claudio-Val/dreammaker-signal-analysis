import os
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

def obtener_datos_facebook(facebook_token: str, datalake_path: str):
    """
    Extrae datos de Facebook usando la API Graph y guarda un archivo CSV en la carpeta `datalake_path`.
    
    Parámetros
    ----------
    facebook_token : str
        Token de acceso de la API de Facebook.
    datalake_path : str
        Ruta absoluta donde se guardarán los datos (ej. "/dbfs/FileStore/dreammaker/output/raw").
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con todas las publicaciones y comentarios extraídos.
    """

    if not facebook_token:
        raise ValueError("Token de acceso no encontrado. Define FACEBOOK_TOKEN como variable de entorno.")

    print("Extrayendo datos desde Facebook...")
    PAGE_ID = "154787275135903"
    base_url = f"https://graph.facebook.com/v18.0/{PAGE_ID}/posts"

    all_posts = []

    # Paginación de publicaciones
    while base_url:
        response = requests.get(base_url, params={
            "fields": "id,message,created_time,attachments{media_type,media,url},"
                      "reactions.summary(true),comments.limit(100){message,created_time,id,reactions.summary(true)}",
            "access_token": facebook_token
        })
        data = response.json()
        all_posts.extend(data.get("data", []))
        base_url = data.get("paging", {}).get("next", None)

    # Paginación de comentarios
    all_comments_by_post = {}
    for post in all_posts:
        post_id = post["id"]
        comments = []

        if "comments" in post:
            comments.extend(post["comments"]["data"])
            next_comments_url = post["comments"].get("paging", {}).get("next", None)

            while next_comments_url:
                resp = requests.get(next_comments_url)
                comment_data = resp.json()
                comments.extend(comment_data.get("data", []))
                next_comments_url = comment_data.get("paging", {}).get("next", None)

        all_comments_by_post[post_id] = comments

    # Construcción del DataFrame
    rows = []
    for post in all_posts:
        post_id = post.get('id')
        post_message = post.get('message', '')
        post_created = post.get('created_time')
        post_reactions = post.get('reactions', {}).get('summary', {}).get('total_count', 0)

        post_image_url = ''
        attachments = post.get('attachments', {}).get('data', [])
        if attachments:
            post_image_url = attachments[0].get('url', '')

        comments = all_comments_by_post.get(post_id, [])
        if comments:
            for comment in comments:
                comment_id = comment.get('id')
                comment_message = comment.get('message', '')
                comment_created = comment.get('created_time')
                comment_reactions = comment.get('reactions', {}).get('summary', {}).get('total_count', 0)

                rows.append({
                    'post_id': post_id,
                    'post_message': post_message,
                    'post_created': post_created,
                    'post_reactions': post_reactions,
                    'post_image_url': post_image_url,
                    'comment_id': comment_id,
                    'comment_message': comment_message,
                    'comment_created': comment_created,
                    'comment_reactions': comment_reactions
                })
        else:
            rows.append({
                'post_id': post_id,
                'post_message': post_message,
                'post_created': post_created,
                'post_reactions': post_reactions,
                'post_image_url': post_image_url,
                'comment_id': '',
                'comment_message': '',
                'comment_created': '',
                'comment_reactions': 0
            })

    df = pd.DataFrame(rows)
    print(f"📊 {len(df)} registros extraídos desde la API de Facebook.")

    # === Guardado en el Databricks  ===
    output_dir = Path(datalake_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fecha_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"facebook_data_{fecha_str}.csv"

    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Datos de Facebook guardados en: {file_path}")

    return df

