import pandas as pd
import re
import unicodedata
from pathlib import Path
from datetime import datetime

def clean_text(text, inchar=" "):
    """
    Realiza una limpieza básica de texto eliminando signos, tildes,
    repeticiones de símbolos y unifica espacios.
    """
    signos = '()[]{}<>^*#@=+#¡!'
    for k in signos:
        text = text.replace(k, " ")

    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'\?{2,}', '?', text)
    text = text.replace(' ', inchar)
    text = re.sub('%s+' % inchar, inchar, text)
    return text


def clean_text_column(df, source_col, target_col):
    """
    Aplica la limpieza profunda a una columna de texto (como comentarios o publicaciones)
    y guarda el resultado en una nueva columna.
    """
    df[target_col] = df[source_col].astype(str).str.replace(r"\n", " ", regex=True)
    df[target_col] = df[target_col].apply(lambda x: clean_text(x, " "))
    df[target_col] = df[target_col].str.strip()
    return df


def remove_empty_comments(df, column_name):
    """
    Elimina filas donde el campo especificado (por ejemplo 'comment_message')
    esté vacío o contenga solo espacios.
    """
    print(f"Comentarios antes de eliminar filas vacías: {len(df)}")
    original_len = len(df)
    df = df[df[column_name].astype(str).str.strip().astype(bool)]
    print(f"Comentarios eliminados: {original_len - len(df)}")
    print(f"Quedaron {len(df)} comentarios luego de eliminar filas vacías")
    return df.reset_index(drop=True)


def preprocess_dataframe(df, datalake_path):
    """
    Ejecuta todo el pipeline de preprocesamiento:
    - Limpieza de textos (post y comentario)
    - Eliminación de filas vacías
    - Guarda los resultados en dreammaker/output/processed
    """
    df = clean_text_column(df, "comment_message", "clean_comment_message")
    df["clean_comment_message"] = df["clean_comment_message"].astype(str)
    df = clean_text_column(df, "post_message", "clean_post_message")
    df = remove_empty_comments(df, "clean_comment_message")
    df = remove_empty_comments(df, "clean_post_message")

    # Guardado en Databricks
    output_dir = Path(datalake_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fecha_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"datos_procesados_{fecha_str}.parquet"
    df.to_parquet(file_path, index=False)
    print(f"Datos preprocesados guardados en: {file_path}")

    return df

