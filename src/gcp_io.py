"""
src/gcp_io.py
==============
Helpers de I/O contra Google Cloud (Cloud Storage + BigQuery), usados por
todos los módulos de src/ para reemplazar la lectura/escritura local de
Bronze/Silver/Gold. Ningún módulo debe instanciar sus propios clientes de
`google-cloud-storage` / `google-cloud-bigquery` — todos pasan por acá.

Autenticación: Application Default Credentials (ADC). Los clientes se
instancian sin pasar ninguna ruta de credenciales — la resuelve la propia
librería (ver docstring de config.py).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, storage

from src import config


# ── Clientes (uno por proceso, reutilizados) ──────────────────────────────────
_storage_client: storage.Client | None = None
_bq_client: bigquery.Client | None = None


def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=config.PROJECT_ID)
    return _storage_client


def get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=config.PROJECT_ID)
    return _bq_client


# ── Cloud Storage: Bronze (CSV crudo) ─────────────────────────────────────────
def leer_csv_gcs(blob_path: str) -> pd.DataFrame:
    """Lee un CSV desde el bucket configurado. DataFrame vacío si el blob no existe aún."""
    bucket = get_storage_client().bucket(config.BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return pd.DataFrame()
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def escribir_csv_gcs(df: pd.DataFrame, blob_path: str, append: bool = True) -> pd.DataFrame:
    """
    Escribe un DataFrame como CSV en GCS. Como los objetos de GCS son
    inmutables (no existe un "append" real a un blob existente), el
    patrón es: descargar lo existente, concatenar en memoria y volver a
    subir el archivo completo — el mismo patrón que ya usábamos con CSV
    local, solo que ahora la lectura/escritura es contra el bucket.

    Nota: para el volumen actual (miles de comentarios) esto es rápido y
    simple. Si el archivo creciera mucho, el siguiente paso natural sería
    particionar Bronze por fecha (un CSV por día) en vez de un único blob.

    Retorna el DataFrame total (existente + nuevo) ya persistido.
    """
    bucket = get_storage_client().bucket(config.BUCKET_NAME)
    blob = bucket.blob(blob_path)

    if append and blob.exists():
        existente = leer_csv_gcs(blob_path)
        df_total = pd.concat([existente, df], ignore_index=True)
    else:
        df_total = df

    blob.upload_from_string(df_total.to_csv(index=False), content_type="text/csv")
    return df_total


def ids_existentes_gcs(blob_path: str, columna: str) -> set:
    """IDs ya presentes en un CSV de GCS. Set vacío si el blob no existe todavía."""
    df = leer_csv_gcs(blob_path)
    if df.empty or columna not in df.columns:
        return set()
    return set(df[columna].dropna().astype(str).unique())


def descargar_blob_si_falta(blob_path: str, local_path: Path) -> None:
    """
    Descarga un blob desde GCS a disco local, solo si no existe localmente.
    Usado para traer los modelos entrenados (.pkl) desde GCS antes de
    cargarlos con joblib — ver classification.py.
    """
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = get_storage_client().bucket(config.BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    print(f"   ✔ Descargado desde GCS: gs://{config.BUCKET_NAME}/{blob_path} → {local_path}")


# ── BigQuery: creación idempotente de dataset/tabla ───────────────────────────
def crear_dataset_si_no_existe(dataset_id: str) -> None:
    """Crea el dataset (idempotente) si aún no existe. dataset_id sin proyecto, ej. 'dreammaker_silver'."""
    client = get_bq_client()
    ref = bigquery.DatasetReference(config.PROJECT_ID, dataset_id)
    try:
        client.get_dataset(ref)
    except NotFound:
        dataset = bigquery.Dataset(ref)
        dataset.location = config.BQ_LOCATION
        client.create_dataset(dataset)
        print(f"   ✔ Dataset creado: {dataset_id} ({config.BQ_LOCATION})")


def crear_tabla_si_no_existe(table_id: str, schema: list[bigquery.SchemaField]) -> None:
    """Crea la tabla (idempotente) con el schema explícito si aún no existe."""
    client = get_bq_client()
    try:
        client.get_table(table_id)
    except NotFound:
        tabla = bigquery.Table(table_id, schema=schema)
        client.create_table(tabla)
        print(f"   ✔ Tabla creada: {table_id}")


# ── BigQuery: lectura ──────────────────────────────────────────────────────────
def ids_existentes_bq(table_id: str, columna: str) -> set:
    """IDs ya presentes en una tabla de BigQuery. Set vacío si la tabla no existe todavía."""
    client = get_bq_client()
    try:
        client.get_table(table_id)
    except NotFound:
        return set()

    query = f"SELECT DISTINCT {columna} FROM `{table_id}`"
    resultado = client.query(query).result()
    return {str(row[columna]) for row in resultado}


def leer_tabla_bq(table_id: str) -> pd.DataFrame:
    """Lee una tabla completa de BigQuery. DataFrame vacío si no existe todavía."""
    client = get_bq_client()
    try:
        client.get_table(table_id)
    except NotFound:
        return pd.DataFrame()
    return client.query(f"SELECT * FROM `{table_id}`").result().to_dataframe()


# ── BigQuery: escritura ────────────────────────────────────────────────────────
def cargar_dataframe_bq(
    df: pd.DataFrame,
    table_id: str,
    schema: list[bigquery.SchemaField],
    dataset_id: str,
) -> None:
    """
    Crea el dataset/tabla si no existen e inserta el DataFrame mediante
    un load job en modo WRITE_APPEND. A diferencia del patrón de Parquet
    local, esto no requiere traer la tabla completa a memoria: BigQuery
    hace el append del lado del servidor.
    """
    if df.empty:
        return

    crear_dataset_si_no_existe(dataset_id)
    crear_tabla_si_no_existe(table_id, schema)

    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # espera a que termine el load job; lanza excepción si falla


def sobrescribir_dataframe_bq(
    df: pd.DataFrame,
    table_id: str,
    schema: list[bigquery.SchemaField],
    dataset_id: str,
) -> None:
    """
    Igual que cargar_dataframe_bq, pero en modo WRITE_TRUNCATE: reemplaza
    todo el contenido de la tabla en vez de agregar filas. Se usa solo
    para facebook_post_metrics, que se recalcula completo en cada corrida
    (no es un append incremental como el resto de las tablas).
    """
    crear_dataset_si_no_existe(dataset_id)
    crear_tabla_si_no_existe(table_id, schema)

    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
