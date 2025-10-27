import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path

def agrupar_post(df, n_clusters=11, plot=True, output_path="/dbfs/FileStore/dreammaker/output/processed", images_path="/dbfs/FileStore/dreammaker/output/images"):
    """
    Agrupa los posts usando KMeans sobre los embeddings,
    calcula proporciones de comentarios por clase y guarda
    los resultados y gráficos en Databricks.
    """
    # Agrupar por post y clase
    df_agrupados = df.groupby(
        ["post_id", "clean_post_message", "predicted_label"]
    ).size().unstack().fillna(0).reset_index()
    
    # Embeddings únicos por post_id
    embeddings_df = df[["post_id", "post_embedding"]].drop_duplicates(subset="post_id")
    df_agrupados = df_agrupados.merge(embeddings_df, how="left", on="post_id")
  
    # Volumen total de comentarios por post
    cols_clases = [
        "Comentario político emocional", "Crítica genuina",
        "Elogio al producto", "Otro", "Pregunta sobre el producto"
    ]
    df_agrupados["volumen"] = df_agrupados[cols_clases].sum(axis=1)

    # Suavizado bayesiano para proporciones
    N = 10
    df_agrupados["proporcion_politica"] = df_agrupados["Comentario político emocional"] / (df_agrupados["volumen"] + N)
    df_agrupados["proporcion_de_interes"] = df_agrupados["Pregunta sobre el producto"] / (df_agrupados["volumen"] + N)

    # Matriz X para clustering
    X = np.vstack(df_agrupados["post_embedding"].values)

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, init='k-means++',
        n_init=10, max_iter=300
    )
    df_agrupados["cluster"] = kmeans.fit_predict(X)

    # --- Guardado de gráficos ---
    time_str = datetime.now().strftime("%Y-%m-%d")
    image_dir = Path(images_path)
    image_dir.mkdir(exist_ok=True, parents=True)

    # Gráfico proporción política
    fig1_path = image_dir / f"proporcion_politica_por_cluster_{time_str}.png"
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_agrupados, x='cluster', y="proporcion_politica")
    plt.savefig(fig1_path)
    print(f"✅ Guardé gráfico: {fig1_path}")
    if plot:
        plt.show()
    plt.close()

    # Gráfico proporción de interés
    fig2_path = image_dir / f"proporcion_interes_por_cluster_{time_str}.png"
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_agrupados, x='cluster', y="proporcion_de_interes")
    plt.savefig(fig2_path)
    print(f"✅ Guardé gráfico: {fig2_path}")
    if plot:
        plt.show()
    plt.close()

    # --- Guardado de clusters ---
    path_clust = Path(output_path)
    path_clust.mkdir(exist_ok=True, parents=True)
    archivo_clust = path_clust / f"Clustered_posts_{time_str}.parquet"
    df_agrupados.to_parquet(archivo_clust, index=False)
    print(f"✅ Clusters guardados en: {archivo_clust}")

    return df_agrupados


