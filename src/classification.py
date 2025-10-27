import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

def predecir_clases(df, output_path):
    """
    Carga el modelo local desde dreammaker/models/, 
    realiza la predicción de clases y guarda los resultados en output_path.
    """
    # Ruta del modelo entrenado
    model_path = Path("/dbfs/FileStore/dreammaker/models/modelo_reg.pkl")
    if not model_path.exists():
        raise RuntimeError(f"No se encontró el modelo en: {model_path}. "
                           "Debes colocar el archivo 'modelo_reg.pkl' dentro de /dbfs/FileStore/dreammaker/models/ antes de ejecutar.")

    # Cargar modelo
    model = joblib.load(model_path)

    # Convertir embeddings en un array 2D de NumPy
    X = np.vstack(df["comment_embedding"].values)

    # Predicciones
    y_pred = model.predict(X)
    df["predicted_class"] = y_pred

    # Mapeo de clase → etiqueta descriptiva
    mapeo = {
        0: "Crítica genuina",
        1: "Comentario político emocional",
        2: "Elogio al producto",
        3: "Pregunta sobre el producto",
        4: "Otro"
    }
    df["predicted_label"] = df["predicted_class"].map(mapeo)

    # Guardado de resultados en la ruta especificada
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fecha_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"comentarios_clasificados_{fecha_str}.parquet"
    df.to_parquet(file_path, index=False)

    print(f"✅ Resultados de clasificación guardados en: {file_path}")

    return df