from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# === Cargar modelo BETO ===
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_embeddings_batch(texts, batch_size=32):
    """
    Genera embeddings por lotes a partir de una lista de textos,
    usando el modelo BETO. Retorna un arreglo numpy con los embeddings medios.
    """
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

            outputs = model(**encoded_input)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings).numpy()


def generar_embeddings(df, output_path, s3=None):
    """
    Genera embeddings para publicaciones y comentarios usando BETO.
    
    - Si existen columnas 'clean_comment_message' o 'clean_post_message', se crean
      las columnas 'comment_embedding' y 'post_embedding' respectivamente.
    - Los resultados se guardan localmente en:
        dreammaker/output/processed/embeddings_<fecha>.parquet
    - Opcionalmente, puede subirse el archivo resultante a S3 si se especifica.
    """
    if "clean_comment_message" in df.columns:
        print("Generando embeddings para comentarios...")
        embeddings = get_embeddings_batch(df["clean_comment_message"].tolist())
        df["comment_embedding"] = list(embeddings)

    if "clean_post_message" in df.columns:
        print("Generando embeddings para publicaciones...")
        embeddings_post = get_embeddings_batch(df["clean_post_message"].tolist())
        df["post_embedding"] = list(embeddings_post)

    # Guardado en la ruta especificada
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fecha_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"embeddings_{fecha_str}.parquet"
    df.to_parquet(file_path, index=False)

    print(f"✅ Datos con embeddings guardados en: {file_path}")

    return df

