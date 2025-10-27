import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import re
from pathlib import Path


# Esta es una sección únicamente para generar algunos datos rápidos preliminares antes de 
# pasar a PowerBI (u otro software).
# Las tablas estadísticas, reportes y visualizaciones reales se armaron en PowerBI
# alimentándose de los datos generados por classification, clustering y 'Resultados finales'.

# -------------------
# Funciones de clasificación y colores
# -------------------
def crear_columnas_clasificacion(df):
    """
    Crea columnas binarias para detectar palabras clave de productos
    y para clasificar posts según institución o tema relevante.
    """
    keywords = ["cm", "dimensiones", "escala", "impreso", "fabricados", "material", "pintura","pintado"]
    pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
    df['post_producto_materializado'] = df['clean_post_message'].str.contains(pattern, flags=re.IGNORECASE, regex=True)

    clases = {
        "carabineros": ["carabinero", "carabineros"],
        "ejercito": ["ejercito", "comando"],
        "gdp": ["guerra del pacifico", "gdp"],
        "fach": ["fuerza aerea", "fach"],
        "pdi": ["pdi", "policia de investigaciones"],
        "marina_inf": ["infanteria", "marina"]
    }

    for clase, palabras in clases.items():
        df[clase] = df["clean_post_message"].apply(lambda x: any(palabra.lower() in x.lower() for palabra in palabras))

    return df

def asignar_colores_y_bordes(df):
    """
    Asigna colores a cada post según institución y bordes si es producto materializado.
    """
    def color(row):
        if row['ejercito']:
            return 'red'
        elif row['fach']:
            return 'blue'
        elif row['carabineros']:
            return 'green'
        elif row['marina_inf']:
            return 'saddlebrown'
        elif 'pinochet' in row['clean_post_message'].lower():
            return 'purple'
        else:
            return 'grey'

    df['color'] = df.apply(color, axis=1)
    df['edge_color'] = df['post_producto_materializado'].apply(lambda x: 'black' if x else 'none')
    return df

# -------------------
# Función de guardado de figuras
# -------------------
def save_fig(fig, name_prefix, folder=None):
    """
    Guarda la figura en la carpeta especificada.
    Por defecto, se guarda en '/dbfs/FileStore/dreammaker/output/images'.
    """
    if folder is None:
        folder = "/dbfs/FileStore/dreammaker/output/images"
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = folder_path / f"{name_prefix}_{fecha}.png"
    fig.savefig(filename, bbox_inches='tight')
    print(f"✅ Gráfico '{name_prefix}' guardado en: {filename}")

# -------------------
# Funciones de graficado
# -------------------
def plot_proporcion(df, images_path=None):
    """
    Grafica la proporción de comentarios políticos vs de interés
    por institución y guarda la figura.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.scatter(
            row['proporcion_politica'],
            row['proporcion_de_interes'],
            c=row['color'],
            edgecolors=row['edge_color'],
            s=100,
            linewidths=1
        )
    ax.set_xlabel("Proporción política")
    ax.set_ylabel("Proporción de interés")
    ax.set_title("Proporción de interés vs Proporción política por institución")
    ax.grid(True, linestyle='--', alpha=0.5)

    legend_elements = [
        mpatches.Patch(color='red', label='Ejército'),
        mpatches.Patch(color='blue', label='Fuerza Aérea (FACH)'),
        mpatches.Patch(color='green', label='Carabineros'),
        mpatches.Patch(color='saddlebrown', label='Infantería Marina'),
        mpatches.Patch(color='purple', label='Pinochet'),
        mpatches.Patch(color='grey', label='Otros'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Producto materializado')
    ]
    ax.legend(handles=legend_elements, loc='best')
    save_fig(fig, "proporcion_interes_vs_politica", folder=images_path)
    plt.close(fig)

def plot_volumen(df, x_col="Comentario político emocional", y_col="Pregunta sobre el producto", log_scale=False, images_path=None):
    """
    Grafica volumen de comentarios por post y clase.
    Permite escala logarítmica.
    """
    df_plot = df.copy()
    if log_scale:
        df_plot[x_col + "_log"] = np.log1p(df_plot[x_col])
        df_plot[y_col + "_log"] = np.log1p(df_plot[y_col])
        x_col, y_col = x_col + "_log", y_col + "_log"

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df_plot.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            c=row['color'],
            edgecolors=row['edge_color'],
            s=100,
            linewidths=1,
            alpha=0.7
        )
    ax.set_xlabel(x_col.replace('_log',' (log1p)'))
    ax.set_ylabel(y_col.replace('_log',' (log1p)'))
    ax.set_title(f"{y_col.replace('_log','')} vs {x_col.replace('_log','')}")
    ax.grid(True, linestyle='--', alpha=0.5)

    legend_elements = [
        mpatches.Patch(color='red', label='Ejército'),
        mpatches.Patch(color='blue', label='Fuerza Aérea (FACH)'),
        mpatches.Patch(color='green', label='Carabineros'),
        mpatches.Patch(color='saddlebrown', label='Infantería Marina'),
        mpatches.Patch(color='purple', label='Pinochet'),
        mpatches.Patch(color='grey', label='Otros'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Producto materializado')
    ]
    ax.legend(handles=legend_elements, loc='best')

    prefix = "volumen_log" if log_scale else "volumen"
    save_fig(fig, prefix, folder=images_path)
    plt.close(fig)

# -------------------
# Función de reporte estadístico
# -------------------
def generar_reporte_estadistico(df, filename=None, reports_path=None):
    """
    Genera un CSV con estadísticas de posts por clase y tipo de comentario,
    y lo guarda en la carpeta especificada (default: '/dbfs/FileStore/dreammaker/output/reports').
    """
    if reports_path is None:
        reports_path = "/dbfs/FileStore/dreammaker/output/reports"

    clases = ['carabineros','ejercito','gdp','fach','pdi','marina_inf','post_producto_materializado']
    tipos_comentario = ['Crítica genuina','Elogio al producto','Otro','Comentario político emocional','Pregunta sobre el producto']
    stats_list = []

    for clase in clases:
        df_clase = df[df[clase]]
        total_posts = len(df_clase)
        fila = {'clase': clase, 'total_posts': total_posts}
        for tipo in tipos_comentario:
            count = df_clase[tipo].sum() if tipo in df_clase.columns else 0
            fila[f"count_{tipo}"] = count
            fila[f"prop_{tipo}"] = count / total_posts if total_posts > 0 else np.nan
        stats_list.append(fila)

    df_stats = pd.DataFrame(stats_list)

    if filename is None:
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reporte_estadistico_{fecha}.csv"

    folder_path = Path(reports_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    path_local = folder_path / filename
    df_stats.to_csv(path_local, index=False)
    print(f"✅ Reporte estadístico guardado en: {path_local}")

    return df_stats


