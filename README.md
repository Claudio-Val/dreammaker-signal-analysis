#**DreamMaker Data Pipeline**

## Contexto del Proyecto

DreamMaker desarrolla modelos de instituciones chilenas y personajes históricos, como Carabineros, Ejército de Chile, Pinochet, Allende, entre otros. Los productos generan distintos tipos de reacciones en el público, que incluyen:

Comentarios políticos o emocionales (odio, fanatismo, debate político).

Comentarios de producto (elogios a la calidad, críticas genuinas, preguntas de interés de compra o despacho).

Para DreamMaker es crucial identificar el verdadero interés en compra, filtrando el “ruido” generado por las reacciones políticas o emocionales. Esto permite tomar decisiones de diseño y estrategia más informadas, enfocándose en las interacciones que realmente predicen ventas.

## Objetivo del Pipeline

El pipeline de DreamMaker permite:

Extraer datos desde Facebook usando la API Graph.

Preprocesar y limpiar los textos de publicaciones y comentarios.

Generar embeddings para posts y comentarios usando BETO (modelo BERT en español).

Clasificar comentarios mediante un modelo supervisado previamente entrenado (XGBoost/Regresión logística) en categorías como:

Críticas genuinas al producto

Elogios al producto

Comentarios relacionados con interés en compra/despacho

Comentarios políticos/emocionales (ruido)

Agrupar posts en clusters según embeddings y proporciones de tipo de comentario.

Analizar métricas e intereses por institución, personaje histórico o tipo de modelo.

Generar reportes y gráficos listos para alimentar Power BI y visualizar indicadores clave como:

Interés en compra vs. reacciones políticas

Instituciones o tipos de modelos más alabados

Clusters de publicaciones y patrones de reacción

Este pipeline permite a DreamMaker anticipar la respuesta del público a nuevos prototipos, optimizando decisiones de diseño y estrategias de lanzamiento.

## Estructura del Proyecto
DreamMaker/
- models/       # Modelos entrenados para clasificación de comentarios
- src/          # Scripts del pipeline
  - scraper_facebook.py
  - preprocessing.py
  - embedding.py
  - classification.py
  - clustering.py
  - analysis.py
- output/       # Carpeta para resultados: raw, processed, reports, images
- .env          # Variables de entorno (ej. FACEBOOK_TOKEN)
- requirements.txt  # Dependencias de Python
- README.md

## Instalación

Crear un entorno virtual de Python (opcional):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Instalar dependencias:

pip install -r requirements.txt


Crear un archivo .env en la raíz con tus variables de entorno:

FACEBOOK_TOKEN=your_facebook_token_here

## Uso

El pipeline se ejecuta desde pipeline.py:

python pipeline.py


El flujo incluye:

Scraping de publicaciones y comentarios de Facebook.

Preprocesamiento de textos.

Generación de embeddings con BETO.

Clasificación de comentarios según su tipo.

Clustering de publicaciones.

Análisis y generación de reportes y gráficos.

Guardado de resultados finales en output/.

Actualmente, el pipeline está configurado para ejecutarse en Databricks usando un mount point de Data Lake (/mnt/dreammaker-scraper/output/). Para ejecución local, es necesario cambiar las rutas de salida a carpetas locales.

Salidas del Pipeline

Raw: Datos crudos extraídos de Facebook.

Processed: Datos preprocesados, embeddings y clasificaciones.

Reports: Reportes estadísticos y métricas agregadas.

Images: Gráficos generados para análisis y visualización en Power BI.

Notas Adicionales

El pipeline no expone datos sensibles; la conexión a Facebook se realiza mediante el token definido en .env.

Los resultados clave se utilizan en Power BI, donde se crean dashboards interactivos para monitorear interés de compra, polarización política y desempeño de modelos e instituciones.

La estructura modular permite actualizar componentes individuales sin afectar todo el flujo.