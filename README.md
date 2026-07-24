# Facebook Comments NLP Pipeline

Pipeline de análisis de lenguaje natural sobre comentarios de Facebook para **DreamMaker**, e-commerce chileno de figuras coleccionables de instituciones y personajes históricos nacionales.

El sistema extrae comentarios en español chileno desde la Facebook Graph API, los vectoriza con BETO (BERT en español), y los clasifica mediante un **clasificador jerárquico en dos capas** que detecta intenciones comerciales y sarcasmo condicionado. El resultado final es un dataset analítico a nivel publicación, enriquecido con métricas agregadas y variables temáticas, listo para consumo en herramientas de visualización como Power BI.

> **Versionado:** esta rama (`main`) contiene la arquitectura jerárquica multilabel descrita en este README, que es la versión activa del proyecto. La versión inicial del clasificador (modelo multiclase de una sola etapa) se conserva en la rama [`legacy/v1`](../../tree/legacy/v1) como referencia histórica. Ver la sección [Evolución del clasificador: de v1 a la arquitectura jerárquica](#evolución-del-clasificador-de-v1-a-la-arquitectura-jerárquica) para el detalle de por qué se migró.

---

## El problema

Las publicaciones de DreamMaker generan un alto volumen de interacción donde los comentarios no siempre reflejan intención real de compra. Gran parte del contenido está influenciado por factores externos: emociones intensas, polarización ideológica, ironía y sarcasmo, introduciendo una fuerte distorsión en la lectura del valor comercial de dichas interacciones.

El problema no se limita a un análisis de sentimiento tradicional. Se trata de un problema de **interpretación semántica estructurada**, donde un mismo comentario puede contener simultáneamente intención de compra, evaluación del producto, carga emocional y distorsión sarcástica.

En particular, el sistema debe diferenciar entre:

- Intención real de compra o interés comercial
- Elogios o críticas directas hacia el producto
- Solicitudes explícitas de diseño, personalización o información
- Comentarios irrelevantes o ruido conversacional
- Expresiones emocionales o ideológicas sin asociación comercial
- **Sarcasmo como mecanismo de inversión semántica del significado literal**

El sarcasmo es especialmente crítico: puede alterar completamente la interpretación de otras etiquetas, convirtiendo un elogio aparente en una crítica real, o una solicitud en ruido ideológico. Por ello, su detección no puede tratarse como una tarea aislada.

---

## Objetivo

Desarrollar un sistema de clasificación basado en embeddings semánticos que identifique múltiples dimensiones del contenido de los comentarios y transforme dichas predicciones en una **señal de valor comercial real**.

El objetivo final no es clasificar comentarios individuales, sino construir un **dataset analítico a nivel de publicación** donde cada post queda caracterizado mediante métricas agregadas, proporciones suavizadas y variables temáticas que permiten análisis exploratorio, segmentación y extracción de conocimiento comercial.

Este dataset es la entrada principal del análisis en `02_commercial_signal_analysis.ipynb`.

---

## Arquitectura

El pipeline implementa una arquitectura **Medallion** (Bronze → Silver → Gold) con procesamiento incremental en cada capa.

```
pipeline.py
│
├── [1] scraper_facebook.py   →  data/bronze/
├── [2] preprocessing.py      →  data/silver/  (CSV)
├── [3] embedding.py          →  data/silver/  (Parquet)
├── [4] classification.py     →  data/gold/    (comentarios)
└── [5] aggregation.py        →  data/gold/    (publicaciones)
```

### Comportamiento incremental

En cada ejecución, el pipeline detecta automáticamente qué comentarios son nuevos comparando contra Bronze y procesa **únicamente el delta**. Si no hay comentarios nuevos, cada módulo lo informa y sale limpiamente sin modificar los archivos existentes.

En la primera ejecución, procesa y persiste todo desde cero sin configuración adicional.

La única excepción es el módulo de agregación (`aggregation.py`), que siempre recalcula sobre el **Gold completo acumulado**, no solo sobre el delta (ver detalle más abajo).

---

## Etapas del pipeline

### 1. Scraping — Bronze

Extrae posts y comentarios desde la **Facebook Graph API v18.0** con paginación segura.

- Deduplicación de posts y comentarios por ID en memoria
- Control de loops de paginación mediante set de URLs visitadas
- Descarte temprano de comentarios sin texto (stickers, GIFs, imágenes) antes de persistir
- Detección incremental: carga los `comment_id` existentes en Bronze y hace append solo con los nuevos

**Salida:** `data/bronze/facebook_comments_raw.csv`

### 2. Preprocesamiento — Silver CSV

Limpieza de texto sobre el delta recibido desde Bronze.

- Filtrado de filas con `comment_message` vacío (segunda línea de defensa)
- Normalización: lowercase, eliminación de tildes (NFD), colapso de saltos de línea y espacios múltiples, colapso de signos repetidos
- Genera `clean_comment_message` y `clean_post_message`
- Append incremental al Silver CSV

**Salida:** `data/silver/facebook_comments_clean.csv`

### 3. Embeddings — Silver Parquet

Vectorización con **BETO** (`dccuchile/bert-base-spanish-wwm-cased`), modelo BERT preentrenado en español.

- Mean pooling sobre la última capa oculta → vector de 768 dimensiones
- Genera `embedding` (comentario) y `post_embedding` (publicación)
- Procesamiento por lotes configurable (`batch_size=32`)
- El modelo se carga una sola vez al importar el módulo
- Append incremental al Parquet Silver

**Salida:** `data/silver/facebook_comments_embeddings.parquet`

### 4. Clasificación jerárquica — Gold (comentarios)

El problema de clasificación se aborda mediante una **arquitectura jerárquica de dos capas**, diseñada para desacoplar la interpretación semántica general de la detección específica de sarcasmo.

Inicialmente se evaluó un único modelo multilabel; sin embargo, se observó que el sarcasmo es un fenómeno altamente dependiente del contexto de intención, dificultando su detección cuando se analiza únicamente el texto. Por este motivo se implementó la arquitectura jerárquica: las predicciones de intención de Capa 1 actúan como contexto explícito para el detector de sarcasmo en Capa 2.

> Esta arquitectura de dos capas es, a su vez, la segunda generación del clasificador del proyecto. La primera versión (rama `legacy/v1`) usaba un único modelo multiclase de etiqueta única — ver [Evolución del clasificador](#evolución-del-clasificador-de-v1-a-la-arquitectura-jerárquica) para el detalle completo de ese cambio.

#### Capa 1 — Clasificación multilabel de intenciones

Modelo de **Regresión Logística One-vs-Rest** entrenado sobre embeddings BETO de 768 dimensiones.

Predice 8 etiquetas no mutuamente excluyentes que representan la interpretación semántica base del comentario:

| Etiqueta | Descripción |
|---|---|
| `interes` | Interés comercial en el producto |
| `elogio_producto` | Elogio o valoración positiva del producto |
| `critica_producto` | Crítica o reclamo sobre el producto |
| `fanatismo_emocional` | Comentario emocional o fanático sin intención comercial |
| `conflictivo` | Contenido político o polarizante |
| `solicitud` | Solicitud de información, precio o disponibilidad |
| `cuidado` | Expresión de afecto o cuidado |
| `otro` | Sin categoría identificable |

Cada etiqueta tiene un **threshold optimizado individualmente** para maximizar F1 en clases desbalanceadas.

#### Capa 2 — Detección de sarcasmo condicionado

Modelo **XGBoost** que recibe como input la **concatenación** del embedding del comentario (768-dim) con las predicciones binarias de Capa 1 (8-dim), totalizando un vector de 776 dimensiones.

Desde el punto de vista del aprendizaje automático, el sarcasmo deja de ser una tarea aislada para convertirse en una **clasificación contextualizada**: el modelo dispone de información explícita sobre la intención aparente del comentario, lo que le permite distinguir con mayor precisión entre expresiones literales e irónicas.

#### Reglas de negocio

Las predicciones de ambas capas se combinan mediante reglas lógicas para construir etiquetas orientadas al análisis comercial:

| Variable | Lógica |
|---|---|
| `interes_real` | `interes=1` AND `sarcasmo=0` |
| `solicitud_real` | `solicitud=1` AND `sarcasmo=0` |
| `elogio_real` | `elogio_producto=1` AND `sarcasmo=0` |
| `critica_genuina` | `critica_producto=1` AND `sarcasmo=0` AND `conflictivo=0` |
| `polarizacion_politica` | `conflictivo=1` |
| `fanatismo` | `fanatismo_emocional=1` AND `sarcasmo=0` |
| `fanatismo_no_comercial` | `fanatismo_emocional=1` AND `sarcasmo=0` AND `interes=0` |

**Salidas:**
- `data/gold/facebook_comments_classified.parquet` — predicciones binarias
- `data/gold/facebook_comments_probabilities.parquet` — probabilidades por etiqueta
- `data/gold/facebook_comments_gold_enriched.parquet` — columnas originales + predicciones + reglas de negocio

### 5. Agregación — Gold (publicaciones)

Una vez clasificadas todas las interacciones individuales, el pipeline cambia la unidad de análisis desde el comentario hacia la **publicación**. Esta transformación constituye uno de los principales aportes del pipeline.

#### Construcción de señales compuestas

Las etiquetas derivadas por las reglas de negocio no son mutuamente excluyentes: un mismo comentario puede manifestar interés comercial, realizar una solicitud y además elogiar el producto. Si se sumaran todas las categorías directamente, un mismo comentario sería contabilizado múltiples veces.

Para evitar este sesgo se construyen primero dos **señales compuestas binarias** mediante operadores lógicos OR, de modo que cada comentario aporte como máximo una unidad:

- **`comentario_senal_comercial`**: `interes_real` OR `solicitud_real` OR `elogio_real` OR `critica_genuina`
- **`comentario_ruido_social`**: `polarizacion_politica` OR `fanatismo_no_comercial`

#### Métricas por publicación con suavizado bayesiano

Las publicaciones presentan volúmenes de comentarios muy dispares. Utilizar proporciones clásicas produce estimaciones inestables cuando una publicación tiene poca evidencia.

Para reducir este problema se aplica **Bayesian Smoothing** con un prior estimado a partir del comportamiento global del dataset:

```
prop = (volumen_label + α × p_global) / (volumen_total + α)
```

donde `α = mediana del volumen de comentarios por publicación` (prior adaptativo) y `p_global` es la proporción global de cada etiqueta. Esto reduce la variabilidad en publicaciones con pocos comentarios sin modificar sustancialmente aquellas con abundante evidencia.

Se calculan volúmenes y proporciones suavizadas para: interés, solicitudes, elogios, críticas, polarización, fanatismo, fanatismo no comercial, señal comercial y ruido social.

Este módulo siempre opera sobre el **Gold completo acumulado** (no solo el delta) para que el prior sea consistente con todos los datos disponibles, y re-aplica las reglas de negocio de `classification.py` en cada ejecución para reflejar siempre la versión actual de las reglas.

#### Clasificación temática de publicaciones

Variables binarias inferidas por regex sobre el texto de la publicación (no mutuamente excluyentes):

| Variable | Detecta |
|---|---|
| `post_producto_materializado` | Menciones a características físicas del producto (cm, escala, material, etc.) |
| `post_carabineros` | Carabineros de Chile |
| `post_ejercito` | Ejército de Chile |
| `post_fach` | Fuerza Aérea de Chile |
| `post_armada` | Armada / Infantería de Marina |
| `post_pdi` | Policía de Investigaciones |
| `post_guerra_pacifico` | Guerra del Pacífico |
| `post_personajes_historicos` | Personajes históricos chilenos (Allende, Pinochet, O'Higgins, Prat, etc.) |

**Salida:** `data/gold/facebook_post_metrics.parquet`

---

## Estructura del repositorio

```
dreammaker-signal-analysis/
│
├── src/
│   ├── __init__.py
│   ├── scraper_facebook.py      # Extracción desde Facebook Graph API
│   ├── preprocessing.py         # Limpieza de texto
│   ├── embedding.py             # Vectorización con BETO
│   ├── classification.py        # Clasificador jerárquico + reglas de negocio
│   └── aggregation.py           # Métricas por publicación + suavizado bayesiano
│
├── models/
│   ├── logistic_ovr_model.pkl       # Capa 1: modelo OvR entrenado
│   ├── logistic_ovr_thresholds.pkl  # Capa 1: thresholds por etiqueta
│   ├── xgb_sarcasm_model.pkl        # Capa 2: modelo XGBoost
│   └── xgb_sarcasm_threshold.pkl    # Capa 2: threshold de sarcasmo
│
├── data/                         # Bronze / Silver / Gold (no incluida en el repo)
├── labeling/                     # Dataset de entrenamiento (no incluida en el repo)
├── outputs/                      # Resultados y exportaciones del análisis
├── reports/
│   └── DreamMaker_Commercial_Analysis_Report.pdf
│
├── pipeline.py                   # Orquestador principal
├── README.md
├── requirements.txt
├── .gitignore
├── my_env.env.example            # Plantilla de variables de entorno
├── 01_semantic_classification_pipeline.ipynb
└── 02_commercial_signal_analysis.ipynb
```

> **Nota:** Las carpetas `data/` y `labeling/` no se incluyen en el repositorio. `data/` se crea automáticamente al ejecutar el pipeline. `labeling/` contiene muestras reales de comentarios y publicaciones de la PYME utilizadas para entrenar los modelos de Capa 1 y Capa 2, por lo que se omite por confidencialidad de los datos del cliente. El notebook `01_semantic_classification_pipeline.ipynb` documenta la metodología de entrenamiento (features, arquitectura, thresholds, evaluación), pero no es 100% reproducible de punta a punta sin ese dataset.

---

## Instalación

```bash
git clone https://github.com/Claudio-Val/dreammaker-signal-analysis.git
cd dreammaker-signal-analysis
pip install -r requirements.txt
```

Requiere **Python 3.12**.

---

## Configuración

Crea el archivo `my_env.env` en la raíz del proyecto a partir de la plantilla:

```bash
cp my_env.env.example my_env.env
```

Edita `my_env.env` con tus credenciales:

```env
FACEBOOK_TOKEN=your_facebook_page_access_token_here
PAGE_ID=your_facebook_page_id_here
```

### Cómo obtener las credenciales

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `FACEBOOK_TOKEN` | Token de acceso de página con permisos `pages_read_engagement` y `pages_read_user_content` | [Meta for Developers → Graph API Explorer](https://developers.facebook.com/tools/explorer/) |
| `PAGE_ID` | ID numérico de la página de Facebook a analizar | URL de la página o desde el Graph API Explorer |

> **Importante:** El token de acceso de página tiene una duración limitada. Para uso en producción se recomienda generar un token de larga duración desde la documentación oficial de Meta.

---

## Uso

```bash
python pipeline.py
```

El pipeline detecta automáticamente si es la primera ejecución o si hay comentarios nuevos desde la última vez que se corrió.

**Salida esperada:**

```
[1/5] Extrayendo comentarios nuevos desde Facebook...
  Bronze existente cargado: 2961 comment_id registrados.
  Posts extraídos desde la API: 197
  Comentarios únicos extraídos desde la API: 2985
  ─────────────────────────────────────────────
  24 comentarios nuevos agregados a Bronze.
  Total Bronze acumulado: 2985 registros.
  ─────────────────────────────────────────────

[2/5] Preprocesando comentarios nuevos...
[3/5] Generando embeddings BETO...
[4/5] Clasificando comentarios nuevos (pipeline jerárquico)...
[5/5] Calculando métricas por publicación...

✅ Pipeline incremental finalizado.
   Comentarios nuevos procesados : 24
   Publicaciones con métricas    : 197
```

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `transformers` | Modelo BETO para embeddings |
| `torch` | Backend de inferencia |
| `scikit-learn` | Regresión Logística OvR, métricas |
| `xgboost` | Clasificador de sarcasmo |
| `pandas` | Manipulación de datos |
| `requests` | Llamadas a la Facebook Graph API |
| `python-dotenv` | Carga de variables de entorno |
| `joblib` | Serialización de modelos |
| `tqdm` | Barras de progreso en generación de embeddings |

---

## Outputs finales

| Archivo | Nivel | Descripción |
|---|---|---|
| `data/bronze/facebook_comments_raw.csv` | Comentario | Datos crudos de la API |
| `data/silver/facebook_comments_clean.csv` | Comentario | Texto limpio y normalizado |
| `data/silver/facebook_comments_embeddings.parquet` | Comentario | Embeddings BETO (768-dim) |
| `data/gold/facebook_comments_classified.parquet` | Comentario | Predicciones binarias |
| `data/gold/facebook_comments_probabilities.parquet` | Comentario | Probabilidades por etiqueta |
| `data/gold/facebook_comments_gold_enriched.parquet` | Comentario | Predicciones + reglas de negocio |
| `data/gold/facebook_post_metrics.parquet` | **Publicación** | Métricas agregadas + suavizado bayesiano + clasificación temática |

---

## Resultados e insights comerciales

El análisis comercial completo y los principales hallazgos están disponibles en:

📄 [Informe de análisis comercial de DreamMaker](reports/DreamMaker_Commercial_Analysis_Report.pdf)

El informe cubre:

- Análisis exploratorio de señales comerciales vs. sociales
- Contraste de hipótesis estadísticas
- Regresión binomial negativa
- Análisis de árboles de decisión sobre regímenes de comportamiento viral
- Insights de negocio derivados de 9 años de interacciones en Facebook

---

## Evolución del clasificador: de v1 a la arquitectura jerárquica

La versión inicial del proyecto (conservada en la rama [`legacy/v1`](../../tree/legacy/v1)) usaba un enfoque distinto y considerablemente más simple, tanto en el problema de clasificación como en la infraestructura:

- **Un único modelo**, entrenado como **clasificación multiclase de etiqueta única** (`model.predict(X)` retorna una sola clase por comentario), no multilabel.
- **5 clases mutuamente excluyentes**: `Crítica genuina`, `Comentario político emocional`, `Elogio al producto`, `Pregunta sobre el producto`, `Otro`.
- Desplegado sobre Databricks, cargando el modelo desde `dbfs:/FileStore/dreammaker/models/modelo_reg.pkl`, sin la arquitectura Bronze/Silver/Gold incremental ni el pipeline modular de la versión actual.

**El problema que motivó el cambio:** el supuesto de exclusividad mutua no se sostenía en los datos reales. Un mismo comentario puede, al mismo tiempo, expresar interés comercial *y* ser políticamente cargado ("me encantaría comprarlo pero apoyar esto es una vergüenza"), o combinar una solicitud de información con fanatismo emocional sin intención de compra. Forzar una sola etiqueta por comentario obligaba al modelo a elegir una de esas dimensiones y descartar la otra, perdiendo información comercial válida en exactamente los casos más ambiguos y polarizados — que en este catálogo son frecuentes, no marginales.

**La solución:** migrar a la arquitectura jerárquica multilabel descrita en la sección [Clasificación jerárquica](#4-clasificación-jerárquica--gold-comentarios) de este README: una Capa 1 que predice 8 etiquetas no excluyentes de forma independiente, y una Capa 2 que usa esas predicciones como contexto para detectar sarcasmo — permitiendo que un comentario sea, por ejemplo, simultáneamente `interes=1` y `conflictivo=1`, en vez de forzarlo a una sola categoría.

La rama `legacy/v1` se mantiene por trazabilidad histórica del proyecto, pero no recibe mantenimiento; toda mejora activa ocurre sobre `main`.

---

## Notas de diseño

**¿Por qué arquitectura jerárquica para el sarcasmo?**
Se evaluó inicialmente un único modelo multilabel que incluyera sarcasmo como una etiqueta más. Sin embargo, el sarcasmo es un fenómeno contextual que depende fuertemente de la intención aparente del mensaje: un elogio puede ser sarcástico, pero solo puede identificarse como tal si se conoce primero que el comentario pretende elogiar. Al separar en dos capas y pasar las predicciones de intención como input al detector de sarcasmo, el modelo tiene contexto explícito para identificar ese contraste.

**¿Por qué señales compuestas antes de agregar?**
Las etiquetas de negocio no son mutuamente excluyentes. Un comentario puede tener `interes_real=1`, `solicitud_real=1` y `elogio_real=1` simultáneamente. Si se sumaran directamente las tres columnas al agregar por publicación, ese comentario contaría tres veces en el volumen de señal comercial. Las señales compuestas (`comentario_senal_comercial`, `comentario_ruido_social`) garantizan que cada comentario aporte como máximo una unidad, independientemente de cuántas etiquetas active.

**¿Por qué suavizado bayesiano en las métricas por post?**
Las publicaciones tienen volúmenes de comentarios muy dispares. Sin suavizado, una publicación con 2 comentarios donde 1 es de interés tendría `prop_interes = 0.5`, lo cual es una estimación poco confiable estadísticamente. El prior adaptativo (`α = mediana del volumen`) regula esta varianza: publicaciones con pocos comentarios se acercan al promedio global, mientras que publicaciones con mucha evidencia no se ven afectadas.

**¿Por qué el módulo de agregación recalcula siempre sobre el Gold completo?**
Las proporciones globales usadas como prior bayesiano deben reflejar el estado real del dataset en cada ejecución. Si solo se recalculara sobre el delta, el prior estaría sesgado por la muestra más reciente y las métricas de todas las publicaciones cambiarían de forma incoherente entre ejecuciones.

---

## Autor

**Claudio Valenzuela**
- GitHub / Portafolio: [github.com/Claudio-Val](https://github.com/Claudio-Val)
- Email: claudio.valenzuela.val@gmail.com
