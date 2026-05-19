# Actividad AA1 — Asistente Virtual de Voz para Cardiología
**Módulo:** Procesamiento de Lenguaje Natural · Unidad 4  
**Máster IA en Ciencias de la Salud — Universidad Europea de Madrid**  
**Código:** 0MSR001107_UA4_AA1

---

## 1. ¿Qué hace el sistema?

El sistema permite a un médico hacer una pregunta clínica hablando al micrófono del navegador. En segundos recibe una respuesta cardiológica relevante, y la consulta queda registrada en una base de datos remota.

El flujo completo, de principio a fin, es:

```
Voz del médico
      │
      ▼
  Streamlit — st.audio_input()       ← graba en el navegador sin instalar nada
      │ bytes de audio (WAV)
      ▼
  Whisper (HuggingFace API)          ← convierte voz a texto en español
      │ texto transcrito
      ▼
  TF-IDF + Similitud Coseno          ← NLP: encuentra la frase más relevante
      │ respuesta + score de confianza
      ▼
  MySQL en Railway                   ← guarda todo con timestamp
      │
      ▼
  Streamlit — muestra resultado e historial
```

---

## 2. Capa de voz — reconocimiento de voz (speech.py)

### ¿Qué hace?

Convierte los bytes de audio grabados por el navegador en texto en español.

### Tecnología: Whisper

Se usa **OpenAI Whisper large-v3**, accedido a través de la API de HuggingFace. Whisper es un modelo de reconocimiento de voz (ASR — *Automatic Speech Recognition*) entrenado por OpenAI sobre 680.000 horas de audio multilingüe. La variante `large-v3` es la más precisa del modelo.

### ¿Cómo funciona Whisper internamente?

Whisper es una red neuronal del tipo **encoder-decoder**:

1. El audio se convierte en un **espectrograma de mel** (representación visual de las frecuencias del sonido a lo largo del tiempo).
2. El **encoder** (basado en Transformer) procesa ese espectrograma y extrae sus características.
3. El **decoder** genera el texto palabra a palabra, prediciendo en cada paso el siguiente token más probable dado lo que ya ha generado.

El modelo fue entrenado con supervisión directa: pares (audio, transcripción) reales, por lo que no necesita adaptación adicional para funcionar en español médico.

### Código

```python
# src/speech.py
cliente = InferenceClient(token=os.getenv("HF_TOKEN"))
resultado = cliente.automatic_speech_recognition(
    audio=audio_bytes,
    model="openai/whisper-large-v3",
)
return resultado.text
```

La función recibe los bytes del audio (formato WAV), los envía a la API de HuggingFace y devuelve el texto transcrito como cadena de caracteres.

---

## 3. Capa de PLN — recuperación de información con TF-IDF (nlp.py)

Esta es la parte central del sistema desde el punto de vista del PLN. No se usa ningún modelo generativo externo: toda la lógica corre localmente con `scikit-learn`.

### El problema

Dado el texto de una pregunta médica ("¿Cuáles son los síntomas del infarto agudo de miocardio?"), encontrar la frase del conocimiento cardiológico almacenado que mejor responde a esa pregunta.

### La solución: TF-IDF + Similitud Coseno

Se implementa un sistema de **recuperación de información** (*Information Retrieval*) clásico en tres pasos:

---

#### Paso 1 — Segmentación del contexto en frases

El contexto cardiológico de cada categoría (Síntomas, Medicamentos, Protocolos) es un texto largo. Se divide en frases individuales usando expresiones regulares:

```python
frases = re.split(r'(?<=[.?!])\s+', texto.strip())
```

Esta expresión divide el texto en cada punto, signo de interrogación o exclamación seguido de espacio, preservando el signo al final de la frase. Las frases de menos de 20 caracteres se descartan (son demasiado cortas para ser útiles).

**Ejemplo** — el contexto de Síntomas produce frases como:
- *"Los síntomas del infarto agudo de miocardio son dolor torácico intenso..."*
- *"La disnea cardiogénica aparece con el esfuerzo en estadios iniciales..."*
- *"Las palpitaciones son la percepción anormal del propio latido cardíaco."*
- *(y así sucesivamente)*

---

#### Paso 2 — Vectorización con TF-IDF

**TF-IDF** significa *Term Frequency — Inverse Document Frequency*. Es una técnica matemática que transforma texto en vectores numéricos capturando qué tan importante es cada palabra en cada frase.

Para cada palabra en cada frase se calcula:

```
TF-IDF(palabra, frase) = TF × IDF

TF  = frecuencia de la palabra en esa frase
      ──────────────────────────────────────
      total de palabras en esa frase

IDF = log( número total de frases / número de frases que contienen esa palabra )
```

**Intuición:** una palabra que aparece mucho en una frase concreta (TF alto) pero poco en el resto del corpus (IDF alto) recibe un peso elevado — es una palabra característica de esa frase. Palabras comunes como "el", "la", "es" aparecen en todas las frases, su IDF es casi cero y prácticamente no influyen.

En el código se usan **bigramas** además de palabras individuales (`ngram_range=(1, 2)`). Esto significa que "infarto agudo" se trata como una unidad, no como dos palabras sueltas, lo que mejora la precisión para términos médicos compuestos.

Todas las frases del contexto **más la pregunta del médico** se vectorizan a la vez:

```python
vectorizador = TfidfVectorizer(ngram_range=(1, 2))
matriz = vectorizador.fit_transform(frases + [pregunta])
```

El resultado es una matriz donde cada fila es un vector numérico que representa una frase. Todas las filas tienen la misma longitud (igual al número de términos únicos en el vocabulario).

---

#### Paso 3 — Similitud Coseno

Una vez que tenemos vectores, comparamos la pregunta con cada frase del contexto usando la **similitud coseno**:

```
similitud(A, B) = cos(θ) = (A · B) / (|A| × |B|)
```

La similitud coseno mide el ángulo entre dos vectores en un espacio de alta dimensión. Si dos vectores apuntan exactamente en la misma dirección (θ = 0°), la similitud es 1. Si son perpendiculares (sin palabras en común), es 0.

**¿Por qué coseno y no distancia euclidiana?** Porque el coseno es independiente de la longitud del vector, es decir, no penaliza que una frase sea más larga que otra. Sólo mide si comparten vocabulario importante en las mismas proporciones.

```python
similitudes = cosine_similarity(vec_pregunta, vec_frases).flatten()
idx = similitudes.argmax()
```

Se devuelve la frase con el índice de mayor similitud, y ese score (entre 0 y 1) es la **confianza** que muestra la aplicación.

---

### Ejemplo concreto paso a paso

**Pregunta:** *"¿Cuáles son los síntomas del infarto agudo de miocardio?"*

1. El contexto de Síntomas se divide en ~18 frases.
2. Se vectorizan las 18 frases + la pregunta con TF-IDF.
3. Palabras con peso alto en la pregunta: *infarto*, *agudo*, *miocardio*, *síntomas*, *infarto agudo* (bigrama).
4. La frase del contexto que más comparte ese vocabulario es: *"Los síntomas del infarto agudo de miocardio son dolor torácico intenso..."*
5. Similitud coseno ≈ 0.33 → confianza 33%.

La confianza no llega al 100% porque la pregunta y la frase no son idénticas y TF-IDF trabaja con vocabulario exacto (no con sinónimos ni semántica). Es un valor honesto: refleja la coincidencia real entre el texto de la pregunta y el texto de la respuesta.

---

## 4. Capa de persistencia — MySQL en Railway (db.py)

Cada consulta procesada se almacena en la base de datos remota.

### Esquema de la tabla

```sql
CREATE TABLE consultas (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    categoria     VARCHAR(20)   NOT NULL,   -- Síntomas / Medicamentos / Protocolos
    transcripcion TEXT          NOT NULL,   -- texto producido por Whisper
    respuesta     TEXT          NOT NULL,   -- frase devuelta por TF-IDF
    confianza     FLOAT,                    -- similitud coseno (0-1)
    duracion_ms   INT,                      -- tiempo de proceso en milisegundos
    fecha_hora    DATETIME DEFAULT NOW()    -- timestamp automático
);
```

### Operaciones implementadas

| Función | SQL | Descripción |
|---|---|---|
| `guardar()` | `INSERT INTO consultas ...` | Guarda una consulta tras cada respuesta |
| `obtener_historial(n)` | `SELECT ... LIMIT n` | Recupera las últimas n consultas |
| `borrar_historial()` | `DELETE FROM consultas` | Vacía la tabla completamente |

La conexión usa `mysql-connector-python` con las credenciales almacenadas en variables de entorno (nunca en el código fuente).

---

## 5. Interfaz — Streamlit (app.py)

Streamlit permite construir una aplicación web interactiva con código Python puro, sin HTML ni JavaScript.

### Componentes de la interfaz

- **`st.radio()`** — selector de categoría (Síntomas / Medicamentos / Protocolos)
- **`st.audio_input()`** — graba directamente desde el micrófono del navegador; devuelve los bytes del audio
- **`st.spinner()`** — indicador visual mientras se ejecutan Whisper y TF-IDF
- **`st.progress()`** — barra de confianza con código de color: verde ≥ 70%, naranja ≥ 40%, rojo < 40%
- **`st.dataframe()`** — tabla del historial de consultas
- **`st.download_button()`** — exporta el historial a CSV
- **Botón "Borrar historial"** — ejecuta `DELETE FROM consultas` y recarga la página

---

## 6. Conocimiento médico — contexto_cardio.py

El sistema no "sabe" medicina por sí solo: consulta un texto de referencia estructurado. El archivo `src/contexto_cardio.py` contiene un texto cardiológico por cada categoría, redactado con terminología clínica real.

Este diseño tiene una ventaja académica importante: permite **auditar exactamente qué sabe el sistema**. No hay una caja negra — cualquier respuesta que dé el sistema está literalmente escrita en ese archivo.

---

## 7. Arquitectura de archivos

```
pln-voz/
├── app.py                    ← interfaz Streamlit (punto de entrada)
├── requirements.txt          ← dependencias Python
├── .env                      ← credenciales (no sube a git)
├── .env.example              ← plantilla de variables de entorno
├── sql/
│   └── schema.sql            ← definición de la tabla MySQL
└── src/
    ├── speech.py             ← capa de voz: audio → texto (Whisper)
    ├── nlp.py                ← capa PLN: texto → respuesta (TF-IDF)
    ├── contexto_cardio.py    ← base de conocimiento cardiológico
    └── db.py                 ← capa de datos: MySQL
```

---

## 8. Variables de entorno necesarias

| Variable | Servicio | Uso |
|---|---|---|
| `HF_TOKEN` | HuggingFace | Autenticación para la API de Whisper |
| `DB_HOST` | Railway | Dirección del servidor MySQL |
| `DB_PORT` | Railway | Puerto público (≠ 3306 interno) |
| `DB_USER` | Railway | Usuario de la base de datos |
| `DB_PASSWORD` | Railway | Contraseña |
| `DB_NAME` | Railway | Nombre de la base de datos (`railway`) |

---

## 9. Cómo ejecutar el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Copiar y rellenar las variables de entorno
cp .env.example .env

# 3. Crear la tabla en MySQL (solo la primera vez)
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway < sql/schema.sql

# 4. Lanzar la aplicación
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

---

## 10. Limitaciones conocidas y reflexión crítica

### Limitación de TF-IDF: vocabulario exacto

TF-IDF trabaja con palabras literales, no con significado. Si el médico pregunta *"¿qué produce el IAM?"* y el contexto usa la expresión completa *"infarto agudo de miocardio"*, la similitud será baja porque *IAM* y *infarto agudo de miocardio* son cadenas distintas. Un sistema más avanzado usaría *embeddings* semánticos (como BERT o Sentence Transformers) que capturan el significado aunque las palabras sean diferentes.

### Limitación de la base de conocimiento

El sistema sólo puede responder sobre lo que está escrito en `contexto_cardio.py`. Una pregunta fuera de ese vocabulario devolverá la frase menos mala del contexto, no un reconocimiento de ignorancia. En un sistema de producción se añadiría un umbral: si la confianza es inferior a un valor mínimo, el sistema responde que no tiene información suficiente.

### Ventaja del enfoque para este contexto académico

La transparencia total es una ventaja: cada decisión del sistema es auditable, reproducible y explicable sin acceso a servicios externos de pago. Esto contrasta con los modelos de lenguaje generativos (GPT, Claude, Llama) donde el proceso interno es opaco.
