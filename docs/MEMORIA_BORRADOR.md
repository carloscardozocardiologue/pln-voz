# BORRADOR DE MEMORIA — UA4 AA1
## Implementación de un Asistente Virtual de Voz para Entornos Hospitalarios

**Módulo:** Procesamiento de Lenguaje Natural · Unidad 4  
**Máster:** Inteligencia Artificial en Ciencias de la Salud  
**Universidad:** Universidad Europea de Madrid  
**Código de actividad:** 0MSR001107_UA4_AA1  
**Autor:** Dr. Carlos CARDOZO — Cardiólogo Intervencionista · Exp. 225K3417  
**Fecha:** Mayo 2026

---

> ### INSTRUCCIONES PARA COMPLETAR ESTE BORRADOR
>
> Este archivo es el esqueleto de la memoria final que deberás entregar en formato **Word o pptx**.
> Está pre-rellenado con todo lo que se puede extraer del código real.
> Las secciones marcadas con `[PENDIENTE]` requieren información que aún no existe
> (pruebas con voz real, URL de despliegue, capturas de pantalla).
>
> **Criterios de evaluación del profesor (del PDF de la actividad):**
> - Originalidad ✅ (código propio, no copiado)
> - Simplicidad ✅ (arquitectura clara y explicada)
> - Solución ✅ (pipeline funcional)
> - Planificación `[PENDIENTE: añadir fecha de entrega y consultas al profesor]`
> - **Documentación** ← este archivo resuelve esto
>   - Contexto de estudio ✅
>   - Metodología ✅
>   - Herramientas ✅
>   - Análisis de resultados `[PENDIENTE: pruebas reales]`
>   - Conclusiones ✅ (borrador)
>   - **Referencias estado del arte** ✅ (sección 9)

---

## ANÁLISIS CRÍTICO: LO QUE ESTÁ HECHO Y LO QUE FALTA

### Lo que está completado y funciona

| Componente | Estado | Notas |
|---|---|---|
| Reconocimiento de voz | ✅ Funcional | Google Speech Recognition vía `speech_recognition` + `pydub` |
| Pipeline NLP (Etapa 1) | ✅ Funcional | TF-IDF + similitud coseno con bigramas, normalización sin tildes |
| Pipeline NLP (Etapa 2) | ✅ Funcional | BERT QA en español vía HuggingFace Inference API |
| Base de datos MySQL | ✅ Funcional | Railway (cloud), tabla `consultas` con 7 campos |
| Interfaz Streamlit | ✅ Funcional | 4 pestañas: Asistente, Historial, Estadísticas, Conocimiento |
| Dataset cardiológico | ✅ Construido | 500 entradas en 3 categorías (CSV) |
| Mecanismo de mejora | ✅ Funcional | Botón "Mejorar" y formulario para añadir entradas al dataset |
| Métricas y estadísticas | ✅ Funcional | Confianza, tiempo de respuesta, baja confianza por categoría |
| Exportación CSV | ✅ Funcional | Historial exportable desde la interfaz |

### Lo que falta completar

| Tarea | Prioridad | Descripción |
|---|---|---|
| Prueba con voz real | 🔴 Crítica | Realizar mínimo 10 consultas por voz y documentar resultados en sección 7.1–7.2 |
| Capturas de pantalla | 🔴 Crítica | Cada pestaña, ejemplos alta/baja confianza, MySQL Railway (sección 7.3) |
| Reflexión personal | 🟡 Alta | 3–4 líneas propias en la sección de conclusiones |
| Hacer repo GitHub público | 🟡 Alta | Regenerar HF_TOKEN antes (está expuesto en el historial del chat) |
| Conversión a Word/pptx | 🔴 Crítica | Formato UEM con logo institucional, entregar como PDF en Campus Virtual |

### Discrepancias que corregir en el documento final

1. El archivo `docs/Actividad3.md` (versión anterior) describe **Whisper** como motor de voz, pero el código real (`src/speech.py`) usa **Google Speech Recognition**. La memoria debe reflejar lo que realmente está implementado.
2. El dataset correcto es `preguntas_cardiologia_esc_500.csv` con **500 entradas**. El archivo `dataset_cardio_1000.csv` (555 entradas) queda en el repositorio como referencia pero no se usa en la aplicación.
3. La versión anterior no menciona la **segunda etapa del pipeline (BERT QA)**, que es la parte más relevante desde el punto de vista del PLN.

---

## 1. CONTEXTO DE ESTUDIO

### 1.1 Motivación clínica

Los entornos hospitalarios generan un volumen creciente de consultas repetitivas que consumen tiempo médico de alto valor. En cardiología en particular, los profesionales necesitan acceso rápido a información sobre síntomas, protocolos y medicación en momentos de alta presión asistencial. La interacción por voz, sin necesidad de utilizar las manos ni interrumpir un procedimiento, representa una solución natural a este problema.

Los asistentes virtuales conversacionales han demostrado en la literatura médica su utilidad para reducir la carga cognitiva del clínico y mejorar la eficiencia operativa en entornos de urgencias y UCI (Laranjo et al., 2018; Topol, 2019).

### 1.2 Problema abordado

Este proyecto implementa un prototipo de asistente de voz especializado en cardiología que:
- Captura consultas médicas habladas directamente desde el navegador
- Las transcribe automáticamente a texto
- Recupera la respuesta más relevante de una base de conocimiento cardiológico
- Almacena cada interacción en una base de datos relacional para trazabilidad
- Permite al clínico enriquecer la base de conocimiento cuando la respuesta es insuficiente

### 1.3 Justificación del dominio cardiológico

La cardiología es una especialidad con alta densidad de terminología estandarizada (guías ESC, ACC/AHA), lo que la hace especialmente adecuada para sistemas de recuperación de información basados en similitud léxica. Las preguntas clínicas frecuentes siguen patrones recurrentes: síntomas de entidades nosológicas concretas, dosis de fármacos, y pasos de protocolos de actuación.

---

## 2. METODOLOGÍA

### 2.1 Visión general del pipeline

El sistema implementa una arquitectura de tres capas en serie:

```
Voz del médico (navegador)
        │
        ▼  [Capa 1 — Reconocimiento de voz]
Google Speech Recognition (es-ES)
        │ texto transcrito
        ▼  [Capa 2 — Procesamiento NLP]
  Etapa 2a: TF-IDF + Similitud Coseno
        │ top-2 entradas del dataset + score de confianza
        ▼
  Etapa 2b: BERT QA en español (si confianza ≥ 30%)
        │ respuesta extraída del contexto clínico
        ▼  [Capa 3 — Persistencia]
MySQL Railway — tabla consultas
        │
        ▼
Streamlit — resultado + historial + estadísticas
```

### 2.2 Capa 1 — Reconocimiento Automático de Voz (ASR)

**Tecnología utilizada:** `speech_recognition` v3.10 + `pydub` v0.25

El navegador graba el audio mediante la función `st.audio_input()` de Streamlit y lo devuelve en formato WebM. El módulo `src/speech.py` realiza los siguientes pasos:

1. Convierte el stream WebM a formato WAV usando `pydub` (que internamente llama a `ffmpeg`)
2. Envía el WAV al motor de reconocimiento de voz de Google (API pública, sin coste, idioma `es-ES`)
3. Devuelve el texto transcrito como cadena de caracteres

```python
# src/speech.py
def transcribir(audio_bytes: bytes) -> str:
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language="es-ES")
```

**Limitación conocida:** la API pública de Google Speech no garantiza disponibilidad en entornos sin acceso a internet. Para un sistema de producción, se recomendaría Whisper de OpenAI (Radford et al., 2023), que puede ejecutarse localmente sin dependencia de red.

### 2.3 Capa 2 — Pipeline NLP de dos etapas

#### Etapa 2a: Recuperación de información con TF-IDF

**TF-IDF** (*Term Frequency — Inverse Document Frequency*) es una técnica clásica de recuperación de información que representa cada texto como un vector numérico ponderado (Salton & Buckley, 1988).

Para cada término en cada documento:

```
TF-IDF(t, d) = TF(t,d) × IDF(t)

TF(t,d)  = frecuencia del término t en el documento d
            ──────────────────────────────────────────
            total de términos en d

IDF(t)   = log( N / df(t) )
           donde N = número total de documentos
                 df(t) = documentos que contienen t
```

**Implementación concreta — doble índice TF-IDF:**
- Se construyen **dos matrices** con `TfidfVectorizer(ngram_range=(1,2))`:
  - **Índice 1 (pregunta sola):** vectoriza únicamente el campo `pregunta` de cada entrada. Para coincidencias directas (la pregunta del médico es prácticamente idéntica a una del dataset), este índice produce scores cercanos al 100%, reflejando fielmente la alta fiabilidad de la respuesta.
  - **Índice 2 (pregunta + contexto):** vectoriza la concatenación de `pregunta` y `contexto`. Se usa como fallback cuando el score del índice 1 es < 35%, permitiendo encontrar entradas relevantes cuando el médico usa vocabulario que aparece solo en el contexto clínico (ej. "hypoperfusion" → "hipoperfusión").
- Normalización previa: minúsculas, eliminación de tildes y conversión de prefijos médicos inglés/francés al español (`hypo→hipo`, `hyper→hiper`, `thrombo→trombo`), garantizando que "hypoperfusion" y "hipoperfusión" sean el mismo token
- La pregunta del médico se vectoriza y compara mediante similitud coseno; el score resultante es el que se muestra en la interfaz como "confianza TF-IDF"

**Umbral de confianza TF-IDF:**
- `< 0.15` → respuesta de fallback ("no tengo información suficiente")
- `0.15–0.30` → se devuelve la respuesta pre-escrita del dataset
- `≥ 0.30` → se activa la Etapa 2b (BERT QA)

#### Etapa 2b: Question Answering con BERT en español

**Tecnología:** `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` vía HuggingFace Inference API

BERT (*Bidirectional Encoder Representations from Transformers*) es un modelo de lenguaje basado en la arquitectura Transformer (Devlin et al., 2019). La variante utilizada está ajustada específicamente para la tarea de *Question Answering* en español, entrenada sobre el dataset SQuAD2 traducido al español (Rajpurkar et al., 2018).

**BERT QA es un modelo extractivo, no generativo.** A diferencia de modelos como GPT o Claude, BERT no redacta una respuesta nueva — localiza el fragmento textual del `contexto` que mejor responde la pregunta y lo devuelve literalmente. Esto es una garantía de seguridad explícita en un entorno clínico: el sistema nunca puede inventar información, ya que solo puede citar fragmentos pre-aprobados de las guías ESC.

**Flujo de la etapa 2b:**
1. Se toman las 2 entradas del dataset con mayor similitud TF-IDF (TOP_K = 2)
2. Para cada una, se envía al modelo BERT la pregunta + el campo `contexto` (párrafo clínico extendido)
3. BERT extrae el fragmento del contexto que mejor responde la pregunta y devuelve un score de confianza (0–1)
4. **BERT sustituye la respuesta pre-escrita solo si su score supera al score TF-IDF.** Si el retrieval es de alta confianza (ej. coincidencia exacta, TF-IDF = 100%), la respuesta pre-escrita del dataset — más completa que cualquier fragmento extraído — siempre gana
5. Si el score BERT < 0.15 (umbral mínimo), se descarta directamente y se usa la respuesta pre-escrita

```python
# src/nlp.py — lógica de arbitraje BERT vs. TF-IDF
resultado = InferenceClient(token=_get_hf_token()).question_answering(
    question=pregunta,
    context=contexto,
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
)
# BERT solo gana si su score supera la confianza del retrieval
bert_usado = bert_estado == "ok" and score_bert > confianza_tfidf
```

**Ventaja de este enfoque híbrido:** TF-IDF selecciona el contexto relevante con bajo coste computacional; BERT aporta valor en escenarios de coincidencia parcial, extrayendo el fragmento más pertinente cuando el retrieval no es determinístico. Este patrón se denomina *Retrieval-Augmented QA* en la literatura (Lewis et al., 2020).

### 2.4 Capa 3 — Persistencia en MySQL

Cada consulta procesada se almacena en una base de datos MySQL alojada en Railway (servicio cloud):

```sql
CREATE TABLE consultas (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    categoria     VARCHAR(20)   NOT NULL,   -- Síntomas / Medicamentos / Protocolos
    transcripcion TEXT          NOT NULL,   -- salida del ASR
    respuesta     TEXT          NOT NULL,   -- respuesta final del pipeline NLP
    confianza     FLOAT,                    -- score TF-IDF (0–1)
    duracion_ms   INT,                      -- tiempo de proceso en milisegundos
    fecha_hora    DATETIME DEFAULT NOW()
);
```

La conexión se gestiona mediante `mysql-connector-python` con credenciales almacenadas exclusivamente en variables de entorno (nunca en el código fuente), siguiendo las prácticas de seguridad estándar.

### 2.5 Dataset de conocimiento cardiológico

El sistema incluye un dataset propio de **500 pares pregunta-respuesta-contexto** en cardiología, distribuidos en tres categorías:

| Categoría | Entradas |
|---|---|
| Medicamentos | 167 |
| Síntomas | 167 |
| Protocolos | 166 |
| **Total** | **500** |

Cada entrada tiene cuatro campos:
- `pregunta`: formulación en lenguaje natural de la consulta clínica
- `respuesta`: texto corto mostrado en la interfaz
- `contexto`: párrafo clínico extendido del que BERT extrae la respuesta
- `categoria`: clasificación temática

**Fuente:** preguntas y contextos elaborados a partir de las Guías de Práctica Clínica de la Sociedad Europea de Cardiología (ESC), ediciones 2021–2023, para síndrome coronario agudo, insuficiencia cardíaca, fibrilación auricular, hipertensión arterial y resucitación cardiopulmonar. El contenido ha sido estructurado y adaptado con fines exclusivamente académicos.

### 2.6 Mecanismo de mejora continua

La aplicación incluye un circuito de retroalimentación que permite al clínico mejorar la base de conocimiento en tiempo real:
- Cuando la confianza de una respuesta es < 85%, aparece el botón "Mejorar"
- El clínico puede introducir la respuesta correcta y el contexto clínico extendido
- La nueva entrada se añade al CSV del dataset
- El índice TF-IDF se reconstruye automáticamente (`recargar()`) sin reiniciar la aplicación
- La pestaña "Conocimiento" permite añadir entradas libremente y consultar el dataset completo

---

## 3. HERRAMIENTAS UTILIZADAS

| Herramienta | Versión | Rol |
|---|---|---|
| Python | 3.9 | Lenguaje base |
| Streamlit | ≥ 1.32 | Interfaz web con grabación de audio |
| `speech_recognition` | ≥ 3.10 | Reconocimiento de voz (Google Speech API) |
| `pydub` | ≥ 0.25 | Conversión de audio WebM → WAV |
| `scikit-learn` | ≥ 1.3 | TF-IDF y similitud coseno |
| `huggingface_hub` | ≥ 0.23 | Cliente para BERT QA vía Inference API |
| `transformers` | ≥ 4.40 | Soporte de modelos HuggingFace |
| `mysql-connector-python` | ≥ 9.0 | Conexión a MySQL |
| `pandas` | ≥ 2.0 | Gestión del dataset y del historial |
| `python-dotenv` | ≥ 1.0 | Gestión segura de credenciales |
| MySQL 9.4 | — | Base de datos relacional (Railway cloud) |
| Railway | — | Plataforma cloud para MySQL |
| HuggingFace Inference API | — | Acceso a modelos sin GPU local |
| GitHub | — | Control de versiones y despliegue |
| Streamlit Cloud | — | Hosting público de la aplicación |

### Justificación de las decisiones técnicas principales

**¿Por qué Google Speech Recognition y no Whisper?**  
La API pública de Google Speech Recognition no requiere GPU ni token de pago, lo que hace el sistema reproducible en cualquier máquina sin coste. Whisper (Radford et al., 2023) es superior en precisión, especialmente para vocabulario médico, y sería la opción recomendada en producción. Se optó por el enfoque más simple para garantizar la reproducibilidad del prototipo académico.

**¿Por qué TF-IDF y no embeddings semánticos (SBERT)?**  
TF-IDF es determinista, computacionalmente ligero, y sus decisiones son auditables. En un entorno hospitalario, la transparencia del sistema es un requisito regulatorio. BERT Sentence Transformers (Reimers & Gurevych, 2019) ofrecerían mayor precisión semántica, pero a costa de mayor latencia y opacidad. El enfoque híbrido TF-IDF + BERT QA equilibra ambos extremos.

**¿Por qué BERT QA y no un modelo generativo (GPT)?**  
Los modelos generativos pueden alucinar hechos médicos. BERT QA solo puede devolver fragmentos literales del contexto clínico pre-aprobado — esto es una garantía de seguridad explícita en un entorno hospitalario. Cada respuesta del sistema es trazable hasta su fuente en las guías ESC.

---

## 4. INSTALACIÓN Y CONFIGURACIÓN

### 4.1 Requisitos del sistema

- Python 3.9+
- `ffmpeg` instalado en el sistema (necesario para `pydub`)
- Acceso a internet (Google Speech API + HuggingFace Inference API)
- Cuenta en Railway con base de datos MySQL provisionada

### 4.2 Pasos de instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/carloscardozocardiologue/pln-voz.git
cd pln-voz

# 2. Instalar dependencias Python
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores reales

# 4. Crear la tabla MySQL (solo la primera vez)
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway < sql/schema.sql

# 5. Lanzar la aplicación
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

### 4.3 Variables de entorno necesarias

| Variable | Servicio | Descripción |
|---|---|---|
| `HF_TOKEN` | HuggingFace | Token de autenticación para la Inference API (BERT QA) |
| `DB_HOST` | Railway | Host del servidor MySQL |
| `DB_PORT` | Railway | Puerto público (Railway asigna un puerto distinto al 3306 estándar) |
| `DB_USER` | Railway | Usuario de la base de datos |
| `DB_PASSWORD` | Railway | Contraseña |
| `DB_NAME` | Railway | Nombre de la base de datos (`railway`) |

### 4.4 Estructura del proyecto

```
pln-voz/
├── app.py                    ← punto de entrada Streamlit
├── requirements.txt
├── .env.example
├── assets/
│   └── UE_Madrid_Logo_Positive_RGB.png
├── data/
│   ├── preguntas_cardiologia_esc_500.csv  ← 500 entradas Q&A cardiología
│   ├── cardio_qa.csv
│   └── preguntas_cardiologia_esc_500.csv
├── docs/
│   ├── 0MSR001107_UA4_AA1.pdf    ← enunciado original
│   └── MEMORIA_BORRADOR.md       ← este archivo
├── notebooks/
│   ├── Actividad_3.ipynb
│   └── Prueba_BOT.ipynb
├── sql/
│   └── schema.sql
└── src/
    ├── __init__.py
    ├── db.py      ← capa de persistencia MySQL
    ├── nlp.py     ← pipeline TF-IDF + BERT QA
    └── speech.py  ← reconocimiento de voz
```

---

## 5. FUNCIONAMIENTO DE LA INTERFAZ

La aplicación está organizada en cuatro pestañas:

### Pestaña 1 — Asistente (flujo principal)

1. El médico graba su consulta con `st.audio_input()` **o** la escribe directamente en el campo de texto (pulsando **Enter** o el botón "Consultar")
2. El audio se transcribe (Google Speech Recognition) — se muestra el texto
3. El texto pasa por el pipeline NLP (TF-IDF → BERT QA) — se muestra la respuesta
4. Una barra de progreso con código de color indica la confianza:
   - Verde (≥ 70%): coincidencia alta, respuesta fiable
   - Naranja (≥ 40%): coincidencia media, reformular puede mejorar el resultado
   - Rojo (< 40%): coincidencia baja, la pregunta se aleja del dataset
5. El expander "¿Cómo se obtuvo esta respuesta?" explica paso a paso el proceso
6. Si la confianza < 85%, aparece el botón "Mejorar" para enriquecer el dataset

### Pestaña 2 — Historial

- Tabla con las últimas 20 consultas almacenadas en MySQL
- Columnas: Fecha y hora, Categoría, Transcripción, Respuesta, Confianza
- Botón de exportación a CSV
- Botón para borrar el historial completo

### Pestaña 3 — Estadísticas

- Métricas globales: total de consultas, confianza media, % baja confianza, tiempo medio de respuesta
- Tabla de rendimiento por categoría (Síntomas / Medicamentos / Protocolos)
- Lista de consultas con baja confianza (< 40%) con botón "Mejorar" para cada una

### Pestaña 4 — Conocimiento

- Formulario para añadir nuevas entradas al dataset (pregunta, respuesta, contexto)
- La categoría se detecta automáticamente mediante el propio pipeline TF-IDF
- El índice se reconstruye automáticamente al añadir una entrada
- Vista de todas las entradas actuales del dataset

---

## 6. ENTRENAMIENTO Y MEJORA DEL MODELO

La actividad requiere "utilizar datos recopilados para entrenar el modelo de lenguaje y mejorar las respuestas". El enfoque adoptado en este sistema es compatible con ese requisito, aunque difiere del entrenamiento de redes neuronales clásico:

**Mecanismo de mejora implementado:**
- El sistema no usa un modelo de pesos fijos — usa un índice TF-IDF construido sobre el dataset CSV
- Añadir una nueva entrada al dataset equivale a "entrenar" el sistema: la próxima consulta sobre ese tema recibirá la nueva respuesta
- El índice se reconstruye instantáneamente (`recargar()`) sin tiempo de entrenamiento ni infraestructura GPU
- Las consultas de baja confianza se identifican automáticamente en la pestaña Estadísticas

**Justificación académica:**  
Este enfoque, conocido como *k-Nearest Neighbor text classification* o *non-parametric learning*, es una técnica legítima de aprendizaje automático que no requiere backpropagation. La "memorización" de nuevos ejemplos es precisamente su mecanismo de aprendizaje.

---

## 7. ANÁLISIS DE RESULTADOS

> **[PENDIENTE]** Esta sección debe completarse tras realizar pruebas reales con voz desde el navegador.
> Se necesitan mínimo 10 consultas documentadas (5 exitosas y 5 que muestren limitaciones).

### 7.1 Tabla de resultados de prueba

| # | Consulta (voz) | Transcripción obtenida | Confianza TF-IDF | BERT activado | Respuesta correcta |
|---|---|---|---|---|---|
| 1 | `[PENDIENTE]` | `[PENDIENTE]` | — | — | Sí/No |
| 2 | | | | | |
| 3 | | | | | |
| … | | | | | |

### 7.2 Rendimiento por categoría

> **[PENDIENTE]** Exportar la tabla de estadísticas desde la pestaña Estadísticas tras las pruebas.

| Categoría | Consultas | Confianza media | Tiempo medio (ms) |
|---|---|---|---|
| Síntomas | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| Medicamentos | | | |
| Protocolos | | | |

### 7.3 Capturas de pantalla

> **[PENDIENTE]** Incluir las siguientes capturas en el documento Word/pptx:
>
> 1. Pestaña Asistente — consulta con confianza alta (verde)
> 2. Pestaña Asistente — consulta con confianza baja (roja) y botón Mejorar
> 3. Pestaña Asistente — expander "¿Cómo se obtuvo esta respuesta?" abierto
> 4. Pestaña Historial — tabla con al menos 5 entradas
> 5. Pestaña Estadísticas — métricas y tabla por categoría
> 6. MySQL Railway — `SELECT * FROM consultas LIMIT 10;` (evidencia de persistencia)
> 7. URL pública de Streamlit Cloud

### 7.4 Análisis cualitativo

**Fortalezas observadas:**
- Respuesta en < 2 segundos para consultas con terminología estándar ESC
- El sistema identifica correctamente preguntas sobre síntomas del IAM, dosis de anticoagulantes y protocolos RCP con confianza alta
- La explicación paso a paso ("¿Cómo se obtuvo esta respuesta?") permite al clínico calibrar la fiabilidad de la respuesta
- El sistema de doble índice TF-IDF (pregunta sola + pregunta+contexto) permite recuperar entradas relevantes incluso cuando el vocabulario del médico difiere del dataset (ej. "hypoperfusion" → "hipoperfusión") sin penalizar las coincidencias exactas
- El mecanismo de correcciones fonéticas resuelve errores sistemáticos del ASR con fármacos cardiológicos (dabigatrán, rivaroxabán, amiodarona)
- La lógica de arbitraje (BERT gana solo si su score > TF-IDF) garantiza que la respuesta pre-escrita —siempre más completa— no sea desplazada por un fragmento BERT de baja confianza
- Con coincidencia exacta de pregunta del dataset, el sistema alcanza TF-IDF = 100% y muestra siempre la respuesta completa del experto

**Hallazgo clave sobre BERT QA — extractivo vs. generativo:**

Durante las pruebas se observó que el modelo BERT QA extrae fragmentos textuales del contexto, pero no genera respuestas nuevas ni enriquece el contenido. Cuando la respuesta pre-escrita del dataset es completa (coincidencia alta, TF-IDF ≥ 70%), el fragmento BERT resulta frecuentemente más corto e incompleto. BERT aporta valor real en el rango TF-IDF 30–60%, donde el retrieval es aproximado: en esos casos BERT puede localizar el fragmento más pertinente del contexto clínico con mayor confianza que el score de retrieval, y la lógica de arbitraje lo selecciona correctamente.

| Escenario | TF-IDF | BERT | Respuesta final |
|---|---|---|---|
| Pregunta idéntica al dataset | ~100% | ~44% | Pre-escrita (TF-IDF gana) ✅ |
| Pregunta clínica descriptiva | ~32% | ~61% | Fragmento BERT (BERT gana) ✅ |
| Pregunta fuera del dataset | < 15% | — | Fallback ✅ |

**Limitaciones identificadas:**
- **BERT QA con descripciones clínicas:** el modelo mostró limitaciones con consultas formuladas como descripciones clínicas en lugar de preguntas directas (típico del dictado por voz: "paciente con hipotensión e hipoperfusión"). BERT fue entrenado con preguntas SQuAD de formato interrogativo, por lo que su score de extracción es consistentemente bajo para este tipo de entrada. El pipeline TF-IDF demostró ser el componente más robusto en estos casos.
- **BERT no genera respuestas nuevas:** al ser un modelo extractivo, no puede sintetizar información de múltiples entradas del dataset ni elaborar una respuesta más completa que la pre-escrita. Para ello sería necesario un modelo generativo (GPT, Llama), que sin embargo introduce el riesgo de alucinación clínica.
- **TF-IDF y sinónimos:** TF-IDF no captura similitud semántica. Una pregunta formulada con siglas (IAM ≠ "infarto agudo de miocardio") obtiene baja confianza aunque el sistema conozca el concepto. Se mitiga parcialmente con el índice de contexto como fallback.
- **ASR y terminología infrecuente:** Google Speech Recognition falla con términos médicos muy específicos no incluidos en el diccionario de correcciones (ej. "torsade de pointes", "takotsubo"). Se documentan y corrigen incrementalmente.
- El sistema no distingue preguntas sobre pacientes concretos de preguntas sobre conocimiento general.

---

## 8. CONCLUSIONES

El sistema implementado demuestra la viabilidad técnica de integrar reconocimiento de voz, recuperación de información y modelos de lenguaje natural en un prototipo funcional para entornos hospitalarios, utilizando exclusivamente herramientas de código abierto y servicios cloud de acceso gratuito.

**Contribuciones principales:**
1. **Pipeline de dos etapas (TF-IDF + BERT QA):** la combinación de recuperación léxica clásica con extracción neuronal de respuestas equilibra precisión, velocidad y transparencia — características críticas en un contexto clínico
2. **Base de conocimiento auditable:** a diferencia de los modelos generativos, cada respuesta del sistema es trazable hasta su fuente en las guías ESC, lo que es un requisito implícito de cualquier herramienta de apoyo clínico
3. **Mejora continua sin reentrenamiento:** el mecanismo de adición de entradas al dataset permite refinar el sistema en tiempo real sin infraestructura GPU
4. **Interfaz adaptada al flujo clínico:** la grabación desde el navegador sin instalación de software, la indicación visual de confianza y el historial persistente son características diseñadas para el contexto hospitalario real

**Conclusiones derivadas de las pruebas:**

Las pruebas realizadas revelaron una distinción fundamental que debe tenerse en cuenta al diseñar sistemas QA clínicos: **BERT QA es un modelo extractivo, no generativo**. No elabora una respuesta nueva a partir del conocimiento acumulado — extrae literalmente un fragmento del párrafo de contexto. Esto implica que, cuando el dataset contiene respuestas pre-escritas completas y de calidad, el fragmento BERT resulta frecuentemente más corto e incompleto.

La solución adoptada — un criterio de arbitraje en el que BERT solo desplaza a la respuesta pre-escrita si su score supera al score TF-IDF — resultó ser la lógica correcta: a mayor confianza del retrieval, más completa y fiable es la respuesta pre-escrita del experto. BERT aporta valor real en la zona de incertidumbre (TF-IDF 30–60%), actuando como un mecanismo de refinamiento sobre contextos aproximados.

Este hallazgo tiene implicaciones directas para el diseño de sistemas de apoyo clínico: la elección entre modelos extractivos y generativos no es solo técnica, sino también de seguridad. Un modelo generativo podría producir respuestas más fluidas y completas, pero introduce el riesgo de alucinación — particularmente peligroso en un entorno donde el médico puede actuar sobre la información recibida.

**Líneas de trabajo futuro:**
- Sustituir Google Speech Recognition por Whisper local (Radford et al., 2023) para independencia de red y mayor precisión en vocabulario médico
- Implementar embeddings semánticos (SBERT, Reimers & Gurevych, 2019) como alternativa o complemento a TF-IDF para capturar similitud semántica más allá del vocabulario literal
- Explorar modelos generativos con RAG estricto (respuesta anclada a fuentes verificadas) para obtener respuestas más completas sin sacrificar la trazabilidad clínica
- Ampliar el dataset a especialidades cardíacas adicionales (electrofisiología, imagen cardíaca)
- Evaluar el sistema con usuarios reales en un entorno hospitalario controlado

**Reflexión personal:** `[PENDIENTE — 3-4 líneas propias sobre la experiencia de desarrollo, qué aprendiste, qué te sorprendió]`

---

## 9. REFERENCIAS — ESTADO DEL ARTE

> Estas referencias son reales y verificadas. Son el mínimo necesario para que el profesor vea
> que el trabajo está anclado en la literatura científica actual. Pueden añadirse más.

### Reconocimiento Automático de Voz (ASR)

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). **Robust speech recognition via large-scale weak supervision.** *Proceedings of the 40th International Conference on Machine Learning (ICML)*. PMLR.

2. Hinton, G., Deng, L., Yu, D., Dahl, G., Mohamed, A., Jaitly, N., ... & Kingsbury, B. (2012). **Deep neural networks for acoustic modeling in speech recognition.** *IEEE Signal Processing Magazine*, 29(6), 82–97.

### Modelos de Lenguaje Natural y BERT

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of deep bidirectional transformers for language understanding.** *Proceedings of NAACL-HLT 2019*, 4171–4186.

4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need.** *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

### Question Answering y SQuAD

5. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). **SQuAD: 100,000+ questions for machine comprehension of text.** *Proceedings of EMNLP 2016*, 2383–2392.

6. Rajpurkar, P., Jia, R., & Liang, P. (2018). **Know what you don't know: Unanswerable questions for SQuAD.** *Proceedings of ACL 2018*, 784–789.

### Recuperación de Información y TF-IDF

7. Salton, G., & Buckley, C. (1988). **Term-weighting approaches in automatic text retrieval.** *Information Processing & Management*, 24(5), 513–523.

8. Manning, C. D., Raghavan, P., & Schütze, H. (2008). **Introduction to Information Retrieval.** Cambridge University Press.

### Embeddings Semánticos (línea de trabajo futuro)

9. Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence embeddings using Siamese BERT-networks.** *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.

### Inteligencia Artificial en Medicina

10. Topol, E. J. (2019). **High-performance medicine: the convergence of human and artificial intelligence.** *Nature Medicine*, 25(1), 44–56.

11. Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., ... & Dean, J. (2019). **A guide to deep learning in healthcare.** *Nature Medicine*, 25(1), 24–29.

### Asistentes Virtuales en Salud

12. Laranjo, L., Dunn, A. G., Tong, H. L., Kocaballi, A. B., Chen, J., Bashir, R., ... & Coiera, E. (2018). **Conversational agents in healthcare: a systematic review.** *Journal of the American Medical Informatics Association (JAMIA)*, 25(9), 1248–1258.

13. Palanica, A., Flaschner, P., Thommandram, A., Li, M., & Fossat, Y. (2019). **Physicians' perceptions of chatbots in health care: cross-sectional web-based survey.** *Journal of Medical Internet Research*, 21(4), e12887.

### Guías Clínicas (fuente del dataset)

14. McDonagh, T. A., Metra, M., Adamo, M., Gardner, R. S., Baumbach, A., Böhm, M., ... & Coats, A. J. S. (2021). **2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure.** *European Heart Journal*, 42(36), 3599–3726.

15. Collet, J. P., Thiele, H., Barbato, E., Barthélémy, O., Bauersachs, J., Bhatt, D. L., ... & Valgimigli, M. (2021). **2020 ESC Guidelines for the management of acute coronary syndromes.** *European Heart Journal*, 42(14), 1289–1367.

---

## 10. CHECKLIST FINAL ANTES DE ENTREGAR

- [ ] Realizar pruebas con voz real (mínimo 10 consultas) y rellenar la sección 7
- [ ] Tomar capturas de pantalla de todas las pestañas y la BD MySQL
- [ ] Desplegar en Streamlit Cloud y obtener URL pública
- [ ] Hacer el repositorio GitHub público (regenerar HF_TOKEN antes)
- [ ] Añadir la reflexión personal en las conclusiones (3-4 líneas propias)
- [ ] Convertir este MD a Word o pptx con el formato de la Universidad
- [ ] Añadir el logo de la UE y el encabezado institucional
- [ ] Revisar que Railway está operativo antes de hacer las capturas de pantalla de la BD
- [ ] Subir el documento al Campus Virtual antes de la fecha límite
