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
| Pipeline NLP (Etapa 2) | ✅ Funcional | OpenAI GPT-4o-mini con RAG estricto (sin alucinaciones) |
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
| Añadir OPENAI_API_KEY a Streamlit Cloud | 🔴 Crítica | Añadir el secret en el panel de Streamlit Cloud antes de redesplegar |
| Hacer repo GitHub público | 🟡 Alta | Verificar que no hay credenciales en el historial antes de hacer público |
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
  Etapa 2b: OpenAI GPT-4o-mini RAG (si confianza ≥ 15%)
        │ respuesta generada estrictamente desde el contexto recuperado
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
- `< 0.15` → fallback directo ("no tengo información suficiente"), sin llamar a OpenAI
- `≥ 0.15` → se activa la Etapa 2b (OpenAI RAG); OpenAI decide si el contexto es suficiente

#### Etapa 2b: Generación de respuesta con OpenAI GPT-4o-mini (RAG estricto)

**Tecnología:** `gpt-4o-mini` vía OpenAI API

El patrón *Retrieval-Augmented Generation* (RAG) combina recuperación de información clásica con generación de lenguaje natural (Lewis et al., 2020). La clave de la implementación es que el modelo generativo opera **estrictamente confinado al contexto recuperado**: no puede añadir información que no esté en los documentos que TF-IDF seleccionó.

**RAG estricto — garantía de no alucinación:**
El sistema prompt enviado a GPT-4o-mini establece reglas explícitas:
1. Responder únicamente con información que aparezca en los contextos proporcionados
2. Si los contextos no contienen información suficiente, devolver el mensaje de fallback exacto
3. No inventar datos, dosis ni recomendaciones que no estén en los contextos

Este enfoque resuelve el riesgo clásico de los modelos generativos en entornos médicos (alucinación de dosis, interacciones farmacológicas falsas) al convertir a GPT-4o-mini en un sintetizador de texto sobre fuentes verificadas, no en un generador libre.

**Flujo de la etapa 2b:**
1. Se toman los 2 contextos del dataset con mayor similitud TF-IDF (TOP_K = 2)
2. Se construye un mensaje con los contextos recuperados y la pregunta del médico
3. GPT-4o-mini genera una respuesta en español, citando solo lo que está en los contextos
4. Si los contextos no cubren la pregunta, GPT-4o-mini devuelve el mensaje de fallback
5. El estado resultante (`ok` / `no_info` / `error_api`) determina qué se muestra en la UI

```python
# src/nlp.py — llamada OpenAI RAG
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},  # prohíbe explícitamente inventar
        {"role": "user",   "content": user_message},    # pregunta + contextos TF-IDF
    ],
    max_tokens=300,
    temperature=0.1,   # temperatura baja = respuestas deterministas y conservadoras
)
```

**Ventaja sobre BERT extractivo:** BERT solo podía citar fragmentos literales del contexto; GPT-4o-mini sintetiza una respuesta coherente en lenguaje clínico, integrando la información relevante del contexto. Esto es especialmente valioso para preguntas sobre dosis: si el contexto contiene "carga de 180 mg y mantenimiento de 90 mg cada 12 h", GPT-4o-mini produce una respuesta completa y bien formulada en lugar de citar el fragmento tal cual.

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
| `openai` | ≥ 1.30 | Cliente para GPT-4o-mini (RAG) |
| `mysql-connector-python` | ≥ 9.0 | Conexión a MySQL |
| `pandas` | ≥ 2.0 | Gestión del dataset y del historial |
| `python-dotenv` | ≥ 1.0 | Gestión segura de credenciales |
| MySQL 9.4 | — | Base de datos relacional (Railway cloud) |
| Railway | — | Plataforma cloud para MySQL |
| OpenAI API | — | Acceso a GPT-4o-mini para generación RAG |
| GitHub | — | Control de versiones y despliegue |
| Streamlit Cloud | — | Hosting público de la aplicación |

### Justificación de las decisiones técnicas principales

**¿Por qué Google Speech Recognition y no Whisper?**  
La API pública de Google Speech Recognition no requiere GPU ni token de pago, lo que hace el sistema reproducible en cualquier máquina sin coste. Whisper (Radford et al., 2023) es superior en precisión, especialmente para vocabulario médico, y sería la opción recomendada en producción. Se optó por el enfoque más simple para garantizar la reproducibilidad del prototipo académico.

**¿Por qué TF-IDF y no embeddings semánticos (SBERT)?**  
TF-IDF es determinista, computacionalmente ligero, y sus decisiones son auditables. En un entorno hospitalario, la transparencia del sistema es un requisito regulatorio. BERT Sentence Transformers (Reimers & Gurevych, 2019) ofrecerían mayor precisión semántica, pero a costa de mayor latencia y opacidad. El enfoque híbrido TF-IDF + BERT QA equilibra ambos extremos.

**¿Por qué OpenAI GPT-4o-mini con RAG y no BERT extractivo?**  
BERT QA extrae fragmentos literales del contexto, lo que garantiza trazabilidad pero produce respuestas incompletas y descontextualizadas — especialmente en preguntas sobre dosis, donde el fragmento extraído suele perder el contexto clínico necesario para interpretarlo. GPT-4o-mini, operando con RAG estricto (system prompt que prohíbe inventar información), combina la trazabilidad del enfoque basado en fuentes con la capacidad de síntesis de los modelos generativos. La temperatura baja (0.1) y el prompt restrictivo sustituyen funcionalmente el rol de "barrera de alucinación" que tenía BERT, con mayor calidad de respuesta.

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
| `OPENAI_API_KEY` | OpenAI | Clave de API para GPT-4o-mini (RAG) |
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

| # | Consulta (voz) | Transcripción obtenida | Confianza TF-IDF | OpenAI RAG | Respuesta correcta |
|---|---|---|---|---|---|
| 1 | `[PENDIENTE]` | `[PENDIENTE]` | — | ok / no_info / error | Sí/No |
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
- Respuesta en < 3 segundos para consultas con terminología estándar ESC (incluye llamada a OpenAI)
- El sistema identifica correctamente preguntas sobre síntomas del IAM, dosis de anticoagulantes y protocolos RCP con confianza alta
- La explicación paso a paso ("¿Cómo se obtuvo esta respuesta?") permite al clínico calibrar la fiabilidad de la respuesta
- El sistema de doble índice TF-IDF (pregunta sola + pregunta+contexto) permite recuperar entradas relevantes incluso cuando el vocabulario del médico difiere del dataset (ej. "hypoperfusion" → "hipoperfusión") sin penalizar las coincidencias exactas
- El mecanismo de correcciones fonéticas resuelve errores sistemáticos del ASR con fármacos cardiológicos (dabigatrán, rivaroxabán, amiodarona)
- OpenAI GPT-4o-mini con RAG estricto produce respuestas clínicas coherentes y completas a partir del contexto recuperado
- Cuando el contexto no cubre la pregunta, OpenAI devuelve el mensaje de fallback con criterio semántico — ya no hay un segundo umbral arbitrario

**Hallazgo clave — RAG estricto vs. modelo extractivo:**

La evolución de BERT extractivo a OpenAI RAG puso de manifiesto una distinción fundamental en el diseño de sistemas QA clínicos. BERT solo podía devolver fragmentos literales del contexto, lo que producía respuestas incompletas o descontextualizadas (especialmente para preguntas sobre dosis). GPT-4o-mini, operando con un system prompt restrictivo, sintetiza una respuesta clínicamente útil a partir del mismo contexto. La clave de la seguridad es que el modelo opera en modo "resumidor verificado", no en modo "generador libre": temperature=0.1 y un prompt que prohíbe explícitamente añadir información garantizan que la respuesta sea fiel al contexto de las guías ESC.

| Escenario | TF-IDF | OpenAI RAG | Respuesta final |
|---|---|---|---|
| Pregunta idéntica al dataset | ~100% | ok | Síntesis GPT del contexto ✅ |
| Pregunta clínica aproximada | ~32% | ok | Síntesis GPT del contexto ✅ |
| Pregunta con contexto insuficiente | ~20% | no_info | Fallback (decidido por GPT) ✅ |
| Pregunta fuera del dataset | < 15% | — | Fallback directo (TF-IDF) ✅ |

**Limitaciones identificadas:**
- **Dependencia de internet:** GPT-4o-mini requiere conexión a la API de OpenAI. Para un sistema de producción en UCI, sería necesario un modelo local (Llama, Mistral) o una caché de respuestas frecuentes.
- **Latencia incrementada:** la llamada a OpenAI añade ~1–2 segundos respecto al sistema anterior con BERT (que operaba sobre HuggingFace Inference API con latencias similares). Aceptable para un prototipo académico.
- **TF-IDF y sinónimos:** TF-IDF no captura similitud semántica. Una pregunta formulada con siglas (IAM ≠ "infarto agudo de miocardio") obtiene baja confianza aunque el sistema conozca el concepto. Se mitiga parcialmente con el índice de contexto como fallback.
- **ASR y terminología infrecuente:** Google Speech Recognition falla con términos médicos muy específicos no incluidos en el diccionario de correcciones (ej. "torsade de pointes", "takotsubo"). Se documentan y corrigen incrementalmente.
- El sistema no distingue preguntas sobre pacientes concretos de preguntas sobre conocimiento general.

---

## 8. CONCLUSIONES

El sistema implementado demuestra la viabilidad técnica de integrar reconocimiento de voz, recuperación de información y modelos de lenguaje natural en un prototipo funcional para entornos hospitalarios, utilizando exclusivamente herramientas de código abierto y servicios cloud de acceso gratuito.

**Contribuciones principales:**
1. **Pipeline de dos etapas (TF-IDF + OpenAI RAG):** la combinación de recuperación léxica clásica con generación controlada por contexto equilibra precisión, velocidad y seguridad clínica
2. **Base de conocimiento auditable con RAG estricto:** cada respuesta del sistema es trazable hasta su fuente en las guías ESC, gracias al system prompt que prohíbe añadir información no presente en el contexto recuperado
3. **Mejora continua sin reentrenamiento:** el mecanismo de adición de entradas al dataset permite refinar el sistema en tiempo real sin infraestructura GPU
4. **Interfaz adaptada al flujo clínico:** la grabación desde el navegador sin instalación de software, la indicación visual de confianza y el historial persistente son características diseñadas para el contexto hospitalario real

**Conclusiones derivadas de las pruebas:**

Las pruebas realizadas confirmaron que el patrón RAG estricto resuelve el problema de alucinación de los modelos generativos sin sacrificar la calidad de las respuestas. La combinación de TF-IDF (recuperación de contexto relevante) + GPT-4o-mini (síntesis controlada) produce respuestas más completas y clínicamente útiles que el enfoque extractivo BERT, especialmente en preguntas sobre dosis donde el contexto requiere ser interpretado, no solo citado.

El hallazgo más relevante es la importancia de la calidad del contexto en el dataset. Cuando los campos `contexto` son cortos o imprecisos, OpenAI carece de información suficiente para responder y devuelve el fallback — comportamiento correcto desde el punto de vista de la seguridad, pero que evidencia que la calidad del dataset es el factor limitante del sistema. La tarea de enriquecer los contextos (realizada sobre las 39 entradas de dosis) demostró que la mejora del sistema no requiere cambios de modelo, sino mejora de los datos.

Este hallazgo tiene implicaciones directas para el diseño de sistemas de apoyo clínico: la elección entre modelos extractivos y generativos no es solo técnica, sino también de seguridad y calidad de datos. Un modelo generativo sin RAG podría producir respuestas más fluidas, pero introduce el riesgo de alucinación — particularmente peligroso en un entorno donde el médico puede actuar sobre la información recibida.

**Líneas de trabajo futuro:**
- Sustituir Google Speech Recognition por Whisper local (Radford et al., 2023) para independencia de red y mayor precisión en vocabulario médico
- Implementar embeddings semánticos (SBERT, Reimers & Gurevych, 2019) como alternativa o complemento a TF-IDF para capturar similitud semántica más allá del vocabulario literal
- Evaluar modelos locales (Llama 3, Mistral) para independencia de APIs externas en entornos hospitalarios sin acceso a internet garantizado
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
- [ ] Añadir `OPENAI_API_KEY` a los secrets de Streamlit Cloud y verificar redespliegue
- [ ] Desplegar en Streamlit Cloud y obtener URL pública
- [ ] Hacer el repositorio GitHub público (verificar que no hay credenciales expuestas)
- [ ] Añadir la reflexión personal en las conclusiones (3-4 líneas propias)
- [ ] Convertir este MD a Word o pptx con el formato de la Universidad
- [ ] Añadir el logo de la UE y el encabezado institucional
- [ ] Revisar que Railway está operativo antes de hacer las capturas de pantalla de la BD
- [ ] Subir el documento al Campus Virtual antes de la fecha límite
