<div align="center">

# 🫀 Asistente Virtual de Voz para Cardiología

**Procesamiento del Lenguaje Natural aplicado a entornos hospitalarios**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![MySQL](https://img.shields.io/badge/MySQL-9.4-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://mysql.com)
[![License](https://img.shields.io/badge/Licencia-Académica-green?style=flat-square)](LICENSE)

*Proyecto académico · Máster IA en Ciencias de la Salud · Universidad Europea de Madrid*  
*PLN Unidad 4 — Actividad AA1 · Código: 0MSR001107\_UA4\_AA1*

---

**[🚀 Ver demo en vivo](https://pln-voz.streamlit.app)** · **[📄 Código documentado](CODIGO_PROTOTIPO.md)**

</div>

---

## 📋 Descripción

Sistema de asistencia médica por voz diseñado para el entorno hospitalario cardiológico. El médico formula una pregunta de viva voz o por escrito; el sistema la transcribe, busca en una base de conocimiento de 500 entradas clínicas y genera una respuesta precisa apoyándose en Generación Aumentada por Recuperación (RAG) con GPT-4o-mini.

> ⚠️ **Aviso:** este sistema es un ejercicio académico de PLN. No constituye un dispositivo médico ni una herramienta clínica validada. No debe utilizarse para tomar decisiones diagnósticas o terapéuticas en pacientes reales.

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                   Navegador del médico                  │
└─────────────────────────┬───────────────────────────────┘
                          │  Audio WebM/Opus
                          ▼
              ┌───────────────────────┐
              │   st.audio_input()    │  Interfaz Streamlit
              └───────────┬───────────┘
                          │  bytes de audio
                          ▼
              ┌───────────────────────┐
              │   pydub               │  WebM → WAV
              └───────────┬───────────┘
                          │  WAV
                          ▼
              ┌───────────────────────┐
              │  Google Speech        │  Transcripción es-ES
              │  Recognition          │  + 50 correcciones
              └───────────┬───────────┘  fonéticas médicas
                          │  texto transcrito
                          ▼
              ┌───────────────────────┐
              │   TF-IDF              │  Doble índice:
              │   (scikit-learn)      │  pregunta / pregunta+contexto
              └───────────┬───────────┘  Top-2 similitud coseno
                          │  contextos recuperados
                          ▼
              ┌───────────────────────┐
              │  OpenAI GPT-4o-mini   │  RAG estricto:
              │  (RAG)                │  responde solo con
              └───────────┬───────────┘  las fuentes recuperadas
                          │  respuesta generada
                          ▼
              ┌───────────────────────┐
              │   MySQL (Railway)     │  Persistencia de
              │   src/db.py           │  consultas e historial
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Streamlit UI         │  Respuesta + confianza
              │  app.py               │  + historial + estadísticas
              └───────────────────────┘
```

---

## ✨ Funcionalidades

| Característica | Descripción |
|---|---|
| 🎙️ **Entrada por voz** | Grabación directa desde el navegador con `st.audio_input()` |
| ⌨️ **Entrada por texto** | Campo de texto con envío por Enter |
| 🔍 **TF-IDF de doble índice** | Búsqueda semántica sobre pregunta sola y pregunta+contexto |
| 🤖 **RAG con GPT-4o-mini** | Respuestas grounded, sin alucinaciones, basadas solo en el dataset |
| 📊 **Métricas en tiempo real** | Confianza (%), fuente de la respuesta y tiempo de procesamiento |
| 📋 **Historial MySQL** | Todas las consultas persistidas y exportables a CSV |
| 📈 **Estadísticas** | Rendimiento por categoría, consultas de baja confianza |
| ➕ **Aprendizaje activo** | Añadir entradas al dataset desde la UI sin reiniciar la app |
| 🔤 **Correcciones fonéticas** | ~50 términos cardiológicos corregidos automáticamente |

---

## 🗂️ Estructura del proyecto

```
pln-voz/
├── app.py                                   # Punto de entrada — interfaz Streamlit
├── requirements.txt                         # Dependencias Python
├── packages.txt                             # Dependencias del sistema (ffmpeg)
├── .env.example                             # Plantilla de variables de entorno
├── CODIGO_PROTOTIPO.md                      # Código completo documentado para la memoria
│
├── src/
│   ├── speech.py                            # Transcripción: Google SR + correcciones fonéticas
│   ├── nlp.py                               # Pipeline: TF-IDF retrieval + OpenAI RAG
│   └── db.py                                # Persistencia: MySQL Railway
│
├── data/
│   └── preguntas_cardiologia_esc_500.csv    # Dataset 500 entradas (ESC 2021-2023)
│
├── sql/
│   └── schema.sql                           # Esquema de la tabla MySQL
│
└── assets/
    └── UE_Madrid_Logo_Positive_RGB.png      # Logo institucional
```

---

## 🧠 Pipeline NLP en detalle

### Etapa 1 — Retrieval con TF-IDF

El sistema construye **dos matrices TF-IDF** al arrancar:

- **Matriz A** — vectoriza solo el campo `pregunta` de cada entrada. Produce scores altos cuando la consulta del médico coincide directamente con una pregunta del dataset.
- **Matriz B** — vectoriza `pregunta + contexto`. Actúa como respaldo cuando la pregunta sola puntúa bajo (< 35 %) pero la respuesta está en el contexto clínico.

La similitud coseno devuelve los **top-2 candidatos**. Si el mejor score es < 15 %, se devuelve el mensaje de fallback sin llamar a OpenAI.

### Etapa 2 — Generation con OpenAI RAG

Los contextos recuperados (respuesta de referencia + contexto ESC) se envían a **GPT-4o-mini** con un system prompt estricto:

- La respuesta debe basarse **exclusivamente** en las fuentes proporcionadas.
- Si las fuentes no cubren la pregunta, el modelo devuelve el mensaje de fallback estándar.
- Temperatura `0.1` para respuestas deterministas y clínicas.

---

## 🗃️ Dataset

- **500 entradas** de cardiología clínica en español
- **3 categorías:** Síntomas · Medicamentos · Protocolos
- **Fuente:** Guías de Práctica Clínica de la Sociedad Europea de Cardiología (ESC), ediciones 2021-2023
  - Síndrome coronario agudo · Insuficiencia cardíaca · Fibrilación auricular
  - Hipertensión arterial · Resucitación cardiopulmonar
- Cada entrada contiene: `id`, `categoria`, `pregunta`, `respuesta`, `contexto`

---

## ⚙️ Instalación y ejecución local

### Prerrequisitos

- Python 3.9+
- `ffmpeg` instalado en el sistema
- Cuenta en OpenAI (para la etapa RAG)
- Base de datos MySQL accesible (Railway u otro proveedor)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/carloscardozocardiologue/pln-voz.git
cd pln-voz

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 3. Instalar ffmpeg
brew install ffmpeg              # macOS
# sudo apt install ffmpeg        # Ubuntu/Debian

# 4. Configurar variables de entorno
cp .env.example .env
# → Editar .env con tus claves

# 5. Crear la tabla en MySQL (solo la primera vez)
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway < sql/schema.sql

# 6. Lanzar la aplicación
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

---

## 🔐 Variables de entorno

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | Clave de la API de OpenAI (GPT-4o-mini) |
| `DB_HOST` | Host MySQL (ej: `monorail.proxy.rlwy.net`) |
| `DB_PORT` | Puerto público MySQL Railway |
| `DB_USER` | Usuario MySQL (`root`) |
| `DB_PASSWORD` | Contraseña MySQL |
| `DB_NAME` | Nombre de la base de datos (`railway`) |

> En **Streamlit Cloud** estas variables se configuran en *Settings → Secrets*.

---

## 🗄️ Esquema de la base de datos

```sql
CREATE TABLE consultas (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    categoria     VARCHAR(20)  NOT NULL,          -- Síntomas | Medicamentos | Protocolos
    transcripcion TEXT         NOT NULL,          -- Pregunta del médico
    respuesta     TEXT         NOT NULL,          -- Respuesta generada
    confianza     FLOAT,                          -- Score TF-IDF (0.0 – 1.0)
    duracion_ms   INT,                            -- Tiempo de procesamiento
    fecha_hora    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🖥️ Capturas de pantalla

| Pestaña Asistente | Pestaña Historial |
|---|---|
| *Grabación de voz + respuesta clínica* | *Últimas 20 consultas exportables a CSV* |

| Pestaña Estadísticas | Pestaña Conocimiento |
|---|---|
| *Métricas por categoría y baja confianza* | *Dataset completo + añadir entradas* |

---

## 📚 Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| Interfaz web | Streamlit | ≥ 1.32 |
| Transcripción de voz | Google Speech Recognition | ≥ 3.10 |
| Conversión de audio | pydub + ffmpeg | ≥ 0.25 |
| Retrieval semántico | scikit-learn (TF-IDF) | ≥ 1.3 |
| Generación de respuestas | OpenAI GPT-4o-mini | ≥ 1.30 |
| Persistencia | MySQL + mysql-connector-python | 9.4 |
| Manejo de datos | pandas | ≥ 2.0 |
| Gestión de entorno | python-dotenv | ≥ 1.0 |

---

## 👨‍⚕️ Autor

**Dr. Carlos Cardozo**  
Cardiólogo intervencionista  
Máster IA en Ciencias de la Salud — Universidad Europea de Madrid

---

<div align="center">

*Actividad académica · PLN Unidad 4 · 2025-2026*  
*El contenido del dataset ha sido elaborado con fines exclusivamente académicos*  
*a partir de las Guías ESC de Práctica Clínica (ediciones 2021-2023).*

</div>
