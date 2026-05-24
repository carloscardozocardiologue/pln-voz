# Código del Prototipo — Asistente Virtual de Voz para Cardiología

**Actividad:** 0MSR001107\_UA4\_AA1 · PLN Unidad 4  
**Máster IA en Ciencias de la Salud — Universidad Europea de Madrid**

Este documento presenta el código mínimo necesario para ejecutar el asistente, organizado por módulos en el mismo orden en que fluye la información: voz → transcripción → NLP → base de datos → interfaz.

---

## Estructura del proyecto

```
pln-voz/
├── app.py                          # Interfaz Streamlit (punto de entrada)
├── requirements.txt                # Dependencias Python
├── packages.txt                    # Dependencias del sistema (ffmpeg)
├── .env.example                    # Plantilla de variables de entorno
├── src/
│   ├── speech.py                   # Transcripción de voz
│   ├── nlp.py                      # TF-IDF + OpenAI RAG
│   └── db.py                       # Persistencia MySQL
├── data/
│   └── preguntas_cardiologia_esc_500.csv   # Dataset de conocimiento
├── sql/
│   └── schema.sql                  # Esquema de la base de datos
└── assets/
    └── UE_Madrid_Logo_Positive_RGB.png
```

---

## Dependencias

**`requirements.txt`** — paquetes Python:

```
streamlit>=1.32.0
openai>=1.30.0
SpeechRecognition>=3.10.0
pydub>=0.25.0
scikit-learn>=1.3.0
mysql-connector-python>=9.0.0
python-dotenv>=1.0.1
pandas>=2.0.0
```

**`packages.txt`** — dependencias del sistema operativo (necesario en Streamlit Cloud):

```
ffmpeg
```

---

## Variables de entorno

Copiar `.env.example` a `.env` y rellenar los valores. El archivo `.env` **nunca** se sube a Git.

```ini
# API de OpenAI (necesaria para el módulo RAG)
OPENAI_API_KEY=sk-...

# Conexión a MySQL (Railway u otro proveedor)
DB_HOST=tu-host.railway.app
DB_PORT=12345
DB_USER=root
DB_PASSWORD=tu-contraseña
DB_NAME=railway
```

En Streamlit Cloud estos valores se configuran en **Settings → Secrets** con el mismo formato.

---

## Módulo 1 — Transcripción de voz (`src/speech.py`)

Convierte los bytes de audio grabados en el navegador (formato WebM/Opus) a texto en español mediante Google Speech Recognition, y aplica un diccionario de ~50 correcciones fonéticas para términos cardiológicos que el reconocedor suele malinterpretar.

```python
"""Transcripción de audio mediante Google Speech Recognition + corrección de términos médicos."""

import io
import speech_recognition as sr
from pydub import AudioSegment

# Correcciones fonéticas: lo que Google SR transcribe → término médico correcto
_CORRECCIONES = {
    # ── Anticoagulantes ───────────────────────────────────────────────
    "david galan":          "dabigatrán",
    "david galán":          "dabigatrán",
    "david gatran":         "dabigatrán",
    "david catran":         "dabigatrán",
    "david katran":         "dabigatrán",
    "davigatran":           "dabigatrán",
    "dabiga tran":          "dabigatrán",
    "rival oxaban":         "rivaroxabán",
    "ribaroxaban":          "rivaroxabán",
    "rivaroxaban":          "rivaroxabán",
    "apixa van":            "apixabán",
    "apixa ban":            "apixabán",
    "apixaban":             "apixabán",
    "edoxaban":             "edoxabán",
    # ── Antiagregantes ───────────────────────────────────────────────
    "tica grelor":          "ticagrelor",
    "ticker grelor":        "ticagrelor",
    "tika grelor":          "ticagrelor",
    "prazo grel":           "prasugrel",
    "clopi dogrel":         "clopidogrel",
    # ── Antiarrítmicos ───────────────────────────────────────────────
    "amio darona":          "amiodarona",
    "amio de rona":         "amiodarona",
    "drone darona":         "dronedarona",
    "dron darona":          "dronedarona",
    # ── Insuficiencia cardíaca ────────────────────────────────────────
    "sacubitril":           "sacubitrilo",
    "saku bitrilo":         "sacubitrilo",
    "epler enona":          "eplerenona",
    "espiro nolactona":     "espironolactona",
    "empagliflocina":       "empagliflozina",
    "empagliflozina":       "empagliflozina",
    "dapagliflocina":       "dapagliflozina",
    "canagliflocina":       "canagliflozina",
    # ── IECA / ARA-II ────────────────────────────────────────────────
    "perindo pril":         "perindopril",
    "cande sartan":         "candesartán",
    "candesartan":          "candesartán",
    "irbe sartan":          "irbesartán",
    "irbesartan":           "irbesartán",
    "olme sartan":          "olmesartán",
    "olmesartan":           "olmesartán",
    "telmi sartan":         "telmisartán",
    "telmisartan":          "telmisartán",
    "losartan":             "losartán",
    "valsartan":            "valsartán",
    # ── Betabloqueantes ──────────────────────────────────────────────
    "nebi bolol":           "nebivolol",
    "nebi volol":           "nebivolol",
    # ── Calcioantagonistas ───────────────────────────────────────────
    "amlo dipino":          "amlodipino",
    "vera pamilo":          "verapamilo",
    "dil tia sem":          "diltiazem",
    "diltia sem":           "diltiazem",
    # ── Estatinas ────────────────────────────────────────────────────
    "rosu vastatina":       "rosuvastatina",
    "pita vastatina":       "pitavastatina",
    # ── Términos clínicos con tildes ─────────────────────────────────
    "fibrilacion":          "fibrilación",
    "fibrilacion auricular":"fibrilación auricular",
    "ablacion":             "ablación",
    "cardioversion":        "cardioversión",
    "desfibrilacion":       "desfibrilación",
    "resincronizacion":     "resincronización",
    "resin cronizacion":    "resincronización",
    "trombo embolia":       "tromboembolia",
    "trombo embolismo":     "tromboembolismo",
    "sindrome coronario":   "síndrome coronario",
    "insuficiencia cardiaca": "insuficiencia cardíaca",
    "arritmia":             "arritmia",
    "hipertension":         "hipertensión",
    "hipertension arterial":"hipertensión arterial",
    "disfuncion":           "disfunción",
    "intervencion":         "intervención",
    "oclusion":             "oclusión",
    "estenosis":            "estenosis",
}


def _corregir(texto: str) -> str:
    """Aplica el diccionario de correcciones fonéticas al texto transcrito."""
    t = texto.lower()
    # Orden por longitud descendente: "fibrilación auricular" debe coincidir
    # antes que "fibrilación" para que la frase completa tenga prioridad
    for erroneo, correcto in sorted(_CORRECCIONES.items(), key=lambda x: -len(x[0])):
        t = t.replace(erroneo, correcto)
    return t[0].upper() + t[1:] if t else t


def transcribir(audio_bytes: bytes) -> str:
    """
    Recibe los bytes de audio grabados en Streamlit y devuelve
    el texto transcrito en español usando Google Speech Recognition.
    """
    # El navegador solo puede grabar en WebM/Opus; Google SR exige WAV
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_data = recognizer.record(source)

    # language="es-ES" reduce errores en términos médicos castellanos
    texto = recognizer.recognize_google(audio_data, language="es-ES")
    return _corregir(texto)
```

---

## Módulo 2 — Pipeline NLP: TF-IDF + OpenAI RAG (`src/nlp.py`)

El núcleo del sistema. Implementa un pipeline de dos etapas:

1. **Retrieval (TF-IDF):** busca las entradas más similares del dataset usando dos matrices TF-IDF (una sobre la pregunta sola, otra sobre pregunta+contexto como respaldo).
2. **Generation (OpenAI RAG):** si la similitud supera el umbral mínimo (15 %), envía los contextos recuperados a GPT-4o-mini con un system prompt estricto que prohíbe añadir información fuera de las fuentes.

```python
"""
Pipeline NLP: TF-IDF retrieval + OpenAI RAG en español.

Etapa 1 — TF-IDF: encuentra los contextos más similares a la pregunta del médico.
Etapa 2 — OpenAI: genera una respuesta basándose ESTRICTAMENTE en esos contextos.
           Si el contexto no cubre la pregunta, devuelve el mensaje de fallback.
Fallback absoluto: si TF-IDF < umbral mínimo, se devuelve el mensaje de fallback
           sin llamar a OpenAI.
"""

import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_DATA_PATH           = Path(__file__).parent.parent / "data" / "preguntas_cardiologia_esc_500.csv"
_TFIDF_UMBRAL        = 0.15   # mínimo para intentar responder (umbral absoluto)
_TFIDF_PREGUNTA_MIN  = 0.35   # si pregunta-sola supera esto, no hace falta buscar en contexto
_TOP_K               = 2


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y prefijos médicos normalizados al español."""
    texto = texto.lower()
    texto = re.sub(r'\bhypo', 'hipo', texto)
    texto = re.sub(r'\bhyper', 'hiper', texto)
    texto = re.sub(r'\bpneu', 'neu', texto)
    texto = re.sub(r'\bthrombo', 'trombo', texto)
    texto = re.sub(r'\barrhyth', 'arrit', texto)
    nfkd  = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _cargar():
    """Lee el dataset CSV y construye las dos matrices TF-IDF al arrancar la aplicación."""
    df = pd.read_csv(_DATA_PATH)

    # Matriz 1: solo pregunta — alta similitud para preguntas exactas o casi exactas
    vec_p = TfidfVectorizer(ngram_range=(1, 2))
    mat_p = vec_p.fit_transform(df["pregunta"].apply(_normalizar))

    # Matriz 2: pregunta + contexto — fallback para keywords que solo aparecen en el contexto
    vec_pc = TfidfVectorizer(ngram_range=(1, 2))
    mat_pc = vec_pc.fit_transform(
        (df["pregunta"] + " " + df["contexto"].fillna("")).apply(_normalizar)
    )

    return df, vec_p, mat_p, vec_pc, mat_pc


def _get_openai_key() -> str:
    """Obtiene la clave de OpenAI desde st.secrets (Streamlit Cloud) o desde .env (local)."""
    try:
        import streamlit as st
        val = st.secrets.get("OPENAI_API_KEY")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


_df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()

FALLBACK = (
    "No tengo información suficiente en mi base de conocimiento para responder "
    "esa pregunta. Por favor, reformula la consulta o selecciona otra categoría."
)

_SYSTEM_PROMPT = (
    "Eres un asistente médico especializado en cardiología para uso hospitalario. "
    "Para cada consulta recibirás una o más entradas del dataset, cada una con:\n"
    "- 'Respuesta de referencia': respuesta corta validada por el clínico. Es la fuente principal.\n"
    "- 'Contexto clínico': párrafo de las guías ESC que amplía la información.\n"
    "REGLAS ESTRICTAS:\n"
    "1. Basa tu respuesta principalmente en la 'Respuesta de referencia'.\n"
    "2. Puedes complementar con información del 'Contexto clínico' si aporta valor adicional.\n"
    "3. No añadas información que no esté en ninguna de las dos fuentes.\n"
    "4. Si ninguna fuente contiene información suficiente para responder la pregunta, "
    'responde exactamente: "No tengo información suficiente en mi base de conocimiento '
    'para responder esa pregunta. Por favor, reformula la consulta o selecciona otra categoría."\n'
    "5. Responde en español, de forma concisa y clínica.\n"
    "6. No inventes datos, dosis ni recomendaciones que no estén en las fuentes."
)


def recargar():
    """Reconstruye los índices TF-IDF después de añadir entradas al dataset."""
    global _df, _vec_p, _mat_p, _vec_pc, _mat_pc
    _df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()


def _openai_responder(pregunta: str, contextos: list) -> tuple:
    """
    Genera una respuesta grounded usando OpenAI.
    Devuelve (respuesta: str, estado: str).
    Estado: 'ok' | 'no_info' | 'error_api' | 'sin_clave'.
    """
    key = _get_openai_key()
    if not key:
        return "", "sin_clave"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        fuentes_texto = "\n\n".join(
            f"Entrada {i + 1}:\n"
            f"  Respuesta de referencia: {ctx['respuesta']}\n"
            f"  Contexto clínico: {ctx['contexto']}"
            for i, ctx in enumerate(contextos)
        )
        user_message = (
            f"Pregunta del médico: {pregunta}\n\n"
            f"Fuentes disponibles:\n{fuentes_texto}\n\n"
            "Responde basándote en las fuentes proporcionadas."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=300,
            temperature=0.1,   # temperatura baja = respuestas deterministas, sin alucinaciones
        )

        respuesta = response.choices[0].message.content.strip()
        estado = "no_info" if respuesta.startswith("No tengo información suficiente") else "ok"
        return respuesta, estado

    except Exception:
        return "", "error_api"


def responder(pregunta: str) -> dict:
    """
    Devuelve un dict con 'respuesta' (str) y 'confianza' (float 0-1).

    Flujo:
      1. TF-IDF selecciona los TOP_K contextos más relevantes del dataset.
      2. Si TF-IDF supera el umbral mínimo, OpenAI genera la respuesta basada
         estrictamente en esos contextos (RAG).
      3. Si OpenAI determina que el contexto no es suficiente, se devuelve
         el mensaje de fallback.
      4. Si TF-IDF no supera el umbral mínimo, se devuelve el fallback directamente.
    """
    # — Etapa 1: TF-IDF retrieval (dos matrices) —
    q_norm = _normalizar(pregunta)

    # Primero: solo pregunta — scores altos para coincidencias directas
    sim_p   = cosine_similarity(_vec_p.transform([q_norm]), _mat_p).flatten()
    top_p   = sim_p.argsort()[::-1][:_TOP_K]
    score_p = float(sim_p[top_p[0]])

    # Si pregunta-sola no supera el umbral, buscar también en pregunta+contexto
    if score_p < _TFIDF_PREGUNTA_MIN:
        sim_pc   = cosine_similarity(_vec_pc.transform([q_norm]), _mat_pc).flatten()
        top_pc   = sim_pc.argsort()[::-1][:_TOP_K]
        score_pc = float(sim_pc[top_pc[0]])
        if score_pc > score_p:
            top_indices     = top_pc
            sim_activa      = sim_pc
            confianza_tfidf = score_pc
        else:
            top_indices     = top_p
            sim_activa      = sim_p
            confianza_tfidf = score_p
    else:
        top_indices     = top_p
        sim_activa      = sim_p
        confianza_tfidf = score_p

    if confianza_tfidf < _TFIDF_UMBRAL:
        return {"respuesta": FALLBACK, "confianza": round(confianza_tfidf, 3), "categoria": "Sin categoría"}

    # — Etapa 2: OpenAI RAG —
    contextos = [
        {
            "respuesta": _df.iloc[idx]["respuesta"],
            "contexto":  _df.iloc[idx]["contexto"],
        }
        for idx in top_indices
        if sim_activa[idx] >= _TFIDF_UMBRAL
    ]

    respuesta_openai, openai_estado = _openai_responder(pregunta, contextos)

    mejor_fila = _df.iloc[top_indices[0]]

    if openai_estado == "ok":
        respuesta_final = respuesta_openai
        openai_usado    = True
    elif openai_estado in ("no_info",):
        respuesta_final = FALLBACK
        openai_usado    = False
    else:
        # sin_clave o error_api → usar respuesta pre-escrita del dataset
        respuesta_final = mejor_fila["respuesta"]
        openai_usado    = False

    return {
        "respuesta":        respuesta_final,
        "confianza":        round(confianza_tfidf, 3),
        "categoria":        mejor_fila["categoria"],
        "openai_usado":     openai_usado,
        "openai_estado":    openai_estado,
        "pregunta_dataset": mejor_fila["pregunta"],
        "total_entradas":   len(_df),
    }
```

---

## Módulo 3 — Persistencia MySQL (`src/db.py`)

Gestiona la conexión a la base de datos MySQL en Railway y expone funciones para guardar consultas, recuperar el historial y calcular estadísticas de rendimiento.

```python
"""Módulo de persistencia: guarda y recupera consultas en MySQL."""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def _cfg(key: str, default=None):
    """Lee primero de st.secrets (Streamlit Cloud) y luego de os.getenv (local)."""
    # Prioridad: st.secrets (producción en Streamlit Cloud) → .env (entorno local)
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


def _conectar():
    """Abre y devuelve una conexión MySQL usando las credenciales del entorno."""
    return mysql.connector.connect(
        host=_cfg("DB_HOST"),
        port=int(_cfg("DB_PORT", 3306)),
        user=_cfg("DB_USER"),
        password=_cfg("DB_PASSWORD"),
        database=_cfg("DB_NAME"),
        connection_timeout=3,   # 3 s máximo para no bloquear la UI si Railway no responde
    )


def guardar(categoria: str, transcripcion: str, respuesta: str,
            confianza: float = None, duracion_ms: int = None):
    """Inserta una consulta completa en la tabla 'consultas' de MySQL."""
    conn = _conectar()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO consultas (categoria, transcripcion, respuesta, confianza, duracion_ms) "
        "VALUES (%s, %s, %s, %s, %s)",
        (categoria, transcripcion, respuesta, confianza, duracion_ms),
    )
    conn.commit()
    cursor.close()
    conn.close()


def borrar_historial():
    """Elimina todas las filas de la tabla 'consultas' (acción irreversible)."""
    conn = _conectar()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM consultas")
    conn.commit()
    cursor.close()
    conn.close()


def obtener_estadisticas() -> dict:
    """Devuelve métricas globales y por categoría: total, confianza media, baja confianza y tiempo medio."""
    conn = _conectar()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            COUNT(*)                                              AS total,
            ROUND(AVG(confianza), 3)                             AS confianza_media,
            SUM(confianza < 0.40)                                AS baja_confianza,
            ROUND(AVG(duracion_ms))                              AS tiempo_medio_ms
        FROM consultas
    """)
    resumen = cursor.fetchone()

    cursor.execute("""
        SELECT
            categoria,
            COUNT(*)                    AS consultas,
            ROUND(AVG(confianza), 3)    AS confianza_media,
            ROUND(AVG(duracion_ms))     AS tiempo_medio_ms
        FROM consultas
        GROUP BY categoria
        ORDER BY consultas DESC
    """)
    por_categoria = cursor.fetchall()

    cursor.close()
    conn.close()
    return {"resumen": resumen, "por_categoria": por_categoria}


def obtener_baja_confianza(umbral: float = 0.40, limite: int = 10) -> list[dict]:
    """Devuelve las consultas con confianza TF-IDF por debajo del umbral, ordenadas de menor a mayor."""
    conn = _conectar()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT categoria, transcripcion, confianza
        FROM consultas
        WHERE confianza < %s
        ORDER BY confianza ASC, fecha_hora DESC
        LIMIT %s
        """,
        (umbral, limite),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def obtener_historial(limite: int = 20) -> list[dict]:
    """Devuelve las últimas N consultas ordenadas por fecha descendente."""
    conn = _conectar()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT fecha_hora, categoria, transcripcion, respuesta, confianza "
        "FROM consultas ORDER BY fecha_hora DESC LIMIT %s",
        (limite,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
```

---

## Módulo 4 — Interfaz Streamlit (`app.py`)

Punto de entrada de la aplicación. Gestiona la interfaz de usuario con cuatro pestañas: Asistente (voz/texto), Historial, Estadísticas y Base de conocimiento. Orquesta las llamadas a los tres módulos anteriores.

```python
"""
Asistente Virtual de Voz para Cardiología
Máster IA en Ciencias de la Salud — Universidad Europea de Madrid
PLN Unidad 4 — Actividad AA1
"""

import base64
import time
import pandas as pd
import streamlit as st
from pathlib import Path
from src.speech import transcribir
from src.nlp import responder, recargar, FALLBACK as _FALLBACK
from src.db import (
    guardar, obtener_historial, borrar_historial,
    obtener_estadisticas, obtener_baja_confianza,
)

_DATA_PATH = Path(__file__).parent / "data" / "preguntas_cardiologia_esc_500.csv"

_LOGO_PATH = Path(__file__).parent / "assets" / "UE_Madrid_Logo_Positive_RGB.png"
_LOGO_B64  = (
    base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    if _LOGO_PATH.exists() else None
    # Codificación base64 para incrustar la imagen directamente en HTML
    # sin depender de rutas de archivo en Streamlit Cloud
)

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente Virtual · Cardiología",
    page_icon="🫀",
    layout="centered",
)

# ── Cabecera ──────────────────────────────────────────────────────────────────
if _LOGO_B64:
    st.markdown(
        f'<div style="background:#ffffff;padding:10px;border-radius:8px;display:inline-block;">'
        f'<img src="data:image/png;base64,{_LOGO_B64}" style="height:60px;">'
        f'</div>',
        unsafe_allow_html=True,
    )

st.title("🫀 Asistente Virtual de Cardiología")
st.caption("Universidad Europea de Madrid")
st.caption(
    "Máster IA en Ciencias de la Salud · PLN Unidad 4 · "
    "Implementación de un asistente virtual de voz para entornos hospitalarios"
)

st.warning(
    "⚠️ **Aviso importante:** esta aplicación es un ejercicio académico de Procesamiento "
    "del Lenguaje Natural (PLN). No constituye un dispositivo médico ni una herramienta "
    "clínica validada. **No debe utilizarse para tomar decisiones diagnósticas o terapéuticas "
    "en pacientes reales.**",
    icon=None,
)
st.divider()

# ── Dialog (debe definirse fuera de las pestañas) ─────────────────────────────
# Streamlit renderiza el código de arriba a abajo; si el decorator @st.dialog
# está dentro de un bloque `with tab_X:` falla cuando se llama desde otra pestaña
@st.dialog("Añadir al dataset de conocimiento")
def dialog_mejorar(pregunta: str, cat: str):
    """Muestra un formulario modal para añadir una nueva entrada al dataset cuando la confianza es baja."""
    st.markdown("**Pregunta detectada con baja confianza:**")
    st.info(pregunta)
    nueva_respuesta = st.text_area("Respuesta corta (visible en la app)", height=80)
    nuevo_contexto  = st.text_area(
        "Contexto clínico completo (OpenAI lo usa para generar la respuesta)", height=130
    )
    if st.button("Guardar en el dataset", type="primary"):
        if not nueva_respuesta.strip() or not nuevo_contexto.strip():
            st.error("La respuesta y el contexto son obligatorios.")
        else:
            categoria_detectada = responder(pregunta)["categoria"]
            df_actual = pd.read_csv(_DATA_PATH)
            nuevo_id  = int(df_actual["id"].max()) + 1
            pd.concat([
                df_actual,
                pd.DataFrame([{
                    "id": nuevo_id, "categoria": categoria_detectada,
                    "pregunta": pregunta.strip(),
                    "respuesta": nueva_respuesta.strip(),
                    "contexto":  nuevo_contexto.strip(),
                }])
            ], ignore_index=True).to_csv(_DATA_PATH, index=False)
            recargar()
            st.success(f"Entrada #{nuevo_id} añadida en **{categoria_detectada}**.")
            st.rerun()


# ── Pestañas ──────────────────────────────────────────────────────────────────
tab_asistente, tab_historial, tab_stats, tab_conocimiento = st.tabs([
    "🎙️ Asistente", "📋 Historial", "📊 Estadísticas", "📚 Conocimiento"
])

# ═══════════════════════════════════════════════════════════════════
# PESTAÑA 1 — ASISTENTE
# ═══════════════════════════════════════════════════════════════════
with tab_asistente:
    _k = st.session_state.get("input_key", 0)
    audio = st.audio_input("Pulsa el micrófono, habla y pulsa detener", key=f"audio_{_k}")

    st.write("— o escribe tu consulta —")
    with st.form(key=f"form_{_k}", clear_on_submit=False):
        texto_escrito = st.text_input(
            "Consulta escrita",
            placeholder="¿Cuáles son los síntomas del infarto?",
            label_visibility="collapsed",
        )
        consultar = st.form_submit_button("Consultar", use_container_width=True, type="primary")

    if st.button("Nueva consulta", use_container_width=True, type="secondary"):
        st.session_state["ultima_consulta"] = None
        # Incrementar la key fuerza a Streamlit a destruir y recrear el widget de audio,
        # lo que equivale a "vaciar" el micrófono sin necesidad de recargar la página
        st.session_state["input_key"] = _k + 1
        st.rerun()

    if "ultima_consulta" not in st.session_state:
        st.session_state["ultima_consulta"] = None
    if "input_key" not in st.session_state:
        st.session_state["input_key"] = 0

    transcripcion = None
    fuente_input  = "texto"
    if audio:
        with st.spinner("Transcribiendo audio..."):
            try:
                transcripcion = transcribir(audio.read())
                fuente_input  = "voz"
            except Exception as e:
                st.error(f"Error en la transcripción: {e}")
                st.stop()
    elif consultar and texto_escrito.strip():
        transcripcion = texto_escrito.strip()

    if transcripcion:
        with st.spinner("Procesando con el modelo de lenguaje..."):
            inicio = time.time()
            try:
                resultado = responder(transcripcion)
            except Exception as e:
                st.error(f"Error en el modelo PLN: {e}")
                st.stop()
            duracion_ms = int((time.time() - inicio) * 1000)

        st.session_state["ultima_consulta"] = {
            "transcripcion": transcripcion,
            "resultado":     resultado,
            "duracion_ms":   duracion_ms,
            "fuente":        fuente_input,
        }
        st.session_state["input_key"] = _k + 1
        try:
            guardar(
                resultado["categoria"], transcripcion,
                resultado["respuesta"], resultado["confianza"], duracion_ms,
            )
        except Exception as e:
            st.warning(f"No se pudo guardar en la base de datos: {e}")
        st.rerun()

    if st.session_state["ultima_consulta"]:
        uc         = st.session_state["ultima_consulta"]
        confianza        = uc["resultado"]["confianza"]
        respuesta        = uc["resultado"]["respuesta"]
        categoria        = uc["resultado"]["categoria"]
        openai_usado     = uc["resultado"].get("openai_usado", False)
        openai_estado    = uc["resultado"].get("openai_estado", "sin_clave")
        pregunta_dataset = uc["resultado"].get("pregunta_dataset", "")
        pregunta         = uc["transcripcion"]

        st.markdown(
            f"<span style='color:#1a6fcc; font-weight:bold'>Médico:</span> "
            f"<span style='color:#1a6fcc; font-style:italic'>{pregunta}</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        fuente_label = "🎙️ Entrada por voz" if uc.get("fuente") == "voz" else "⌨️ Entrada por texto"
        st.caption(f"{fuente_label} · Categoría detectada: {categoria}")
        st.markdown(f"**Asistente:** {respuesta}")
        st.write("")

        confianza_display = confianza
        fuente_confianza  = "OpenAI + TF-IDF" if openai_usado else "TF-IDF"
        color = "green" if confianza_display >= 0.7 else "orange" if confianza_display >= 0.4 else "red"
        col_conf, col_mejorar = st.columns([5, 1])
        with col_conf:
            st.markdown(
                f"Confianza de la respuesta ({fuente_confianza}): "
                f"<span style='color:{color}; font-weight:bold'>{confianza_display:.0%}</span>",
                unsafe_allow_html=True,
            )
            st.progress(confianza_display)
        with col_mejorar:
            if confianza_display < 0.85:
                st.write("")
                if st.button("Mejorar", type="secondary", use_container_width=True):
                    dialog_mejorar(pregunta, categoria)

        total_entradas = uc["resultado"].get("total_entradas", 500)
        _openai_textos = {
            "ok":
                "✅ **Paso 2 — OpenAI GPT:** el modelo leyó los contextos clínicos "
                "recuperados por TF-IDF y generó una respuesta basándose estrictamente "
                "en esa información. La respuesta es fiel al contenido del dataset.",
            "no_info":
                "ℹ️ **Paso 2 — OpenAI GPT:** los contextos recuperados no contenían "
                "información suficiente para responder esta pregunta con certeza. "
                "Se muestra el mensaje de aviso.",
            "error_api":
                "⚠️ **Paso 2 — OpenAI GPT:** no se pudo conectar con la API de OpenAI "
                "(problema de red o límite de uso). Se muestra el texto pre-escrito del dataset.",
            "sin_clave":
                "⚠️ **Paso 2 — OpenAI GPT:** no hay clave de API configurada. "
                "Se muestra el texto pre-escrito del dataset.",
        }
        with st.expander("¿Cómo se obtuvo esta respuesta?"):
            st.markdown(
                f"**Paso 1 — Búsqueda en el dataset (TF-IDF): {confianza:.0%}** — "
                f"el sistema comparó tu pregunta con las {total_entradas} entradas "
                f"del dataset y encontró la más similar: "
                f"_\"{pregunta_dataset}\"_"
            )
            st.markdown(_openai_textos.get(openai_estado, ""))
            st.divider()
            if confianza_display >= 0.7:
                st.success("Coincidencia alta — la respuesta es fiable.")
            elif confianza_display >= 0.4:
                st.warning("Coincidencia media — reformular la pregunta puede mejorar el resultado.")
            else:
                st.error("Coincidencia baja — la pregunta se aleja del dataset.")


# ═══════════════════════════════════════════════════════════════════
# PESTAÑA 2 — HISTORIAL
# ═══════════════════════════════════════════════════════════════════
with tab_historial:
    st.subheader("Historial de consultas")
    st.caption("Últimas 20 consultas registradas en la base de datos.")

    try:
        historial = obtener_historial(20)
        if historial:
            df = pd.DataFrame(historial)
            df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]).dt.strftime("%d/%m/%Y %H:%M")
            df["confianza"]  = df["confianza"].apply(
                lambda x: f"{x:.0%}" if x is not None else "-"
            )
            df.columns = ["Fecha y hora", "Categoría", "Transcripción", "Respuesta", "Confianza"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Exportar historial CSV",
                    data=csv,
                    file_name="historial_consultas.csv",
                    mime="text/csv",
                )
            with col2:
                if st.button("Borrar historial completo", type="secondary"):
                    borrar_historial()
                    st.success("Historial borrado.")
                    st.rerun()
        else:
            st.info("Aún no hay consultas registradas.")
    except Exception as e:
        st.warning(f"No se pudo cargar el historial: {e}")


# ═══════════════════════════════════════════════════════════════════
# PESTAÑA 3 — ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("Estadísticas del sistema")
    st.caption("Análisis del rendimiento basado en las consultas almacenadas.")

    try:
        stats         = obtener_estadisticas()
        resumen       = stats["resumen"]
        por_categoria = stats["por_categoria"]

        if resumen["total"]:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total de consultas",        int(resumen["total"]))
            col2.metric("Confianza media",           f"{float(resumen['confianza_media']):.0%}")
            col3.metric("Baja confianza",            int(resumen["baja_confianza"] or 0))
            col4.metric("Tiempo medio de respuesta", f"{int(resumen['tiempo_medio_ms'] or 0)} ms")

            st.divider()

            if por_categoria:
                st.markdown("**Rendimiento por categoría**")
                df_stats = pd.DataFrame(por_categoria)
                df_stats.columns = ["Categoría", "Consultas", "Confianza media", "Tiempo medio (ms)"]
                df_stats["Confianza media"] = df_stats["Confianza media"].apply(
                    lambda x: f"{float(x):.0%}" if x is not None else "-"
                )
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Consultas con baja confianza — pendientes de mejorar**")
            baja_confianza = obtener_baja_confianza(umbral=0.40, limite=10)
            if baja_confianza:
                for i, fila in enumerate(baja_confianza):
                    c1, c2, c3 = st.columns([5, 1, 1])
                    c1.write(fila["transcripcion"])
                    c2.markdown(
                        f"<span style='color:red;font-weight:bold'>"
                        f"{fila['confianza']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    if c3.button("Mejorar", key=f"mejorar_{i}"):
                        dialog_mejorar(fila["transcripcion"], fila["categoria"])
            else:
                st.success("No hay consultas con baja confianza.")
        else:
            st.info("Aún no hay datos suficientes para mostrar estadísticas.")
    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")


# ═══════════════════════════════════════════════════════════════════
# PESTAÑA 4 — BASE DE CONOCIMIENTO
# ═══════════════════════════════════════════════════════════════════
with tab_conocimiento:
    st.subheader("Base de conocimiento")
    st.caption(
        "Añade nuevas preguntas y respuestas al dataset de cardiología. "
        "El modelo se actualiza automáticamente sin reiniciar la app."
    )

    with st.form("form_nueva_entrada", clear_on_submit=True):
        nueva_pregunta  = st.text_input("Pregunta")
        nueva_respuesta = st.text_area("Respuesta corta (visible en la app)", height=80)
        nuevo_contexto  = st.text_area(
            "Contexto completo (OpenAI lo usa para generar la respuesta)", height=120
        )
        enviado = st.form_submit_button("Añadir al dataset", type="primary")

    if enviado:
        if not nueva_pregunta.strip() or not nueva_respuesta.strip() or not nuevo_contexto.strip():
            st.error("Todos los campos son obligatorios.")
        else:
            categoria_detectada = responder(nueva_pregunta.strip())["categoria"]
            df_actual = pd.read_csv(_DATA_PATH)
            nuevo_id  = int(df_actual["id"].max()) + 1
            pd.concat([
                df_actual,
                pd.DataFrame([{
                    "id": nuevo_id, "categoria": categoria_detectada,
                    "pregunta":  nueva_pregunta.strip(),
                    "respuesta": nueva_respuesta.strip(),
                    "contexto":  nuevo_contexto.strip(),
                }])
            ], ignore_index=True).to_csv(_DATA_PATH, index=False)
            recargar()
            st.success(
                f"Entrada #{nuevo_id} añadida en **{categoria_detectada}**. "
                "El modelo ya incluye esta pregunta."
            )

    st.divider()
    st.markdown("**Entradas actuales del dataset**")
    df_dataset = pd.read_csv(_DATA_PATH)
    st.dataframe(
        df_dataset[["id", "categoria", "pregunta", "respuesta", "contexto"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.caption(
        "**Fuente del dataset inicial:** preguntas y contextos clínicos elaborados a partir de las "
        "Guías de Práctica Clínica de la Sociedad Europea de Cardiología (ESC), ediciones 2021-2023. "
        "El contenido ha sido estructurado y adaptado con fines exclusivamente académicos."
    )
```

---

## Esquema de la base de datos (`sql/schema.sql`)

```sql
CREATE DATABASE IF NOT EXISTS railway CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE railway;

CREATE TABLE IF NOT EXISTS consultas (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    categoria     VARCHAR(20)   NOT NULL,
    transcripcion TEXT          NOT NULL,
    respuesta     TEXT          NOT NULL,
    confianza     FLOAT,
    duracion_ms   INT,
    fecha_hora    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

---

## Cómo ejecutar el proyecto en local

```bash
# 1. Clonar el repositorio
git clone https://github.com/carloscardozocardiologue/pln-voz.git
cd pln-voz

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt

# 3. Instalar ffmpeg (necesario para pydub)
brew install ffmpeg              # macOS
# sudo apt install ffmpeg        # Ubuntu/Debian

# 4. Configurar variables de entorno
cp .env.example .env
# → editar .env con tus claves de OpenAI y MySQL

# 5. Crear la tabla en MySQL (solo la primera vez)
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway < sql/schema.sql

# 6. Lanzar la aplicación
streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`.

---

*Código desarrollado por Dr. Carlos Cardozo para la asignatura PLN, Máster IA en Ciencias de la Salud, Universidad Europea de Madrid.*
