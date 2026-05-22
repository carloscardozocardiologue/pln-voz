# CLAUDE.md — UA4 AA1: Asistente Virtual de Voz para Entornos Hospitalarios

## Resumen del proyecto

Actividad académica de la asignatura PLN (Procesamiento de Lenguaje Natural), Unidad 4, Máster de IA en Ciencias de la Salud — Universidad Europea de Madrid.

El sistema captura peticiones de voz de médicos en un entorno hospitalario cardiológico, las transcribe con Google Speech Recognition, las procesa con TF-IDF + OpenAI GPT-4o-mini (RAG estricto) y almacena las consultas y respuestas en una base de datos MySQL remota (Railway).

**Código de actividad:** 0MSR001107_UA4_AA1  
**Producción:** Streamlit Cloud — repo `carloscardozocardiologue/pln-voz`

## Filosofía y producto final

**Principio fundamental:** el sistema debe funcionar con audio real de micrófono — no basta con texto sintético como entrada.

Lo que produce el sistema:
- Aplicación Streamlit con grabación de voz, transcripción automática, respuesta médica e historial
- Base de datos MySQL remota (Railway) con consultas almacenadas
- Documentación del proceso (pptx o Word): contexto, metodología, herramientas, análisis de resultados, conclusiones y referencias al estado del arte

## Arquitectura

```
Navegador del médico
        │
        ▼
   st.audio_input()  ←  graba desde el navegador
        │ audio bytes (WebM)
        ▼
pydub — conversión WebM → WAV
        │
        ▼
Google Speech Recognition (es-ES)   ← src/speech.py + diccionario ~50 correcciones fonéticas
        │ texto transcrito
        ▼
TF-IDF (doble índice: pregunta / pregunta+contexto)   ← src/nlp.py
        │ top-2 contextos del dataset + score de similitud
        ▼
OpenAI GPT-4o-mini (RAG estricto)    ← API OpenAI, si TF-IDF ≥ 15%
        │ respuesta generada solo con el contexto recuperado
        ▼
MySQL Railway                        ← src/db.py
        │
        ▼
Streamlit — respuesta + confianza + historial + estadísticas
```

## Servicios — referencia rápida

| Componente        | Tecnología                                                             | Archivo         | Función                                    |
|-------------------|------------------------------------------------------------------------|-----------------|--------------------------------------------|
| Interfaz          | Streamlit ≥ 1.32                                                       | `app.py`        | UI completa: grabación, texto, historial   |
| Transcripción voz | `speech_recognition` + `pydub` (Google SR es-ES)                      | `src/speech.py` | Audio bytes → texto + correcciones fonéticas |
| Procesamiento PLN | TF-IDF (scikit-learn) + OpenAI GPT-4o-mini (RAG)                      | `src/nlp.py`    | Pregunta → respuesta médica + confianza    |
| Dataset           | CSV 500 entradas cardiología (pregunta / respuesta / contexto)         | `data/preguntas_cardiologia_esc_500.csv` | Base de conocimiento |
| Persistencia      | mysql-connector-python                                                 | `src/db.py`     | Guarda y recupera consultas en MySQL       |
| Base de datos     | MySQL 9.4 en Railway                                                   | `sql/schema.sql`| Tabla `consultas`                          |

## Comandos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Crear la tabla en MySQL (primera vez)
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway < sql/schema.sql

# Ejecutar en local
streamlit run app.py

# Verificar datos guardados
mysql -h <DB_HOST> -P <DB_PORT> -u root -p railway -e "SELECT * FROM consultas LIMIT 10;"
```

## Configuración — variables de entorno

| Variable         | Uso                                              |
|------------------|--------------------------------------------------|
| `OPENAI_API_KEY` | Clave de la API de OpenAI (GPT-4o-mini RAG)      |
| `DB_HOST`        | Host MySQL Railway (proxy público)               |
| `DB_PORT`        | Puerto público Railway (≠ 3306 interno)          |
| `DB_USER`        | Usuario MySQL (root)                             |
| `DB_PASSWORD`    | Contraseña MySQL Railway                         |
| `DB_NAME`        | Nombre de la base de datos (railway)             |

Copia `.env.example` a `.env` y rellena los valores. El `.env` nunca se sube a git.

## Comportamientos clave del pipeline NLP

- **Doble índice TF-IDF:** matriz 1 sobre `pregunta` sola (scores altos para coincidencias exactas); matriz 2 sobre `pregunta+contexto` como fallback cuando el score de la primera es < 35%. Esto evita que una coincidencia exacta puntúe 49% en vez de ~100%.
- **OpenAI RAG estricto (dos fuentes):** TF-IDF recupera los top-2 resultados y pasa a GPT-4o-mini tanto la `respuesta` del dataset (fuente principal, validada por el clínico) como el `contexto` clínico (párrafo ESC, como respaldo). Un system prompt prohíbe añadir información que no esté en ninguna de las dos fuentes. Si las fuentes no cubren la pregunta, OpenAI devuelve el mensaje de fallback.
- **Un solo umbral:** si TF-IDF ≥ 0.15 se llama a OpenAI; por debajo se muestra el fallback directamente. OpenAI actúa como árbitro semántico de si el contexto es suficiente — ya no hay un segundo umbral artificial.
- **Confianza mostrada:** siempre el score TF-IDF (OpenAI no devuelve puntuación numérica). La etiqueta indica "OpenAI + TF-IDF" cuando OpenAI generó la respuesta, "TF-IDF" cuando se usó la pre-escrita.
- Las categorías de consulta son: **Síntomas**, **Medicamentos**, **Protocolos**.
- El historial muestra las últimas 20 consultas y permite exportar a CSV.
- La red del hospital puede bloquear el puerto de Railway — usar hotspot o red doméstica si hay problemas de conexión.

## Comportamientos clave de la UI

- El campo de texto acepta **Enter** para lanzar la consulta (sin necesidad de hacer clic en "Consultar").
- El botón "Nueva consulta" limpia los inputs y refresca la pantalla.
- El botón "Mejorar" aparece cuando la confianza es < 85% y permite añadir una entrada al dataset en tiempo real.
- El índice TF-IDF se reconstruye automáticamente tras añadir una entrada (sin reiniciar la app).

## Estado actual del proyecto (mayo 2026)

**Fase:** implementación completa y desplegada en producción. Pendiente únicamente la documentación final.

**Completado:**
- Pipeline completo funcional: voz → Google SR → TF-IDF (doble índice) → OpenAI GPT-4o-mini RAG → MySQL Railway
- App Streamlit Cloud desplegada y pública
- MySQL Railway conectado y operativo
- Dataset 500 entradas cardiología (IDs 1–500), 3 categorías
- Correcciones fonéticas (~50 términos cardiológicos) en `src/speech.py`
- OpenAI RAG estricto: respuestas generadas solo con el contexto recuperado, sin alucinaciones
- 39 entradas de dosis enriquecidas con contextos clínicos completos (guías ESC)
- Script `scripts/reimportar_contextos.py` para actualizar el dataset por lotes
- Enter en campo de texto lanza la consulta directamente
- Memoria borrador completa en `docs/MEMORIA_BORRADOR.md`

**Pendiente para la entrega:**
- Añadir `OPENAI_API_KEY` a los secrets de Streamlit Cloud
- Pruebas con voz real (mínimo 10 consultas) y rellenar sección 7.1–7.2 de la memoria
- Capturas de pantalla de cada pestaña + MySQL Railway (sección 7.3)
- Reflexión personal en conclusiones (3–4 líneas propias)
- Convertir `MEMORIA_BORRADOR.md` a Word o pptx con formato UEM

## Stack técnico

- Python 3.9
- `streamlit` ≥ 1.32 — interfaz web con grabación de audio
- `speech_recognition` ≥ 3.10 — Google Speech Recognition
- `pydub` ≥ 0.25 — conversión WebM → WAV (requiere `ffmpeg` en el sistema)
- `scikit-learn` ≥ 1.3 — TF-IDF y similitud coseno
- `openai` ≥ 1.30 — cliente GPT-4o-mini para RAG
- `mysql-connector-python` — conexión a MySQL
- `pandas` — manejo del dataset e historial
- `python-dotenv` — gestión de variables de entorno
- MySQL 9.4 en Railway — base de datos relacional remota

## Reglas de mantenimiento (heredadas del global)

- **No tocar lo que ya funciona.** Cambios mínimos por bug, no mezclar refactors con fixes.
- **Diagnosticar antes de optimizar.** Medir antes de actuar.
- **Interfaz en español.** Tildes obligatorias, sin mezcla de idiomas en la UI.

## Notas específicas del proyecto

- Es una entrega individual — el código debe ser original, no copiado de internet ni de compañeros.
- La documentación final va en formato pptx o Word e incluye obligatoriamente referencias al estado del arte.
- No hardcodear respuestas médicas concretas — solo mejoras algorítmicas reales.
- Este es un proyecto académico: priorizar claridad y simplicidad sobre optimización prematura.
- El HF_TOKEN debe regenerarse en HuggingFace antes de hacer el repo público (el token anterior quedó expuesto en el historial del chat).
