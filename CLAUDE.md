# CLAUDE.md — UA4 AA1: Asistente Virtual de Voz para Entornos Hospitalarios

## Resumen del proyecto

Actividad académica de la asignatura PLN (Procesamiento de Lenguaje Natural), Unidad 4, Máster de IA en Ciencias de la Salud — Universidad Europea de Madrid.

El sistema captura peticiones de voz de médicos en un entorno hospitalario cardiológico, las transcribe con Whisper, las procesa con un modelo QA en español (HuggingFace) y almacena las consultas y respuestas en una base de datos MySQL remota (Railway).

**Código de actividad:** 0MSR001107_UA4_AA1  
**Producción:** Streamlit Cloud (URL pública para el profesor)

## Filosofía y producto final

**Principio fundamental:** el sistema debe funcionar con audio real de micrófono — no basta con texto sintético como entrada.

Lo que produce el sistema:
- Aplicación Streamlit con grabación de voz, transcripción automática, respuesta médica y historial
- Base de datos MySQL remota (Railway) con consultas almacenadas
- Documentación del proceso (pptx o Word): contexto, metodología, herramientas, análisis de resultados, conclusiones y referencias al estado del arte

## Arquitectura

```
Navegador del médico
        │
        ▼
   st.audio_input()              ← graba desde el navegador
        │ audio bytes
        ▼
HF API — openai/whisper-large-v3
        │ texto transcrito (es-ES)
        ▼
HF API — BERT QA español         ← extrae respuesta del contexto cardiológico
        │ respuesta + score de confianza
        ▼
MySQL Railway                    ← guarda todo con timestamp
        │
        ▼
Streamlit — transcripción + respuesta + barra de confianza + historial
```

## Servicios — referencia rápida

| Componente         | Tecnología                                               | Archivo                    | Función                               |
|--------------------|----------------------------------------------------------|----------------------------|---------------------------------------|
| Interfaz           | Streamlit ≥ 1.32                                         | `app.py`                   | UI completa con grabación e historial |
| Transcripción voz  | HF API — `openai/whisper-large-v3`                       | `src/speech.py`            | Audio bytes → texto en español        |
| Procesamiento PLN  | HF API — `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` | `src/nlp.py` | Pregunta → respuesta médica + confianza |
| Contexto médico    | Texto cardiológico por categoría                         | `src/contexto_cardio.py`   | Base de conocimiento para el modelo QA |
| Persistencia       | mysql-connector-python                                   | `src/db.py`                | Guarda y recupera consultas en MySQL  |
| Base de datos      | MySQL 9.4 en Railway                                     | `sql/schema.sql`           | Tabla `consultas`                     |

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

| Variable      | Uso                                              |
|---------------|--------------------------------------------------|
| `HF_TOKEN`    | Token de HuggingFace Inference API               |
| `DB_HOST`     | Host MySQL Railway (proxy público)               |
| `DB_PORT`     | Puerto público Railway (≠ 3306 interno)          |
| `DB_USER`     | Usuario MySQL (root)                             |
| `DB_PASSWORD` | Contraseña MySQL Railway                         |
| `DB_NAME`     | Nombre de la base de datos (railway)             |

Copia `.env.example` a `.env` y rellena los valores. El `.env` nunca se sube a git.

## Comportamientos clave

- Las categorías de consulta son: **Síntomas**, **Medicamentos**, **Protocolos** (cardiología).
- El modelo QA extrae la respuesta del contexto cardiológico definido en `src/contexto_cardio.py`.
- La confianza del modelo se muestra como barra de progreso con código de color (verde ≥ 70%, naranja ≥ 40%, rojo < 40%).
- El historial muestra las últimas 20 consultas y permite exportar a CSV.
- La red del hospital bloquea el puerto de Railway — usar hotspot o red doméstica para desarrollo.

## Estado actual del proyecto (mayo 2026)

**Fase:** implementación completa — pipeline funcional en local, pendiente de prueba con voz real y despliegue en Streamlit Cloud.

**Completado:**
- Arquitectura definida y código implementado
- Railway MySQL conectado y tabla `consultas` creada
- Pipeline transcripción → PLN → DB probado con datos de prueba
- Streamlit corriendo en local (http://localhost:8501)

**Pendiente:**
- Prueba con voz real desde el navegador
- Despliegue en Streamlit Cloud
- Hacer el repo público en GitHub
- Crear `Actividad3.md` con documentación para la memoria

## Stack técnico

- Python 3.9
- `streamlit` ≥ 1.32 — interfaz web con grabación de audio
- `huggingface_hub` — cliente para Whisper y BERT QA vía Inference API
- `mysql-connector-python` — conexión a MySQL
- `pandas` — manejo del historial en tabla
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
- El HF_TOKEN está en el chat — regenerarlo en HuggingFace antes de hacer el repo público.
