# CLAUDE.md — UA4 AA1: Asistente Virtual de Voz para Entornos Hospitalarios

## Resumen del proyecto

Actividad académica de la asignatura PLN (Procesamiento de Lenguaje Natural), Unidad 4, Máster de IA en Ciencias de la Salud — Universidad Europea de Madrid.

El sistema captura peticiones de voz de médicos en un entorno hospitalario, las procesa con modelos de lenguaje natural (HuggingFace) y almacena las consultas y respuestas en una base de datos MySQL.

**Código de actividad:** 0MSR001107_UA4_AA1  
**Producción:** no aplica (entorno local de desarrollo)

## Filosofía y producto final

**Principio fundamental:** el sistema debe funcionar con audio real de micrófono en un entorno hospitalario simulado — no basta con texto sintético como entrada.

Lo que produce el sistema:
- Script Python ejecutable que captura voz, procesa con PLN y guarda en MySQL
- Base de datos MySQL con ejemplos de peticiones y respuestas almacenadas
- Documentación del proceso (pptx o Word): contexto, metodología, herramientas, análisis de resultados, conclusiones y referencias al estado del arte

## Arquitectura

```
Micrófono
    │
    ▼
speech_recognition (SpeechRecognition)
    │  texto transcrito
    ▼
Modelo PLN — HuggingFace Transformers
    │  respuesta generada
    ▼
mysql-connector-python
    │
    ▼
MySQL DB  (tabla: peticiones_respuestas)
```

## Servicios — referencia rápida

| Componente          | Tecnología                       | Carpeta / archivo       | Función                              |
|---------------------|----------------------------------|-------------------------|--------------------------------------|
| Reconocimiento voz  | SpeechRecognition + Google STT   | `src/speech.py`         | Captura audio → texto                |
| Procesamiento PLN   | HuggingFace Transformers         | `src/nlp.py`            | Texto → respuesta médica             |
| Persistencia        | mysql-connector-python           | `src/db.py`             | Guarda consulta + respuesta en MySQL |
| Script principal    | Python                           | `main.py`               | Orquesta el flujo completo           |
| Base de datos       | MySQL                            | schema en `sql/`        | Almacena peticiones y respuestas     |

## Comandos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Crear la base de datos
mysql -u root -p < sql/schema.sql

# Ejecutar el asistente
python main.py

# Verificar datos guardados
mysql -u root -p hospital_db -e "SELECT * FROM peticiones_respuestas LIMIT 10;"
```

## Configuración

| Variable        | Uso                                          |
|-----------------|----------------------------------------------|
| `DB_HOST`       | Host de MySQL (por defecto: localhost)        |
| `DB_USER`       | Usuario MySQL                                 |
| `DB_PASSWORD`   | Contraseña MySQL                             |
| `DB_NAME`       | Nombre de la base de datos (hospital_db)     |
| `HF_MODEL`      | Modelo HuggingFace a usar (por definir)      |

## Comportamientos clave

- La entrada es audio de micrófono real — el modelo debe poder responder a consultas médicas comunes en español.
- El modelo de PLN puede ser de QA (question-answering) o de generación de texto; elegir según el tipo de consulta médica que se quiera simular.
- La base de datos guarda tanto la transcripción de voz como la respuesta generada para tener trazabilidad completa.
- Evaluación académica: originalidad, simplicidad, solución correcta, planificación y documentación. No mezclar idiomas en la interfaz.

## Estado actual del proyecto (mayo 2026)

**Fase:** inicio — entorno preparado, sin código aún.  
**Próximos pasos:** definir arquitectura detallada, instalar dependencias, implementar el pipeline.

## Stack técnico

- Python 3.x
- `speech_recognition` — captura y transcripción de voz
- `transformers` (HuggingFace) — modelo de PLN
- `mysql-connector-python` — conexión a MySQL
- MySQL Server — base de datos relacional

## Reglas de mantenimiento (heredadas del global)

- **No tocar lo que ya funciona.** Cambios mínimos por bug, no mezclar refactors con fixes.
- **Diagnosticar antes de optimizar.** Medir antes de actuar.
- **Interfaz en español.** Tildes obligatorias, sin mezcla de idiomas en la UI.

## Notas específicas del proyecto

- Es una entrega individual — el código debe ser original, no copiado de internet ni de compañeros.
- La documentación final va en formato pptx o Word e incluye obligatoriamente referencias al estado del arte.
- No hardcodear respuestas médicas concretas para subir métricas — solo mejoras algorítmicas reales.
- Este es un proyecto académico: priorizar claridad y simplicidad del código sobre optimización prematura.
