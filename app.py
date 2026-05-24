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
st.caption("Desarrollado por: Dr. Carlos CARDOZO · Cardiólogo · Exp. 225K3417")

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

    # Inicializar estado persistente entre reruns
    if "ultima_consulta" not in st.session_state:
        st.session_state["ultima_consulta"] = None
    if "input_key" not in st.session_state:
        st.session_state["input_key"] = 0

    # Determinar fuente de la consulta
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

    # Procesar nueva consulta, limpiar inputs y refrescar
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

    # Mostrar el último resultado (persiste aunque el rerun sea por el botón Mejorar)
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

        # — Explicación del resultado —
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
                st.warning("Coincidencia media — reformular la pregunta con términos más específicos puede mejorar el resultado.")
            else:
                st.error("Coincidencia baja — la pregunta se aleja del dataset y la respuesta puede no ser precisa.")


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
                st.success("No hay consultas con baja confianza. El sistema responde bien.")
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
            "Contexto completo (contexto clínico completo (OpenAI lo usa para generar la respuesta))", height=120
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
        column_config={
            "id":        st.column_config.NumberColumn("ID",        width="small"),
            "categoria": st.column_config.TextColumn("Categoría",  width="small"),
            "pregunta":  st.column_config.TextColumn("Pregunta",   width="medium"),
            "respuesta": st.column_config.TextColumn("Respuesta",  width="medium"),
            "contexto":  st.column_config.TextColumn("Contexto",   width="large"),
        },
    )

    st.divider()
    st.caption(
        "**Fuente del dataset inicial:** preguntas y contextos clínicos elaborados a partir de las "
        "Guías de Práctica Clínica de la Sociedad Europea de Cardiología (ESC), ediciones 2021-2023 "
        "(síndrome coronario agudo, insuficiencia cardíaca, fibrilación auricular, hipertensión arterial "
        "y resucitación cardiopulmonar). El contenido ha sido estructurado y adaptado con fines "
        "exclusivamente académicos."
    )
