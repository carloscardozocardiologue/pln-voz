"""
Asistente Virtual de Voz para Cardiología
Máster IA en Ciencias de la Salud — Universidad Europea de Madrid
PLN Unidad 4 — Actividad AA1
"""

import time
import pandas as pd
import streamlit as st
from src.speech import transcribir
from src.nlp import responder
from src.db import guardar, obtener_historial

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente Virtual · Cardiología",
    page_icon="🫀",
    layout="centered",
)

# ── Cabecera ─────────────────────────────────────────────────────────────────
st.title("🫀 Asistente Virtual de Voz")
st.caption("Cardiología · Universidad Europea de Madrid")
st.divider()

# ── Selector de categoría ────────────────────────────────────────────────────
categoria = st.radio(
    "Categoría de consulta",
    options=["Síntomas", "Medicamentos", "Protocolos"],
    horizontal=True,
)

st.write("")

# ── Grabación de voz ─────────────────────────────────────────────────────────
st.subheader("Grabación de la consulta")
audio = st.audio_input("Pulsa el micrófono, habla y pulsa detener")

if audio:
    with st.spinner("Transcribiendo audio..."):
        audio_bytes = audio.read()
        try:
            transcripcion = transcribir(audio_bytes)
        except Exception as e:
            st.error(f"Error en la transcripción: {e}")
            st.stop()

    st.success("Transcripción completada")
    st.markdown(f"**Médico:** _{transcripcion}_")

    # ── Procesamiento PLN ─────────────────────────────────────────────────
    with st.spinner("Procesando con el modelo de lenguaje..."):
        inicio = time.time()
        try:
            resultado = responder(transcripcion, categoria)
        except Exception as e:
            st.error(f"Error en el modelo PLN: {e}")
            st.stop()
        duracion_ms = int((time.time() - inicio) * 1000)

    st.markdown(f"**Asistente:** {resultado['respuesta']}")

    confianza = resultado["confianza"]
    color = "green" if confianza >= 0.7 else "orange" if confianza >= 0.4 else "red"
    st.markdown(
        f"Confianza del modelo: "
        f"<span style='color:{color}; font-weight:bold'>{confianza:.0%}</span>",
        unsafe_allow_html=True,
    )
    st.progress(confianza)

    # ── Guardado en MySQL ─────────────────────────────────────────────────
    try:
        guardar(categoria, transcripcion, resultado["respuesta"],
                confianza, duracion_ms)
    except Exception as e:
        st.warning(f"No se pudo guardar en la base de datos: {e}")

st.divider()

# ── Historial de consultas ────────────────────────────────────────────────────
st.subheader("Historial de consultas")

try:
    historial = obtener_historial(20)
    if historial:
        df = pd.DataFrame(historial)
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]).dt.strftime("%d/%m/%Y %H:%M")
        df["confianza"] = df["confianza"].apply(
            lambda x: f"{x:.0%}" if x is not None else "-"
        )
        df.columns = ["Fecha y hora", "Categoría", "Transcripción", "Respuesta", "Confianza"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exportar historial CSV",
            data=csv,
            file_name="historial_consultas.csv",
            mime="text/csv",
        )
    else:
        st.info("Aún no hay consultas registradas.")
except Exception as e:
    st.warning(f"No se pudo cargar el historial: {e}")
