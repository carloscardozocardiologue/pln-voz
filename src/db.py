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
