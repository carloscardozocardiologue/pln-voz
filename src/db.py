"""Módulo de persistencia: guarda y recupera consultas en MySQL."""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def _conectar():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        connection_timeout=3,
    )


def guardar(categoria: str, transcripcion: str, respuesta: str,
            confianza: float = None, duracion_ms: int = None):
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
    conn = _conectar()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM consultas")
    conn.commit()
    cursor.close()
    conn.close()


def obtener_estadisticas() -> dict:
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
