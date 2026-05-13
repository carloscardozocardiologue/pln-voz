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
