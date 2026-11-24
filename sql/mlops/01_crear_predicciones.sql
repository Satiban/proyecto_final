-- ============================================================
-- OralSolutions - PSet04
-- Creación del esquema operativo MLOps
-- ============================================================

USE DATABASE ORALSOLUTIONS_DW;

CREATE SCHEMA IF NOT EXISTS MLOPS;

USE SCHEMA MLOPS;

-- ============================================================
-- Registro de predicciones
-- ============================================================

CREATE TABLE IF NOT EXISTS PREDICCIONES (

    ID_PREDICCION VARCHAR(36) NOT NULL,

    TIMESTAMP_UTC TIMESTAMP_TZ NOT NULL,

    ENTRADAS_JSON VARCHAR NOT NULL,

    PROBABILIDAD FLOAT NOT NULL,

    DECISION VARCHAR(30) NOT NULL,

    VERSION_MODELO VARCHAR(100) NOT NULL,

    RUN_ID VARCHAR(100) NOT NULL,

    USO_RESPALDO BOOLEAN NOT NULL,

    LATENCIA_MS FLOAT NOT NULL,

    FECHA_REGISTRO TIMESTAMP_TZ
        DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_PREDICCIONES
        PRIMARY KEY (ID_PREDICCION)

);

-- ============================================================
-- Verificación
-- ============================================================

DESCRIBE TABLE ORALSOLUTIONS_DW.MLOPS.PREDICCIONES;

SELECT COUNT(*) AS TOTAL_PREDICCIONES
FROM ORALSOLUTIONS_DW.MLOPS.PREDICCIONES;