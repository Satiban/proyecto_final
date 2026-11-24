"""
API de servicio de predicción de abandono de tratamientos (OralSolutions).

Endpoints principales:
- GET  /salud          : Estado del servicio y carga del modelo.
- GET  /modelo         : Nombre, versión, alias y run_id del modelo activo (detecta rollback sin reconstruir).
- POST /predecir       : Valida entrada, construye features desde src/pipeline.py, predice y registra.
- GET  /metricas       : Métricas en vivo de sistema (latencia p50/p95, errores 4xx/5xx) y de modelo.
- POST /admin/respaldo : Interruptor protegido para activar/desactivar el modo de respaldo (regla de negocio).
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import math
import secrets
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import RLock

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
BLOQUEO_MODELO = RLock()

# Permitir importar src/pipeline.py sin copiar código
BASE_SERVICIO = Path(__file__).resolve().parent
BASE_PROYECTO = BASE_SERVICIO.parent
if str(BASE_PROYECTO) not in sys.path:
    sys.path.insert(0, str(BASE_PROYECTO))

from src.pipeline import FEATURES_FINALES, construir_features
from api.esquemas import (
    RespuestaModelo,
    RespuestaPrediccion,
    RespuestaSalud,
    SolicitudPrediccion,
    SolicitudRespaldo,
)

# ---------------------------------------------------------------------------
# Configuración por variables de entorno
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "oralsolutions-modelo")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
UMBRAL_DECISION = float(os.getenv("UMBRAL_DECISION", "0.01"))
UMBRAL_RESPALDO = 0.5
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

USAR_MODELO_LOCAL = os.getenv("USAR_MODELO_LOCAL", "false").lower() == "true"
RUTA_MODELO_LOCAL = Path(
    os.getenv("RUTA_MODELO_LOCAL", str(BASE_PROYECTO / "tests" / "fixtures" / "modelo_champion"))
)
RUTA_META_LOCAL = Path(
    os.getenv("RUTA_META_LOCAL", str(BASE_PROYECTO / "tests" / "fixtures" / "modelo_meta.json"))
)
REGISTRAR_SNOWFLAKE = os.getenv("REGISTRAR_SNOWFLAKE", "false").lower() == "true"
RUTA_LOG_PREDICCIONES = Path(
    os.getenv("RUTA_LOG_PREDICCIONES", str(BASE_SERVICIO / "predicciones_log.csv"))
)


# ---------------------------------------------------------------------------
# Estado global del servicio y métricas en memoria
# ---------------------------------------------------------------------------
class EstadoServicio:
    def __init__(self) -> None:
        self.modelo: Any = None
        self.modelo_cargado: bool = False
        self.nombre_modelo: str = MODEL_NAME
        self.alias: str = MODEL_ALIAS
        self.version: str = "desconocida"
        self.run_id: str = "desconocido"
        self.umbral_decision: float = UMBRAL_DECISION
        self.hash_dataset: str = ""
        self.fecha_corte: str = ""
        self.ultimo_error_modelo: str = ""
        self.modo_respaldo: bool = os.getenv("MODO_RESPALDO", "false").lower() == "true"
        self.motivo_respaldo: str = "Variable de entorno MODO_RESPALDO" if self.modo_respaldo else ""

        # Métricas de sistema y de modelo
        self.latencias_ms: deque[float] = deque(maxlen=5000)
        self.probabilidades: deque[float] = deque(maxlen=5000)
        self.total_solicitudes: int = 0
        self.errores_4xx: int = 0
        self.errores_5xx: int = 0
        self.total_contactar: int = 0
        self.total_no_contactar: int = 0
        self.inicio_servicio: str = datetime.now(timezone.utc).isoformat()


estado = EstadoServicio()


def cargar_modelo(forzar_mlflow: bool = False) -> bool:
    """Resuelve el alias y carga una versión concreta; usa fixture en CI."""
    with BLOQUEO_MODELO:
        try:
            if USAR_MODELO_LOCAL and not forzar_mlflow:
                meta = json.loads(RUTA_META_LOCAL.read_text(encoding="utf-8"))
                modelo = mlflow.sklearn.load_model(str(RUTA_MODELO_LOCAL))
                nombre = meta["nombre_modelo"]
                alias = meta["alias"]
                version = str(meta["version"])
                run_id = meta["run_id"]
                umbral = float(meta["umbral_decision"])
                hash_dataset = meta.get("hash_dataset", "")
                fecha_corte = meta.get("fecha_corte", "")
            else:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
                version = str(mv.version)
                nombre, alias, run_id = MODEL_NAME, MODEL_ALIAS, mv.run_id
                run = client.get_run(run_id)
                umbral = float(mv.tags.get("umbral_decision", run.data.params.get("umbral_negocio")))
                hash_dataset = run.data.params.get("hash_dataset", "")
                fecha_corte = run.data.params.get("fecha_corte", "")
                if (estado.modelo_cargado and estado.modelo is not None
                    and estado.version == version and estado.run_id == run_id
                    and estado.nombre_modelo == nombre):
                    modelo = estado.modelo
                else:
                    modelo = mlflow.sklearn.load_model(f"models:/{nombre}/{version}")

            if not math.isfinite(umbral) or not 0 <= umbral <= 1:
                raise ValueError("Umbral inválido")
            if list(modelo.feature_names_in_) != FEATURES_FINALES:
                raise ValueError("Features incompatibles con src/pipeline.py")
            if list(modelo.classes_) != [0, 1] or not hasattr(modelo, "predict_proba"):
                raise ValueError("Clasificador incompatible")

            estado.modelo = modelo
            estado.modelo_cargado = True
            estado.nombre_modelo = nombre
            estado.alias = alias
            estado.version = version
            estado.run_id = run_id
            estado.umbral_decision = umbral
            estado.hash_dataset = hash_dataset
            estado.fecha_corte = fecha_corte
            estado.ultimo_error_modelo = ""
            return True
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo; se utilizará el respaldo")
            estado.modelo = None
            estado.modelo_cargado = False
            estado.version = "fallback-regla-negocio"
            estado.run_id = "fallback"
            estado.umbral_decision = UMBRAL_RESPALDO
            estado.hash_dataset = ""
            estado.fecha_corte = ""
            estado.ultimo_error_modelo = type(exc).__name__
            return False


def predecir_regla_respaldo(df_features: pd.DataFrame) -> float:
    """
    Línea base de negocio del PSet 3 (`Regla1SesionRestante`):
    Si al paciente le queda exactamente 1 sesión pendiente, riesgo alto (0.99);
    en caso contrario riesgo base bajo (0.13).
    """
    sesiones_restantes = int(
        df_features["sesiones_planificadas"].iloc[0] - df_features["numero_sesion"].iloc[0]
    )
    return 0.99 if sesiones_restantes == 1 else 0.13


def registrar_prediccion(registro: dict[str, Any]) -> None:
    """Registra la predicción localmente en CSV y opcionalmente en Snowflake (PREDICCIONES)."""
    existe = RUTA_LOG_PREDICCIONES.exists()
    try:
        RUTA_LOG_PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
        with RUTA_LOG_PREDICCIONES.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(registro.keys()))
            if not existe:
                writer.writeheader()
            writer.writerow(registro)
    except Exception:
        logger.exception(
            "Error al registrar predicción en CSV. "
            "ID de predicción: %s",
            registro["id_prediccion"]
        )

    if not REGISTRAR_SNOWFLAKE:
        return

    try:
        import snowflake.connector

        pwd_b64 = os.environ.get("SECRET_SNOWFLAKE_PASSWORD", "")
        password = base64.b64decode(pwd_b64, validate=True).decode("utf-8")
        conn = snowflake.connector.connect(
            account=os.environ["SF_ACCOUNT"],
            user=os.environ["SF_USER"],
            password=password,
            warehouse=os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
            database=os.getenv("SF_DATABASE", "ORALSOLUTIONS_DW"),
            schema=os.getenv("SF_MLOPS_SCHEMA", "MLOPS"),
            role=os.getenv("SF_ROLE", "ACCOUNTADMIN"),
        )
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO PREDICCIONES (
                    ID_PREDICCION, TIMESTAMP_UTC, ENTRADAS_JSON, PROBABILIDAD,
                    DECISION, VERSION_MODELO, RUN_ID, USO_RESPALDO, LATENCIA_MS
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    registro["id_prediccion"],
                    registro["timestamp"],
                    registro["entradas_json"],
                    registro["probabilidad"],
                    registro["decision"],
                    registro["version_modelo"],
                    registro["run_id"],
                    registro["uso_respaldo"],
                    registro["latencia_ms"],
                ),
            )
        finally:
            conn.close()
    except Exception:
        logger.exception(
            "Error al registrar predicción en Snowflake. "
            "ID de predicción: %s",
            registro["id_prediccion"]
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="OralSolutions — API de Predicción de Abandono (60 días)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def medir_metricas_http(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duracion_ms = (time.perf_counter() - t0) * 1000.0

    estado.total_solicitudes += 1
    estado.latencias_ms.append(duracion_ms)
    if 400 <= response.status_code < 500:
        estado.errores_4xx += 1
    elif response.status_code >= 500:
        estado.errores_5xx += 1

    return response


@app.get("/salud", response_model=RespuestaSalud)
def salud() -> RespuestaSalud:
    return RespuestaSalud(
        estado="ok" if (estado.modelo_cargado or estado.modo_respaldo) else "degradado_respaldo",
        modelo_cargado=estado.modelo_cargado,
        modo_respaldo=estado.modo_respaldo or (not estado.modelo_cargado),
        nombre_modelo=estado.nombre_modelo,
        version_modelo=estado.version,
    )


@app.get("/modelo", response_model=RespuestaModelo)
def info_modelo() -> RespuestaModelo:
    with BLOQUEO_MODELO:
        if not USAR_MODELO_LOCAL:
            cargar_modelo()
        return RespuestaModelo(
            nombre_modelo=estado.nombre_modelo,
            version=estado.version,
            alias=estado.alias,
            run_id=estado.run_id,
            umbral_decision=estado.umbral_decision,
            modo_respaldo=estado.modo_respaldo or not estado.modelo_cargado,
        )


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    t0 = time.perf_counter()
    id_pred = str(uuid.uuid4())
    ts_utc = datetime.now(timezone.utc).isoformat()

    # 1. Construir features importando directamente src/pipeline.py
    datos_entrada = solicitud.model_dump()
    df_raw = pd.DataFrame([datos_entrada])
    df_feat = construir_features(df_raw)[FEATURES_FINALES]

    # 2. Resolver champion y capturar metadatos coherentes con la predicción.
    with BLOQUEO_MODELO:
        if not USAR_MODELO_LOCAL and not estado.modo_respaldo:
            cargar_modelo()
        usar_respaldo = estado.modo_respaldo or not estado.modelo_cargado or estado.modelo is None
        if usar_respaldo:
            prob = predecir_regla_respaldo(df_feat)
            version_usada = "fallback-regla-1-sesion"
            run_usado = "fallback"
            umbral_aplicado = UMBRAL_RESPALDO
        else:
            prob = float(estado.modelo.predict_proba(df_feat)[:, 1][0])
            version_usada = estado.version
            run_usado = estado.run_id
            umbral_aplicado = estado.umbral_decision
        decision = "contactar" if prob >= umbral_aplicado else "no_contactar"
    accion = (
        "Priorizar recordatorio personalizado y seguimiento preventivo de recepción"
        if decision == "contactar"
        else "Mantener seguimiento administrativo estándar"
    )

    latencia_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # 3. Actualizar métricas en memoria
    estado.probabilidades.append(prob)
    if decision == "contactar":
        estado.total_contactar += 1
    else:
        estado.total_no_contactar += 1

    # 4. Registrar predicción para trazabilidad y auditoría
    registrar_prediccion(
        {
            "id_prediccion": id_pred,
            "timestamp": ts_utc,
            "entradas_json": json.dumps(datos_entrada, ensure_ascii=False),
            "probabilidad": prob,
            "decision": decision,
            "version_modelo": version_usada,
            "run_id": run_usado,
            "uso_respaldo": usar_respaldo,
            "latencia_ms": latencia_ms,
        }
    )

    return RespuestaPrediccion(
        id_prediccion=id_pred,
        probabilidad=prob,
        decision=decision,
        accion=accion,
        umbral_aplicado=umbral_aplicado,
        version_modelo=version_usada,
        run_id=run_usado,
        uso_respaldo=usar_respaldo,
        latencia_ms=latencia_ms,
        timestamp=ts_utc,
    )

def evaluar_alertas_servicio() -> list[dict[str, Any]]:
    """Evalúa alertas operativas y de deriva de predicciones en vivo dentro del servicio."""
    alertas: list[dict[str, Any]] = []
    latencias = list(estado.latencias_ms)
    total_preds = estado.total_contactar + estado.total_no_contactar
    tasa_err = (
        (estado.errores_4xx + estado.errores_5xx) / estado.total_solicitudes
        if estado.total_solicitudes > 0
        else 0.0
    )
    p95_ms = float(np.percentile(latencias, 95)) if latencias else 0.0
    tasa_contactar = estado.total_contactar / total_preds if total_preds > 0 else 0.0

    if estado.modo_respaldo or (not estado.modelo_cargado):
        alertas.append({
            "tipo": "respaldo_activo",
            "severidad": "Alta",
            "mensaje": "Servicio operando en modo respaldo (regla de negocio) en lugar del modelo ML.",
        })
    if estado.total_solicitudes >= 10 and tasa_err > 0.05:
        alertas.append({
            "tipo": "tasa_errores_http",
            "severidad": "Alta",
            "mensaje": f"Tasa de errores HTTP elevada ({tasa_err:.1%} > 5%).",
        })
    if len(latencias) >= 10 and p95_ms > 250.0:
        alertas.append({
            "tipo": "latencia_alta",
            "severidad": "Media",
            "mensaje": f"Latencia p95 elevada ({p95_ms:.1f} ms > 250 ms).",
        })
    if total_preds >= 20 and (tasa_contactar < 0.65 or tasa_contactar > 0.98):
        alertas.append({
            "tipo": "deriva_predicciones",
            "severidad": "Alta",
            "mensaje": f"Tasa de decisión 'contactar' fuera de banda esperada ({tasa_contactar:.1%}).",
        })
    return alertas

@app.get("/metricas")
def metricas() -> dict[str, Any]:
    latencias = list(estado.latencias_ms)
    probs = list(estado.probabilidades)
    total_preds = estado.total_contactar + estado.total_no_contactar
    alertas_activas = evaluar_alertas_servicio()

    return {
        "inicio_servicio": estado.inicio_servicio,
        "sistema": {
            "volumen_solicitudes": estado.total_solicitudes,
            "errores_4xx": estado.errores_4xx,
            "errores_5xx": estado.errores_5xx,
            "tasa_error": round(
                (estado.errores_4xx + estado.errores_5xx) / estado.total_solicitudes, 4
            )
            if estado.total_solicitudes > 0
            else 0.0,
            "latencia_p50_ms": round(float(np.percentile(latencias, 50)), 3) if latencias else 0.0,
            "latencia_p95_ms": round(float(np.percentile(latencias, 95)), 3) if latencias else 0.0,
        },
        "modelo": {
            "version_activa": estado.version,
            "modo_respaldo": estado.modo_respaldo or (not estado.modelo_cargado),
            "total_predicciones": total_preds,
            "tasa_contactar": round(estado.total_contactar / total_preds, 4) if total_preds > 0 else 0.0,
            "probabilidad_promedio": round(float(np.mean(probs)), 4) if probs else 0.0,
        },
        "alertas_activas": alertas_activas,
    }

@app.get("/alertas")
def obtener_alertas() -> dict[str, Any]:
    alertas = evaluar_alertas_servicio()
    return {"total_alertas": len(alertas), "alertas": alertas}


@app.post("/admin/respaldo")
def cambiar_respaldo(
    payload: SolicitudRespaldo,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Interruptor administrativo no configurado.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token.encode("utf-8"), ADMIN_TOKEN.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Token de administrador inválido.")

    with BLOQUEO_MODELO:
        estado.modo_respaldo = payload.activar
        estado.motivo_respaldo = payload.motivo
        respaldo_efectivo = estado.modo_respaldo or not estado.modelo_cargado
        return {
            "modo_respaldo": estado.modo_respaldo,
            "respaldo_efectivo": respaldo_efectivo,
            "motivo": estado.motivo_respaldo,
            "mensaje": "Modo respaldo activo." if respaldo_efectivo else "Modelo ML activo.",
        }
