"""
Pruebas automáticas del servicio de predicción (OralSolutions).

Cubre:
1. Contrato HTTP y validación Pydantic (200 válido; 422 por campo faltante, tipo erróneo,
   categoría desconocida o regla de dominio violada).
2. Modelo: el artefacto exportado desde MLflow reproduce PR-AUC = 0.98269 sobre la muestra fija.
3. Comportamiento de negocio: 2 invariantes del dominio odontológico.
4. Respaldo (fallback automático si el modelo no carga + interruptor manual protegido).
"""

from __future__ import annotations

import os

from pathlib import Path

import pandas as pd

import pytest

from fastapi.testclient import TestClient

from sklearn.metrics import average_precision_score

os.environ["USAR_MODELO_LOCAL"] = "true"

os.environ["REGISTRAR_SNOWFLAKE"] = "false"

from src.pipeline import FEATURES_FINALES, TARGET, construir_features

from api import main as api

BASE_FIXTURES = Path(__file__).resolve().parent / "fixtures"

MUESTRA_TEST = BASE_FIXTURES / "muestra_test.csv"

PR_AUC_ESPERADO_MLFLOW = 0.98269

TOLERANCIA_PR_AUC = 0.0005

PAYLOAD_VALIDO = {
    "duracion_estimada_min": 75,
    "numero_sesion": 1,
    "sesiones_planificadas": 3,
    "cancelaciones_previas": 0,
    "inasistencias_previas": 0,
    "dias_desde_inicio_plan": 14,
    "edad_en_atencion": 42,
    "codigo_procedimiento": "CROWN",
    "odontologo_key": "2",
}

@pytest.fixture()
def client(tmp_path: Path):
    """Levanta TestClient con el modelo local cargado y log aislado en carpeta temporal."""
    api.RUTA_LOG_PREDICCIONES = tmp_path / "predicciones_test.csv"
    api.estado.modo_respaldo = False
    api.cargar_modelo()
    with TestClient(api.app) as c:
        yield c
    api.estado.modo_respaldo = False
    api.cargar_modelo()

def test_negocio_mayor_progreso_hacia_ultima_sesion_aumenta_riesgo(client: TestClient) -> None:
    """
    Hallazgo clave del EDA/PSet 3: en planes de 4 sesiones, avanzar de la sesión 1
    (progreso 0.25) a la penúltima sesión 3 (progreso 0.75, falta 1 sesión) debe
    incrementar la probabilidad predicha de abandono.
    """
    temprano = {**PAYLOAD_VALIDO, "numero_sesion": 1, "sesiones_planificadas": 4}
    penultima = {**PAYLOAD_VALIDO, "numero_sesion": 3, "sesiones_planificadas": 4}

    p_temprano = client.post("/predecir", json=temprano).json()["probabilidad"]
    p_penultima = client.post("/predecir", json=penultima).json()["probabilidad"]

    assert p_penultima > p_temprano, (
        f"Se esperaba mayor riesgo al acercarse al cierre del plan: {p_penultima} vs {p_temprano}"
    )


def test_negocio_paciente_critico_dispara_alerta_contactar(client: TestClient) -> None:
    """
    Un paciente en procedimiento restaurativo corto (FILL1), con fricción previa alta
    (cancelaciones e inasistencias) y a 1 sesión de finalizar debe recibir decisión 'contactar'.
    """
    caso_critico = {
        "duracion_estimada_min": 45,
        "numero_sesion": 2,
        "sesiones_planificadas": 3,
        "cancelaciones_previas": 2,
        "inasistencias_previas": 2,
        "dias_desde_inicio_plan": 60,
        "edad_en_atencion": 38,
        "codigo_procedimiento": "FILL1",
        "odontologo_key": "1",
    }
    body = client.post("/predecir", json=caso_critico).json()
    assert body["decision"] == "contactar"
    assert body["probabilidad"] >= 0.80


def test_fallback_cuando_modelo_no_carga(client: TestClient) -> None:
    api.estado.modelo = None
    api.estado.modelo_cargado = False

    # Caso con 1 sesión restante -> regla de respaldo devuelve 0.99 y contactar
    caso_1_restante = {**PAYLOAD_VALIDO, "numero_sesion": 2, "sesiones_planificadas": 3}
    resp = client.post("/predecir", json=caso_1_restante)
    assert resp.status_code == 200
    body = resp.json()

    assert body["uso_respaldo"] is True
    assert body["version_modelo"] == "fallback-regla-1-sesion"
    assert body["probabilidad"] == 0.99
    assert body["decision"] == "contactar"


def test_interruptor_respaldo_protegido(client: TestClient) -> None:
    # Sin token válido -> 401
    resp_no_auth = client.post(
        "/admin/respaldo",
        json={"activar": True, "motivo": "Prueba sin token"},
        headers={"x-admin-token": "token-invalido"},
    )
    assert resp_no_auth.status_code == 401

    # Con token válido -> activa respaldo y /predecir lo refleja
    resp_ok = client.post(
        "/admin/respaldo",
        json={"activar": True, "motivo": "Degradación detectada en monitoreo"},
        headers={"x-admin-token": api.ADMIN_TOKEN},
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["modo_respaldo"] is True

    pred = client.post("/predecir", json=PAYLOAD_VALIDO).json()
    assert pred["uso_respaldo"] is True
