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

def test_contrato_entrada_valida_devuelve_200(client: TestClient) -> None:
    resp = client.post("/predecir", json=PAYLOAD_VALIDO)
    assert resp.status_code == 200

    body = resp.json()
    for campo in (
        "id_prediccion",
        "probabilidad",
        "decision",
        "accion",
        "umbral_aplicado",
        "version_modelo",
        "run_id",
        "uso_respaldo",
        "latencia_ms",
        "timestamp",
    ):
        assert campo in body

    assert 0.0 <= body["probabilidad"] <= 1.0
    assert body["decision"] in {"contactar", "no_contactar"}
    assert body["uso_respaldo"] is False
    assert body["version_modelo"] == "7"


def test_contrato_campo_faltante_devuelve_422(client: TestClient) -> None:
    incompleto = {k: v for k, v in PAYLOAD_VALIDO.items() if k != "edad_en_atencion"}
    resp = client.post("/predecir", json=incompleto)
    assert resp.status_code == 422
    assert "edad_en_atencion" in resp.text


def test_contrato_tipo_incorrecto_devuelve_422(client: TestClient) -> None:
    tipo_malo = {**PAYLOAD_VALIDO, "duracion_estimada_min": "sesenta_minutos"}
    resp = client.post("/predecir", json=tipo_malo)
    assert resp.status_code == 422
    assert "duracion_estimada_min" in resp.text


def test_contrato_categoria_desconocida_devuelve_422(client: TestClient) -> None:
    cat_invalida = {**PAYLOAD_VALIDO, "codigo_procedimiento": "IMPLANTE_DESCONOCIDO"}
    resp = client.post("/predecir", json=cat_invalida)
    assert resp.status_code == 422
    assert "codigo_procedimiento" in resp.text


def test_contrato_regla_sesiones_invalida_devuelve_422(client: TestClient) -> None:
    # No se permite predecir cuando el plan ya llegó a la última sesión
    sin_pendientes = {**PAYLOAD_VALIDO, "numero_sesion": 3, "sesiones_planificadas": 3}
    resp = client.post("/predecir", json=sin_pendientes)
    assert resp.status_code == 422

