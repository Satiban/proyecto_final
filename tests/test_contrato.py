"""
Pruebas del contrato HTTP de la API OralSolutions.

Verifica:
1. Predicción válida: HTTP 200 y estructura de respuesta.
2. Campo obligatorio faltante: HTTP 422.
3. Tipo de dato incorrecto: HTTP 422.
4. Procedimiento desconocido: HTTP 422.
5. Plan sin sesiones pendientes: HTTP 422.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Configuración de pruebas independientes de servicios externos
os.environ["USAR_MODELO_LOCAL"] = "true"
os.environ["REGISTRAR_SNOWFLAKE"] = "false"

from api import main as api


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
    """Carga el champion local y aísla los registros de prueba."""

    api.RUTA_LOG_PREDICCIONES = tmp_path / "predicciones_test.csv"
    api.estado.modo_respaldo = False

    assert api.cargar_modelo(), "No se pudo cargar el champion local."

    with TestClient(api.app) as c:
        yield c

    api.estado.modo_respaldo = False


def test_contrato_entrada_valida_devuelve_200(client: TestClient) -> None:
    """Comprueba respuesta válida y metadata del champion."""

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
    assert body["version_modelo"] == "1"
    assert body["run_id"] == "c90a1c37aef940278afc025b1d22dc6b"
    assert body["umbral_aplicado"] == pytest.approx(0.01)

    # La decisión debe corresponder a la probabilidad y al umbral.
    decision_esperada = (
        "contactar"
        if body["probabilidad"] >= body["umbral_aplicado"]
        else "no_contactar"
    )

    assert body["decision"] == decision_esperada


def test_contrato_campo_faltante_devuelve_422(client: TestClient) -> None:
    """Un campo obligatorio faltante debe rechazarse."""

    incompleto = {
        k: v
        for k, v in PAYLOAD_VALIDO.items()
        if k != "edad_en_atencion"
    }

    resp = client.post("/predecir", json=incompleto)

    assert resp.status_code == 422
    assert "edad_en_atencion" in resp.text


def test_contrato_tipo_incorrecto_devuelve_422(client: TestClient) -> None:
    """Un valor numérico inválido debe rechazarse."""

    tipo_malo = {
        **PAYLOAD_VALIDO,
        "duracion_estimada_min": "sesenta_minutos",
    }

    resp = client.post("/predecir", json=tipo_malo)

    assert resp.status_code == 422
    assert "duracion_estimada_min" in resp.text


def test_contrato_categoria_desconocida_devuelve_422(
    client: TestClient,
) -> None:
    """Un procedimiento fuera del catálogo debe rechazarse."""

    cat_invalida = {
        **PAYLOAD_VALIDO,
        "codigo_procedimiento": "IMPLANTE_DESCONOCIDO",
    }

    resp = client.post("/predecir", json=cat_invalida)

    assert resp.status_code == 422
    assert "codigo_procedimiento" in resp.text


def test_contrato_regla_sesiones_invalida_devuelve_422(
    client: TestClient,
) -> None:
    """No se permite predecir cuando el plan está completado."""

    sin_pendientes = {
        **PAYLOAD_VALIDO,
        "numero_sesion": 3,
        "sesiones_planificadas": 3,
    }

    resp = client.post("/predecir", json=sin_pendientes)

    assert resp.status_code == 422


def test_contrato_tipo_estricto_devuelve_422(
    client: TestClient,
) -> None:
    """Una cadena numérica no debe aceptarse como entero."""

    payload = {
        **PAYLOAD_VALIDO,
        "duracion_estimada_min": "75",
    }

    resp = client.post("/predecir", json=payload)

    assert resp.status_code == 422
    assert "duracion_estimada_min" in resp.text


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("duracion_estimada_min", -10),
        ("numero_sesion", 0),
        ("sesiones_planificadas", 1),
        ("cancelaciones_previas", -1),
        ("inasistencias_previas", -1),
        ("dias_desde_inicio_plan", -5),
        ("edad_en_atencion", 150),
    ],
)
def test_contrato_valor_fuera_rango_devuelve_422(
    client: TestClient,
    campo: str,
    valor: int,
) -> None:
    """Los valores fuera de los rangos válidos se rechazan."""

    payload = {
        **PAYLOAD_VALIDO,
        campo: valor,
    }

    resp = client.post("/predecir", json=payload)

    assert resp.status_code == 422
    assert campo in resp.text


def test_contrato_campo_extra_devuelve_422(
    client: TestClient,
) -> None:
    """La API no debe aceptar el target como entrada."""

    payload = {
        **PAYLOAD_VALIDO,
        "abandono_60d": 1,
    }

    resp = client.post("/predecir", json=payload)

    assert resp.status_code == 422
    assert "abandono_60d" in resp.text