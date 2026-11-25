"""
Pruebas de comportamiento de la API OralSolutions.

Verifica:
1. Las probabilidades responden al modelo registrado.
2. Las decisiones respetan el umbral operativo.
3. El respaldo funciona cuando el modelo no está disponible.
4. El interruptor administrativo está protegido.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["USAR_MODELO_LOCAL"] = "true"
os.environ["REGISTRAR_SNOWFLAKE"] = "false"

from api import main as api
from src.pipeline import FEATURES_FINALES, construir_features


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
    """Carga el champion local y aísla el registro de predicciones."""

    api.RUTA_LOG_PREDICCIONES = tmp_path / "predicciones_test.csv"
    api.estado.modo_respaldo = False

    assert api.cargar_modelo(), "No se pudo cargar el champion local."

    with TestClient(api.app) as c:
        yield c

    api.estado.modo_respaldo = False
    api.cargar_modelo()


def test_api_reproduce_probabilidades_del_modelo(
    client: TestClient,
) -> None:
    """
    Verifica que la API reproduzca las probabilidades del modelo
    para dos etapas distintas de un tratamiento.

    No se impone una relación monotónica que el modelo
    no tiene garantizada.
    """

    temprano = {
        **PAYLOAD_VALIDO,
        "numero_sesion": 1,
        "sesiones_planificadas": 4,
    }

    penultima = {
        **PAYLOAD_VALIDO,
        "numero_sesion": 3,
        "sesiones_planificadas": 4,
    }

    for caso in (temprano, penultima):

        respuesta = client.post("/predecir", json=caso)

        assert respuesta.status_code == 200

        body = respuesta.json()

        df = pd.DataFrame([caso])
        X = construir_features(df)[FEATURES_FINALES]

        prob_modelo = float(
            api.estado.modelo.predict_proba(X)[:, 1][0]
        )

        assert body["uso_respaldo"] is False

        assert body["probabilidad"] == pytest.approx(
            round(prob_modelo, 6),
            abs=1e-6,
        )

        assert 0.0 <= body["probabilidad"] <= 1.0


def test_decision_respeta_umbral_operativo(
    client: TestClient,
) -> None:
    """
    Verifica que la decisión de seguimiento corresponda
    a la probabilidad calculada y al umbral operativo.
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

    respuesta = client.post("/predecir", json=caso_critico)

    assert respuesta.status_code == 200

    body = respuesta.json()

    assert body["uso_respaldo"] is False
    assert body["umbral_aplicado"] == pytest.approx(0.01)

    decision_esperada = (
        "contactar"
        if body["probabilidad"] >= body["umbral_aplicado"]
        else "no_contactar"
    )

    assert body["decision"] == decision_esperada
    assert 0.0 <= body["probabilidad"] <= 1.0


def test_fallback_cuando_modelo_no_carga(
    client: TestClient,
) -> None:
    """
    Comprueba que la API utilice la regla de respaldo
    cuando el modelo no está disponible.
    """

    api.estado.modelo = None
    api.estado.modelo_cargado = False

    caso_1_restante = {
        **PAYLOAD_VALIDO,
        "numero_sesion": 2,
        "sesiones_planificadas": 3,
    }

    respuesta = client.post(
        "/predecir",
        json=caso_1_restante,
    )

    assert respuesta.status_code == 200

    body = respuesta.json()

    assert body["uso_respaldo"] is True
    assert body["version_modelo"] == "fallback-regla-1-sesion"
    assert body["probabilidad"] == 0.99
    assert body["decision"] == "contactar"


def test_interruptor_respaldo_protegido(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Comprueba la autenticación del interruptor
    administrativo y su efecto sobre las predicciones.
    """

    monkeypatch.setattr(
        api,
        "ADMIN_TOKEN",
        "token-exclusivo-de-pruebas",
    )

    # 1. Intento con token incorrecto.
    respuesta_no_auth = client.post(
        "/admin/respaldo",
        json={
            "activar": True,
            "motivo": "Prueba sin token válido",
        },
        headers={
            "x-admin-token": "token-invalido",
        },
    )

    assert respuesta_no_auth.status_code == 401

    # 2. Activación con token válido.
    respuesta_ok = client.post(
        "/admin/respaldo",
        json={
            "activar": True,
            "motivo": "Degradación detectada en monitoreo",
        },
        headers={
            "x-admin-token": api.ADMIN_TOKEN,
        },
    )

    assert respuesta_ok.status_code == 200
    assert respuesta_ok.json()["modo_respaldo"] is True

    # 3. Comprobar que utiliza el respaldo.
    respuesta = client.post(
        "/predecir",
        json=PAYLOAD_VALIDO,
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["uso_respaldo"] is True


@pytest.mark.parametrize(
    "numero_sesion,sesiones_planificadas,decision_esperada,score_esperado",
    [
        (2, 3, "contactar", 0.99),
        (1, 3, "no_contactar", 0.13),
        (1, 4, "no_contactar", 0.13),
    ],
)
def test_respaldo_respeta_regla_negocio(
    client: TestClient,
    numero_sesion: int,
    sesiones_planificadas: int,
    decision_esperada: str,
    score_esperado: float,
) -> None:
    """El respaldo debe respetar la regla de una sesión restante."""

    api.estado.modo_respaldo = True

    payload = {
        **PAYLOAD_VALIDO,
        "numero_sesion": numero_sesion,
        "sesiones_planificadas": sesiones_planificadas,
    }

    respuesta = client.post(
        "/predecir",
        json=payload,
    )

    assert respuesta.status_code == 200

    body = respuesta.json()

    assert body["uso_respaldo"] is True
    assert body["version_modelo"] == "fallback-regla-1-sesion"

    assert body["probabilidad"] == pytest.approx(
        score_esperado
    )

    assert body["umbral_aplicado"] == pytest.approx(0.5)

    assert body["decision"] == decision_esperada


def test_interruptor_sin_token_configurado(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El endpoint administrativo debe estar protegido."""

    monkeypatch.setattr(api, "ADMIN_TOKEN", "")

    respuesta = client.post(
        "/admin/respaldo",
        json={
            "activar": True,
            "motivo": "Prueba sin configuración",
        },
    )

    assert respuesta.status_code == 503
    assert api.estado.modo_respaldo is False


def test_modelo_utiliza_umbral_version_activa(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La inferencia debe usar el umbral del modelo activo."""

    monkeypatch.setattr(
        api.estado,
        "umbral_decision",
        0.75,
    )

    respuesta = client.post(
        "/predecir",
        json=PAYLOAD_VALIDO,
    )

    assert respuesta.status_code == 200

    body = respuesta.json()

    assert body["uso_respaldo"] is False
    assert body["umbral_aplicado"] == pytest.approx(0.75)

    decision_esperada = (
        "contactar"
        if body["probabilidad"] >= 0.75
        else "no_contactar"
    )

    assert body["decision"] == decision_esperada