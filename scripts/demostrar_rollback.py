"""
Demostración de versionado y rollback de OralSolutions.

Verifica:
1. Estado inicial del champion.
2. Cambio temporal del alias champion.
3. Inferencia con el nuevo modelo sin reiniciar FastAPI.
4. Rollback a la versión original.
5. Recuperación de la predicción inicial.

No entrena ni registra nuevos modelos.
"""

from __future__ import annotations

import os
import math

import mlflow
import requests
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

MODEL_NAME = "oralsolutions-modelo"

PAYLOAD = {
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


def consultar_modelo() -> dict:

    respuesta = requests.get(
        f"{API_URL}/modelo",
        timeout=120
    )

    respuesta.raise_for_status()

    return respuesta.json()


def generar_prediccion() -> dict:

    respuesta = requests.post(
        f"{API_URL}/predecir",
        json=PAYLOAD,
        timeout=120
    )

    respuesta.raise_for_status()

    return respuesta.json()


def mostrar_prediccion(resultado: dict) -> None:

    print(f"   ID predicción : {resultado['id_prediccion']}")
    print(f"   Versión       : {resultado['version_modelo']}")
    print(f"   Run ID        : {resultado['run_id']}")
    print(f"   Probabilidad  : {resultado['probabilidad']:.10f}")
    print(f"   Umbral        : {resultado['umbral_aplicado']}")
    print(f"   Decisión      : {resultado['decision']}")
    print(f"   Respaldo      : {resultado['uso_respaldo']}")
    print(f"   Latencia      : {resultado['latencia_ms']} ms")


def validar_prediccion(
    resultado: dict,
    version: str,
    run_id: str,
) -> None:

    assert resultado["version_modelo"] == version, (
        "La API está utilizando una versión incorrecta."
    )

    assert resultado["run_id"] == run_id, (
        "El run_id no corresponde a la versión esperada."
    )

    assert resultado["uso_respaldo"] is False, (
        "La API está utilizando la regla de respaldo."
    )


def demostrar_rollback() -> None:

    print("=" * 65)
    print("ORALSOLUTIONS - DEMOSTRACIÓN DE ROLLBACK")
    print("=" * 65)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger"
    )

    version_original = str(champion.version)
    version_alternativa = str(challenger.version)

    if version_original == version_alternativa:
        raise RuntimeError(
            "Champion y challenger deben ser versiones diferentes."
        )

    print("\n1. ESTADO INICIAL")
    print(f"   Champion   : v{version_original}")
    print(f"   Challenger : v{version_alternativa}")

    modelo_inicial = consultar_modelo()

    assert modelo_inicial["version"] == version_original
    assert modelo_inicial["modo_respaldo"] is False

    prediccion_inicial = generar_prediccion()

    validar_prediccion(
        prediccion_inicial,
        version_original,
        champion.run_id,
    )

    mostrar_prediccion(prediccion_inicial)

    try:

        print("\n2. CAMBIO TEMPORAL DE CHAMPION")

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            version_alternativa
        )

        client.set_registered_model_alias(
            MODEL_NAME,
            "challenger",
            version_original
        )

        print(
            f"   Alias champion actualizado a v{version_alternativa}"
        )

        print("\n3. PREDICCIÓN CON EL NUEVO CHAMPION")

        # Se predice antes de consultar /modelo para comprobar
        # que POST /predecir detecta el cambio por sí mismo.
        prediccion_alternativa = generar_prediccion()

        validar_prediccion(
            prediccion_alternativa,
            version_alternativa,
            challenger.run_id,
        )

        mostrar_prediccion(prediccion_alternativa)

        modelo_alternativo = consultar_modelo()

        assert (
            modelo_alternativo["version"]
            == version_alternativa
        )

        print("   [OK] Cambio de versión detectado por FastAPI.")

    finally:

        print("\n4. ROLLBACK A LA VERSIÓN ORIGINAL")

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            version_original
        )

        client.set_registered_model_alias(
            MODEL_NAME,
            "challenger",
            version_alternativa
        )

        print(
            f"   Champion restaurado a v{version_original}"
        )

        print("\n5. PREDICCIÓN DESPUÉS DEL ROLLBACK")

        prediccion_restaurada = generar_prediccion()

        validar_prediccion(
            prediccion_restaurada,
            version_original,
            champion.run_id,
        )

        mostrar_prediccion(prediccion_restaurada)

        modelo_restaurado = consultar_modelo()

        assert modelo_restaurado["version"] == version_original

        assert math.isclose(
            prediccion_inicial["probabilidad"],
            prediccion_restaurada["probabilidad"],
            rel_tol=0,
            abs_tol=1e-10
        ), "La predicción restaurada no coincide con la original."

        assert (
            prediccion_inicial["decision"]
            == prediccion_restaurada["decision"]
        )

        assert (
            prediccion_inicial["umbral_aplicado"]
            == prediccion_restaurada["umbral_aplicado"]
        )

    print("\n" + "=" * 65)
    print("[OK] DEMOSTRACIÓN COMPLETADA")
    print("=" * 65)

    print("Cambio de champion: verificado")
    print("Rollback: verificado")
    print("Predicción original recuperada: verificado")
    print(f"Champion definitivo: v{version_original}")


if __name__ == "__main__":
    demostrar_rollback()
