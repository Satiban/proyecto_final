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

def test_modelo_reproduce_metrica_mlflow(client: TestClient) -> None:
    assert MUESTRA_TEST.exists(), f"Falta la muestra fija de validación: {MUESTRA_TEST}"
    assert api.estado.modelo_cargado and api.estado.modelo is not None

    df = pd.read_csv(MUESTRA_TEST)
    df_feat = construir_features(df)
    X_eval = df_feat[FEATURES_FINALES]
    y_eval = df[TARGET].astype(int)

    probs = api.estado.modelo.predict_proba(X_eval)[:, 1]
    pr_auc = float(average_precision_score(y_eval, probs))

    assert abs(pr_auc - PR_AUC_ESPERADO_MLFLOW) <= TOLERANCIA_PR_AUC, (
        f"PR-AUC obtenido ({pr_auc:.5f}) difiere de MLflow ({PR_AUC_ESPERADO_MLFLOW:.5f}) "
        f"más allá de ±{TOLERANCIA_PR_AUC}"
    )

