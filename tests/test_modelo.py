"""
Prueba de reproducibilidad del champion de OralSolutions.

Comprueba que el HistGradientBoosting exportado desde MLflow
reproduce la PR-AUC del hold-out del PSet 3.

Se utiliza un fixture local para permitir la ejecución
en GitHub Actions sin conexión al servidor MLflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

# Utilizar el modelo local durante las pruebas CI
os.environ["USAR_MODELO_LOCAL"] = "true"
os.environ["REGISTRAR_SNOWFLAKE"] = "false"

from src.pipeline import FEATURES_FINALES, TARGET, construir_features
from api import main as api


BASE_FIXTURES = Path(__file__).resolve().parent / "fixtures"

MUESTRA_TEST = BASE_FIXTURES / "muestra_test.csv"
META_MODELO = BASE_FIXTURES / "modelo_meta.json"

PR_AUC_ESPERADO_MLFLOW = 0.159858
TOLERANCIA_PR_AUC = 0.0005

RUN_ID_ESPERADO = "c90a1c37aef940278afc025b1d22dc6b"


@pytest.fixture()
def client(tmp_path: Path):
    """Carga el champion local y aísla los registros del test."""

    api.RUTA_LOG_PREDICCIONES = tmp_path / "predicciones_test.csv"
    api.estado.modo_respaldo = False
    api.cargar_modelo()

    with TestClient(api.app) as c:
        yield c

    api.estado.modo_respaldo = False


def test_modelo_reproduce_metrica_mlflow(client: TestClient) -> None:
    """Comprueba identidad del champion y PR-AUC del hold-out."""

    assert MUESTRA_TEST.exists(), (
        f"No existe la muestra de prueba: {MUESTRA_TEST}"
    )
    assert META_MODELO.exists(), (
        f"No existe la metadata del modelo: {META_MODELO}"
    )

    assert api.estado.modelo_cargado
    assert api.estado.modelo is not None

    metadata = json.loads(
        META_MODELO.read_text(encoding="utf-8")
    )

    assert metadata["nombre_modelo"] == "oralsolutions-modelo"
    assert metadata["alias"] == "champion"
    assert str(metadata["version"]) == "1"
    assert metadata["run_id"] == RUN_ID_ESPERADO
    assert float(metadata["umbral_decision"]) == 0.01

    assert api.estado.run_id == RUN_ID_ESPERADO
    assert api.estado.version == "1"

    df = pd.read_csv(MUESTRA_TEST)

    assert len(df) == 826
    assert df[TARGET].value_counts().to_dict() == {
        0: 708,
        1: 118,
    }

    df_features = construir_features(df)

    X_eval = df_features[FEATURES_FINALES]
    y_eval = df[TARGET].astype(int)

    probabilidades = api.estado.modelo.predict_proba(X_eval)[:, 1]

    pr_auc = float(
        average_precision_score(y_eval, probabilidades)
    )

    print(f"Champion: HistGradientBoosting")
    print(f"Versión: {api.estado.version}")
    print(f"Run ID: {api.estado.run_id}")
    print(f"PR-AUC obtenida: {pr_auc:.6f}")

    assert abs(pr_auc - PR_AUC_ESPERADO_MLFLOW) <= TOLERANCIA_PR_AUC, (
        f"PR-AUC obtenida ({pr_auc:.6f}) difiere de MLflow "
        f"({PR_AUC_ESPERADO_MLFLOW:.6f}) "
        f"más allá de ±{TOLERANCIA_PR_AUC}"
    )