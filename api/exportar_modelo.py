"""
Exporta el modelo registrado en MLflow (`oralsolutions-modelo@champion`)
a disco para uso en pruebas automáticas (CI) y respaldo de verificación.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "oralsolutions-modelo")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

BASE_DIR = Path(__file__).resolve().parent
DESTINO_MODELO = BASE_DIR / "fixtures" / "modelo_champion"
DESTINO_META = BASE_DIR / "fixtures" / "modelo_meta.json"


def exportar_modelo() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    modelo = mlflow.sklearn.load_model(uri)

    if DESTINO_MODELO.exists():
        shutil.rmtree(DESTINO_MODELO)
    DESTINO_MODELO.parent.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(sk_model=modelo, path=str(DESTINO_MODELO))

    meta = {
        "nombre_modelo": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
    }
    DESTINO_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Modelo exportado desde {uri}")
    print(f"     Versión : {mv.version}")
    print(f"     run_id  : {mv.run_id}")
    print(f"     Destino : {DESTINO_MODELO}")


if __name__ == "__main__":
    exportar_modelo()