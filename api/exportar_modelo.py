
"""
Exportación del champion de OralSolutions para pruebas automáticas.

Genera:
- tests/fixtures/modelo_champion/
- tests/fixtures/modelo_meta.json
- tests/fixtures/muestra_test.csv

No modifica el Model Registry ni los aliases de MLflow.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score

from src.pipeline import (
    FECHA_CORTE,
    FECHA_EVENTO,
    FEATURES_FINALES,
    TARGET,
    cargar_datos,
    construir_features,
    obtener_hash_datos,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "oralsolutions-modelo"
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion"
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"

DESTINO_MODELO = FIXTURES_DIR / "modelo_champion"
DESTINO_META = FIXTURES_DIR / "modelo_meta.json"
DESTINO_MUESTRA = FIXTURES_DIR / "muestra_test.csv"

TOLERANCIA_PR_AUC = 0.0005


# ============================================================
# EXPORTACIÓN Y VALIDACIÓN
# ============================================================

def exportar_modelo() -> None:

    print("=" * 65)
    print("EXPORTACIÓN DEL CHAMPION — ORALSOLUTIONS")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Conectar a MLflow
    # --------------------------------------------------------

    print("\n1. Conectando a MLflow...")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    mv = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    run = client.get_run(mv.run_id)

    uri_alias = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    # Fijar la versión para garantizar consistencia
    # entre el artefacto y sus metadatos.
    uri_version = f"models:/{MODEL_NAME}/{mv.version}"

    print(f"   Modelo  : {MODEL_NAME}")
    print(f"   Alias   : {MODEL_ALIAS}")
    print(f"   Versión : {mv.version}")
    print(f"   Run ID  : {mv.run_id}")

    # --------------------------------------------------------
    # 2. Recuperar metadatos
    # --------------------------------------------------------

    print("\n2. Recuperando metadatos del entrenamiento...")

    umbral = float(
        mv.tags.get(
            "umbral_decision",
            run.data.params.get("umbral_negocio")
        )
    )

    fecha_corte = run.data.params["fecha_corte"]
    horizonte = int(run.data.params["horizonte_dias"])

    pr_auc_mlflow = float(
        run.data.metrics["pr_auc_test"]
    )

    hash_registrado = run.data.params["hash_dataset"]

    costo_fn = float(run.data.params["costo_fn"])
    costo_fp = float(run.data.params["costo_fp"])

    if not 0 <= umbral <= 1:
        raise ValueError("Umbral de decisión inválido.")

    if fecha_corte != FECHA_CORTE or horizonte != 60:
        raise ValueError(
            "El corte o horizonte registrado no coincide "
            "con el pipeline actual."
        )

    print(f"   Umbral        : {umbral}")
    print(f"   Fecha corte   : {fecha_corte}")
    print(f"   Horizonte     : {horizonte} días")
    print(f"   PR-AUC MLflow : {pr_auc_mlflow:.6f}")

    # --------------------------------------------------------
    # 3. Descargar el modelo y su firma
    # --------------------------------------------------------

    print("\n3. Descargando el modelo registrado...")

    info = mlflow.models.get_model_info(uri_version)
    modelo = mlflow.sklearn.load_model(uri_version)

    if list(modelo.feature_names_in_) != FEATURES_FINALES:
        raise ValueError(
            "Las features del modelo no coinciden "
            "con src/pipeline.py."
        )

    if list(modelo.classes_) != [0, 1]:
        raise ValueError(
            "El orden de las clases del modelo no es [0, 1]."
        )

    print(f"   Estimador: {type(modelo.named_steps['modelo']).__name__}")
    print(f"   Features : {len(FEATURES_FINALES)}")

    # --------------------------------------------------------
    # 4. Verificar la identidad del dataset
    # --------------------------------------------------------

    print("\n4. Verificando dataset del PSet03...")

    hash_actual = obtener_hash_datos()

    if hash_actual != hash_registrado:
        raise ValueError(
            "El hash del dataset actual no coincide "
            "con el registrado en MLflow."
        )

    print(f"   SHA-256: {hash_actual}")
    print("   [OK] Dataset original verificado.")

    # --------------------------------------------------------
    # 5. Construir muestra fija de evaluación
    # --------------------------------------------------------

    print("\n5. Generando muestra de prueba...")

    df = cargar_datos()

    muestra_test = df.loc[
        df[FECHA_EVENTO] >= pd.Timestamp(fecha_corte)
    ].copy().reset_index(drop=True)

    if muestra_test.empty:
        raise ValueError("El hold-out está vacío.")

    df_features = construir_features(muestra_test)

    X_test = df_features[FEATURES_FINALES]
    y_test = df_features[TARGET].astype(int)

    print(f"   Observaciones : {len(muestra_test):,}")
    print(f"   Continuidad   : {(y_test == 0).sum():,}")
    print(f"   Abandono      : {(y_test == 1).sum():,}")

    # --------------------------------------------------------
    # 6. Verificar métrica del modelo registrado
    # --------------------------------------------------------

    print("\n6. Verificando PR-AUC del champion...")

    probabilidades = modelo.predict_proba(X_test)[:, 1]

    pr_auc_obtenida = float(
        average_precision_score(y_test, probabilidades)
    )

    diferencia = abs(pr_auc_obtenida - pr_auc_mlflow)

    print(f"   PR-AUC registrada : {pr_auc_mlflow:.6f}")
    print(f"   PR-AUC obtenida   : {pr_auc_obtenida:.6f}")
    print(f"   Diferencia        : {diferencia:.8f}")
    print(f"   Tolerancia        : {TOLERANCIA_PR_AUC}")

    if diferencia > TOLERANCIA_PR_AUC:
        raise ValueError(
            "El modelo no reproduce la métrica registrada."
        )

    # --------------------------------------------------------
    # 7. Exportar y verificar antes de reemplazar fixtures
    # --------------------------------------------------------

    print("\n7. Exportando modelo para CI...")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=FIXTURES_DIR) as temporal:

        ruta_temporal = Path(temporal) / "modelo"

        mlflow.sklearn.save_model(
            sk_model=modelo,
            path=str(ruta_temporal),
            signature=info.signature,
            input_example=X_test.head(3)
        )

        # Comprobar que el artefacto exportado funciona.
        modelo_local = mlflow.sklearn.load_model(
            str(ruta_temporal)
        )

        probabilidades_locales = (
            modelo_local.predict_proba(X_test)[:, 1]
        )

        np.testing.assert_allclose(
            probabilidades_locales,
            probabilidades,
            rtol=0,
            atol=1e-10
        )

        pr_auc_local = float(
            average_precision_score(
                y_test,
                probabilidades_locales
            )
        )

        # Reemplazar únicamente después de validar.
        if DESTINO_MODELO.exists():
            shutil.rmtree(DESTINO_MODELO)

        shutil.move(
            str(ruta_temporal),
            str(DESTINO_MODELO)
        )

    print(f"   PR-AUC modelo local: {pr_auc_local:.6f}")
    print("   [OK] Artefacto exportado y validado.")

    # --------------------------------------------------------
    # 8. Guardar muestra y metadatos
    # --------------------------------------------------------

    print("\n8. Guardando muestra y metadatos...")

    muestra_test.to_csv(
        DESTINO_MUESTRA,
        index=False,
        encoding="utf-8"
    )

    meta = {
        "nombre_modelo": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "umbral_decision": umbral,
        "fecha_corte": fecha_corte,
        "horizonte_dias": horizonte,
        "hash_dataset": hash_actual,
        "pr_auc_test": pr_auc_mlflow,
        "pr_auc_cv_mean": run.data.metrics.get(
            "pr_auc_cv_mean"
        ),
        "pr_auc_cv_std": run.data.metrics.get(
            "pr_auc_cv_std"
        ),
        "tolerancia_pr_auc": TOLERANCIA_PR_AUC,
        "n_observaciones_test": len(muestra_test),
        "costo_fn": costo_fn,
        "costo_fp": costo_fp,
        "features": FEATURES_FINALES
    }

    DESTINO_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 9. Resumen final
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("[OK] EXPORTACIÓN COMPLETADA CORRECTAMENTE")
    print("=" * 65)

    print(f"Origen     : {uri_alias}")
    print(f"Versión    : {mv.version}")
    print(f"Run ID     : {mv.run_id}")
    print(f"Umbral     : {umbral}")
    print(f"PR-AUC     : {pr_auc_local:.6f}")
    print(f"Modelo     : {DESTINO_MODELO}")
    print(f"Metadata   : {DESTINO_META}")
    print(f"Muestra    : {DESTINO_MUESTRA}")


if __name__ == "__main__":
    exportar_modelo()