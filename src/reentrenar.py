"""
Tubería de reentrenamiento controlado — OralSolutions (PSet 4, Sección 7).

Flujo:
1. Arma la ventana de reentrenamiento (histórico + lotes etiquetados sanos L01-L08).
2. Reentrena usando pipeline_proyecto.py con los hiperparámetros exactos de @champion
   (sin repetir búsqueda de hiperparámetros).
3. Registra la corrida en MLflow con tag `fase=reentrenamiento` y el motivo del disparo.
4. Registra la nueva versión en Model Registry con el alias `@challenger`.
5. Evalúa @champion y @challenger sobre el lote etiquetado más reciente no visto (L10).
6. Aplica la regla de promoción y ejecuta la decisión humana (--decision pendiente|aprobar|rechazar).
7. Exporta docs/corridas.csv y docs/reentrenamientos.csv desde MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

BASE_MODELADO = Path(__file__).resolve().parent
BASE_PROYECTO = BASE_MODELADO.parent
if str(BASE_PROYECTO) not in sys.path:
    sys.path.insert(0, str(BASE_PROYECTO))

from src.pipeline import (
    FECHA_CORTE,
    FECHA_EVENTO,
    FEATURES_FINALES,
    SEMILLA,
    TARGET,
    cargar_datos,
    construir_features,
    construir_pipeline,
)
from monitoreo.deriva import PR_AUC_CV_STD

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "OralSolutions_Reentrenamiento")
MODEL_NAME = os.getenv("MODEL_NAME", "oralsolutions-modelo")
UMBRAL_DECISION = float(os.getenv("UMBRAL_DECISION", "0.01"))

COSTO_FN = 120.0
COSTO_FP = 3.0
INGRESO_RESCATE = COSTO_FN * 0.35  # $42 USD


def evaluar_modelo_lote(modelo, X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    probs = modelo.predict_proba(X_eval)[:, 1]
    preds = (probs >= UMBRAL_DECISION).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()
    costo = float(fn * COSTO_FN + fp * COSTO_FP)
    balance = float(tp * (INGRESO_RESCATE - COSTO_FP) - fp * COSTO_FP - fn * COSTO_FN)
    return {
        "pr_auc": round(float(average_precision_score(y_eval, probs)), 5),
        "recall": round(float(recall_score(y_eval, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_eval, preds, zero_division=0)), 4),
        "costo_errores_usd": round(costo, 2),
        "balance_neto_usd": round(balance, 2),
        "FP": int(fp),
        "FN": int(fn),
    }


def exportar_evidencias_mlflow() -> None:
    docs_dir = BASE_PROYECTO / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    df_runs = mlflow.search_runs(search_all_experiments=True)
    df_runs.to_csv(docs_dir / "corridas.csv", index=False)

    col_fase = "tags.fase"
    if col_fase in df_runs.columns:
        df_reent = df_runs[df_runs[col_fase] == "reentrenamiento"].copy()
    else:
        df_reent = df_runs.iloc[0:0].copy()
    df_reent.to_csv(docs_dir / "reentrenamientos.csv", index=False)

    print(f"[OK] Exportadas {len(df_runs)} corridas totales -> docs/corridas.csv")
    print(f"[OK] Exportadas {len(df_reent)} corridas de reentrenamiento -> docs/reentrenamientos.csv")


def ejecutar_reentrenamiento(
    motivo: str,
    lote_eval: str,
    decision: str,
    motivo_decision: str,
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Crear o usar experimento nuevo para que MLflow sirva artefactos 100% por HTTP
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Cargar @champion actual y extraer su estimador (hiperparámetros sin repetir tuning)
    mv_champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    try:
        champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    except OSError:
        ruta_fixture = BASE_PROYECTO / "tests" / "fixtures" / "modelo_champion"
        champion = mlflow.sklearn.load_model(str(ruta_fixture))
    estimador_base = clone(champion.named_steps["modelo"])

    # 2. Armar ventana de datos: histórico (< 2025-08-01) + lotes etiquetados reales L01..L08
    df_hist = construir_features(cargar_datos())
    df_train_hist = df_hist[df_hist[FECHA_EVENTO] < pd.Timestamp(FECHA_CORTE)].copy()

    ruta_lotes = BASE_PROYECTO / "monitoreo" / "lotes_simulados.csv"
    if not ruta_lotes.exists():
        raise FileNotFoundError(
            "No existe monitoreo/lotes_simulados.csv. Ejecuta primero monitoreo/simular_produccion.py"
        )
    df_lotes = pd.read_csv(ruta_lotes)
    df_lotes = construir_features(df_lotes)

    # Solo lotes reales con target ya disponible (L01-L08); se excluye L09 (deriva de datos) y L10 (evaluación)
    lotes_train_ids = [f"L{i:02d}" for i in range(1, 9)]
    df_lotes_train = df_lotes[df_lotes["lote_id"].isin(lotes_train_ids)].copy()
    df_eval = df_lotes[df_lotes["lote_id"] == lote_eval].copy()
    if df_eval.empty:
        raise ValueError(f"El lote de evaluación '{lote_eval}' no existe en lotes_simulados.csv")

    df_ventana = pd.concat(
        [df_train_hist[FEATURES_FINALES + [TARGET]], df_lotes_train[FEATURES_FINALES + [TARGET]]],
        ignore_index=True,
    )
    X_ventana = df_ventana[FEATURES_FINALES].copy()
    y_ventana = df_ventana[TARGET].astype(int).to_numpy()

    X_eval = df_eval[FEATURES_FINALES].copy()
    y_eval = df_eval[TARGET].astype(int).to_numpy()

    # 3. Reentrenar usando exactamente pipeline_proyecto.py
    nuevo_pipe = construir_pipeline(estimador_base, escalar_numericas=True)

    with mlflow.start_run(run_name=f"REENTRENAMIENTO_{lote_eval}") as run:
        t0 = time.perf_counter()
        nuevo_pipe.fit(X_ventana, y_ventana)
        segundos = round(time.perf_counter() - t0, 4)

        # 4. Evaluar @champion y @challenger sobre el mismo lote no visto (L10)
        res_champ = evaluar_modelo_lote(champion, X_eval, y_eval)
        res_chall = evaluar_modelo_lote(nuevo_pipe, X_eval, y_eval)

        # Regla de promoción de docs/despliegue.md
        mejora_negocio = res_chall["costo_errores_usd"] < res_champ["costo_errores_usd"]
        estable_tecnica = res_chall["pr_auc"] >= (res_champ["pr_auc"] - PR_AUC_CV_STD)
        cumple_regla = bool(mejora_negocio and estable_tecnica)

        mlflow.log_params(
            {
                "ventana_datos": "historico_mas_L01_a_L08",
                "n_filas_entrenamiento": len(X_ventana),
                "lote_evaluacion": lote_eval,
                "n_filas_evaluacion": len(X_eval),
                "champion_previo_version": str(mv_champ.version),
                "semilla": SEMILLA,
                "umbral_decision": UMBRAL_DECISION,
            }
        )
        mlflow.log_metrics(
            {
                "pr_auc_champion_eval": res_champ["pr_auc"],
                "pr_auc_challenger_eval": res_chall["pr_auc"],
                "costo_champion_usd": res_champ["costo_errores_usd"],
                "costo_challenger_usd": res_chall["costo_errores_usd"],
                "balance_champion_usd": res_champ["balance_neto_usd"],
                "balance_challenger_usd": res_chall["balance_neto_usd"],
                "cumple_regla_promocion": int(cumple_regla),
                "segundos_entrenamiento": segundos,
            }
        )
        mlflow.set_tags(
            {
                "fase": "reentrenamiento",
                "motivo_disparo": motivo,
                "lote_evaluacion": lote_eval,
                "cumple_regla_promocion": str(cumple_regla),
                "decision_humana": decision,
                "motivo_decision": motivo_decision,
            }
        )

        input_example = X_ventana.iloc[:5].copy()
        signature = infer_signature(input_example, nuevo_pipe.predict_proba(input_example))
        mlflow.sklearn.log_model(
            sk_model=nuevo_pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )
        new_run_id = run.info.run_id

    # 5. Asignar alias @challenger a la nueva versión creada
    versiones = client.search_model_versions(f"name='{MODEL_NAME}'")
    mv_nuevo = max(versiones, key=lambda v: int(v.version))
    nueva_version = str(mv_nuevo.version)
    client.set_registered_model_alias(MODEL_NAME, "challenger", nueva_version)

    # 6. Aplicar decisión humana (promover o mantener champion intacto)
    promovido = False
    if decision == "aprobar":
        client.set_registered_model_alias(MODEL_NAME, "champion", nueva_version)
        promovido = True

    resumen = {
        "motivo_disparo": motivo,
        "ventana_entrenamiento": f"historico ({len(df_train_hist)}) + L01-L08 ({len(df_lotes_train)}) = {len(X_ventana)} filas",
        "lote_evaluacion": lote_eval,
        "champion_actual": {"version": str(mv_champ.version), "run_id": mv_champ.run_id, **res_champ},
        "challenger_nuevo": {"version": nueva_version, "run_id": new_run_id, **res_chall},
        "cumple_regla_promocion": cumple_regla,
        "decision_humana": decision,
        "motivo_decision": motivo_decision,
        "promovido_a_champion": promovido,
    }

    ruta_resumen = BASE_PROYECTO / "monitoreo" / "ultimo_reentrenamiento.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULTADO DEL REENTRENAMIENTO ===")
    print(f"Ventana datos       : {resumen['ventana_entrenamiento']}")
    print(f"Lote evaluación     : {lote_eval} ({len(X_eval)} sesiones, no visto en entrenamiento)")
    print(
        f"@champion  (v{mv_champ.version})   : PR-AUC={res_champ['pr_auc']} | "
        f"Costo=${res_champ['costo_errores_usd']} | Balance=${res_champ['balance_neto_usd']}"
    )
    print(
        f"@challenger (v{nueva_version})  : PR-AUC={res_chall['pr_auc']} | "
        f"Costo=${res_chall['costo_errores_usd']} | Balance=${res_chall['balance_neto_usd']}"
    )
    print(f"¿Cumple regla?      : {cumple_regla}")
    print(f"Decisión humana     : {decision} ({motivo_decision})")
    if promovido:
        print(f"[PROMOVIDO] Alias @champion movido a v{nueva_version} (API toma el cambio sin reconstruir).")
    else:
        print(f"[INTACTO] Alias @champion permanece en v{mv_champ.version}; v{nueva_version} queda como @challenger.")

    exportar_evidencias_mlflow()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reentrenamiento controlado OralSolutions")
    parser.add_argument(
        "--motivo",
        default="Alerta critica de deriva conceptual en L10_deriva_conceptual (PR-AUC < 0.960)",
    )
    parser.add_argument("--lote-eval", default="L10_deriva_conceptual")
    parser.add_argument(
        "--decision",
        choices=["pendiente", "aprobar", "rechazar"],
        default="pendiente",
    )
    parser.add_argument(
        "--motivo-decision",
        default="Esperando aprobacion manual en flujo de Kestra",
    )
    args = parser.parse_args()
    ejecutar_reentrenamiento(
        motivo=args.motivo,
        lote_eval=args.lote_eval,
        decision=args.decision,
        motivo_decision=args.motivo_decision,
    )