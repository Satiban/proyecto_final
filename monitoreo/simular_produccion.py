"""
Simulación de producción por lotes y monitoreo de deriva — OralSolutions (PSet 4).

1. Divide las 826 sesiones de prueba (>= 2025-08-01) ordenadas cronológicamente
   en 8 lotes sucesivos (L01 a L08), respetando el horizonte de 60 días para
   evaluar el target real.
2. Construye dos lotes con deriva inducida y documentada:
   - L09_deriva_datos: desplaza distribuciones de entrada (dias_desde_inicio_plan,
     edad_en_atencion y codigo_procedimiento) manteniendo la relación con el target.
   - L10_deriva_conceptual: mantiene entradas estables (PSI bajo), pero altera la
     relación entre progreso_plan y abandono_60d para inducir caída de desempeño.
3. Llama a la API en Docker (POST /predecir) por cada sesión, calcula métricas de
   sistema, datos, modelo y negocio, evalúa alertas y exporta evidencias a docs/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

BASE_MONITOREO = Path(__file__).resolve().parent
BASE_PROYECTO = BASE_MONITOREO.parent
if str(BASE_PROYECTO) not in sys.path:
    sys.path.insert(0, str(BASE_PROYECTO))

from src.pipeline import (
    FECHA_CORTE,
    FECHA_EVENTO,
    FEATURES_CAT,
    FEATURES_NUM,
    SEMILLA,
    TARGET,
    cargar_datos,
    construir_features,
)
from monitoreo.alertas import UMBRAL_PSI_CRITICO, evaluar_alertas_lote
from monitoreo.deriva import diagnosticar_deriva, evaluar_deriva_lote

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
N_LOTES_SANOS = 8

# Parámetros de negocio definidos en el PSet 3
COSTO_FN = 120.0
COSTO_FP = 3.0
TASA_RESCATE_TP = 0.35
INGRESO_RESCATE = COSTO_FN * TASA_RESCATE_TP  # $42 USD

COLS_API = [
    "duracion_estimada_min",
    "numero_sesion",
    "sesiones_planificadas",
    "cancelaciones_previas",
    "inasistencias_previas",
    "dias_desde_inicio_plan",
    "edad_en_atencion",
    "codigo_procedimiento",
    "odontologo_key",
]


def calcular_metricas_negocio(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    costo_errores = float(fn * COSTO_FN + fp * COSTO_FP)
    balance_neto = float(tp * (INGRESO_RESCATE - COSTO_FP) - fp * COSTO_FP - fn * COSTO_FN)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "costo_errores_usd": round(costo_errores, 2),
        "balance_neto_usd": round(balance_neto, 2),
    }


def construir_lotes_simulacion(df_test: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """
    Construye 8 lotes cronológicos reales + 1 lote de deriva de datos + 1 lote de deriva conceptual.
    Cada tupla devuelve: (lote_id, tipo_lote, dataframe_lote).
    """
    indices = np.array_split(np.arange(len(df_test)), N_LOTES_SANOS)
    lotes: list[tuple[str, str, pd.DataFrame]] = []

    for i, idx in enumerate(indices, start=1):
        sub = df_test.iloc[idx].copy().reset_index(drop=True)
        lotes.append((f"L{i:02d}", "produccion_real", sub))

    rng = np.random.default_rng(SEMILLA)

    # Lote 9: Deriva de datos inducida (cambian entradas, se mantiene relación X -> y)
    lote_datos = lotes[-1][2].copy()
    lote_datos["dias_desde_inicio_plan"] = (lote_datos["dias_desde_inicio_plan"] + 45).clip(upper=240)
    lote_datos["edad_en_atencion"] = (lote_datos["edad_en_atencion"] + 18).clip(upper=95)
    mask_proc = lote_datos["codigo_procedimiento"].isin(["CROWN", "ROOT"]) & (
        rng.random(len(lote_datos)) < 0.75
    )
    lote_datos.loc[mask_proc, "codigo_procedimiento"] = "WHITEN"
    lote_datos["fecha_realizada"] = lote_datos["fecha_realizada"] + pd.Timedelta(days=14)
    lote_datos = construir_features(lote_datos)
    lotes.append(("L09_deriva_datos", "deriva_datos_inducida", lote_datos))

    # Lote 10: Deriva conceptual inducida (entradas estables, cambia relación X -> y)
    lote_concepto = lotes[-2][2].copy()
    lote_concepto["fecha_realizada"] = lote_concepto["fecha_realizada"] + pd.Timedelta(days=28)
    mask_alto_prog = (lote_concepto["progreso_plan"] >= 0.50) & (rng.random(len(lote_concepto)) < 0.45)
    mask_bajo_prog = (lote_concepto["progreso_plan"] < 0.50) & (rng.random(len(lote_concepto)) < 0.40)
    lote_concepto.loc[mask_alto_prog, TARGET] = 1 - lote_concepto.loc[mask_alto_prog, TARGET]
    lote_concepto.loc[mask_bajo_prog, TARGET] = 1 - lote_concepto.loc[mask_bajo_prog, TARGET]
    lotes.append(("L10_deriva_conceptual", "deriva_conceptual_inducida", lote_concepto))

    return lotes


def ejecutar_simulacion() -> None:
    df_completo = construir_features(cargar_datos())
    corte_ts = pd.Timestamp(FECHA_CORTE)
    df_train = df_completo[df_completo[FECHA_EVENTO] < corte_ts].copy().reset_index(drop=True)
    df_test = df_completo[df_completo[FECHA_EVENTO] >= corte_ts].copy().reset_index(drop=True)

    # Verificar que la API esté arriba
    salud = requests.get(f"{API_URL}/salud", timeout=10).json()
    modelo_info = requests.get(f"{API_URL}/modelo", timeout=10).json()
    print(
        f"[API] Estado={salud['estado']} | Modelo={modelo_info['nombre_modelo']} "
        f"v{modelo_info['version']} (@{modelo_info['alias']}) | run_id={modelo_info['run_id']}"
    )

    lotes = construir_lotes_simulacion(df_test)
    sesion_http = requests.Session()

    todas_predicciones: list[dict[str, Any]] = []
    resumen_monitoreo: list[dict[str, Any]] = []
    todas_alertas: list[dict[str, Any]] = []
    lotes_guardados: list[pd.DataFrame] = []

    for lote_id, tipo_lote, df_lote in lotes:
        df_lote_tag = df_lote.copy()
        df_lote_tag["lote_id"] = lote_id
        df_lote_tag["tipo_lote"] = tipo_lote
        lotes_guardados.append(df_lote_tag)

        latencias: list[float] = []
        errores_4xx = 0
        errores_5xx = 0
        probs: list[float] = []
        preds_bin: list[int] = []
        version_lote = modelo_info["version"]

        fecha_min = pd.Timestamp(df_lote[FECHA_EVENTO].min())
        fecha_max = pd.Timestamp(df_lote[FECHA_EVENTO].max())
        # Respetar horizonte de 60 días: fecha en que el target real queda disponible
        fecha_target_disponible = (fecha_max + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

        for _, row in df_lote.iterrows():
            payload = {
                "duracion_estimada_min": int(row["duracion_estimada_min"]),
                "numero_sesion": int(row["numero_sesion"]),
                "sesiones_planificadas": int(row["sesiones_planificadas"]),
                "cancelaciones_previas": int(row["cancelaciones_previas"]),
                "inasistencias_previas": int(row["inasistencias_previas"]),
                "dias_desde_inicio_plan": int(row["dias_desde_inicio_plan"]),
                "edad_en_atencion": int(row["edad_en_atencion"]),
                "codigo_procedimiento": str(row["codigo_procedimiento"]),
                "odontologo_key": str(row["odontologo_key"]),
            }

            resp = sesion_http.post(f"{API_URL}/predecir", json=payload, timeout=15)
            if 400 <= resp.status_code < 500:
                errores_4xx += 1
                continue
            if resp.status_code >= 500:
                errores_5xx += 1
                continue

            body = resp.json()
            latencias.append(float(body["latencia_ms"]))
            probs.append(float(body["probabilidad"]))
            pred_b = 1 if body["decision"] == "contactar" else 0
            preds_bin.append(pred_b)
            version_lote = body["version_modelo"]

            todas_predicciones.append(
                {
                    "id_prediccion": body["id_prediccion"],
                    "lote_id": lote_id,
                    "tipo_lote": tipo_lote,
                    "fecha_realizada": pd.Timestamp(row[FECHA_EVENTO]).strftime("%Y-%m-%d"),
                    "fecha_target_disponible": fecha_target_disponible,
                    "timestamp_prediccion": body["timestamp"],
                    "entradas_json": json.dumps(payload, ensure_ascii=False),
                    "probabilidad": body["probabilidad"],
                    "decision": body["decision"],
                    "target_real": int(row[TARGET]),
                    "version_modelo": body["version_modelo"],
                    "run_id": body["run_id"],
                    "uso_respaldo": body["uso_respaldo"],
                    "latencia_ms": body["latencia_ms"],
                }
            )

        y_true = df_lote[TARGET].astype(int).to_numpy()
        y_prob = np.array(probs, dtype=float)
        y_pred = np.array(preds_bin, dtype=int)

        # Métricas técnicas y de negocio del modelo
        pr_auc = round(float(average_precision_score(y_true, y_prob)), 5)
        neg_modelo = calcular_metricas_negocio(y_true, y_pred)

        # Líneas base de negocio para comparación lote por lote
        pred_regla = ((df_lote["sesiones_planificadas"] - df_lote["numero_sesion"]) == 1).astype(int).to_numpy()
        neg_regla = calcular_metricas_negocio(y_true, pred_regla)
        neg_todos = calcular_metricas_negocio(y_true, np.ones(len(y_true), dtype=int))
        neg_nada = calcular_metricas_negocio(y_true, np.zeros(len(y_true), dtype=int))

        # Deriva de datos y calidad contra entrenamiento (df_train)
        df_deriva = evaluar_deriva_lote(df_train, df_lote, FEATURES_NUM, FEATURES_CAT)
        top_deriva = df_deriva.iloc[0]
        psi_max = float(top_deriva["psi"])
        feature_psi_max = str(top_deriva["feature"])
        pct_faltantes_max = float(df_deriva["pct_faltantes"].max())
        pct_fuera_rango_max = float(df_deriva["pct_fuera_rango"].max())
        cats_no_vistas_total = int(df_deriva["categorias_no_vistas"].sum())

        # Diagnóstico y alertas
        diag = diagnosticar_deriva(psi_max=psi_max, pr_auc_lote=pr_auc, umbral_psi=UMBRAL_PSI_CRITICO)
        tasa_err = (errores_4xx + errores_5xx) / max(len(df_lote), 1)
        p50_ms = round(float(np.percentile(latencias, 50)), 3) if latencias else 0.0
        p95_ms = round(float(np.percentile(latencias, 95)), 3) if latencias else 0.0

        alertas_lote = evaluar_alertas_lote(
            lote_id=lote_id,
            psi_max=psi_max,
            feature_psi_max=feature_psi_max,
            pr_auc_lote=pr_auc,
            tasa_error_http=tasa_err,
            latencia_p95_ms=p95_ms,
            pct_faltantes_max=pct_faltantes_max,
        )
        todas_alertas.extend(alertas_lote)

        resumen_monitoreo.append(
            {
                "lote_id": lote_id,
                "tipo_lote": tipo_lote,
                "fecha_inicio": fecha_min.strftime("%Y-%m-%d"),
                "fecha_fin": fecha_max.strftime("%Y-%m-%d"),
                "fecha_target_disponible": fecha_target_disponible,
                "n_sesiones": len(df_lote),
                "version_modelo": version_lote,
                "latencia_p50_ms": p50_ms,
                "latencia_p95_ms": p95_ms,
                "errores_4xx": errores_4xx,
                "errores_5xx": errores_5xx,
                "tasa_contactar": round(float(y_pred.mean()), 4),
                "prob_media": round(float(y_prob.mean()), 4),
                "tasa_abandono_real": round(float(y_true.mean()), 4),
                "pr_auc": pr_auc,
                "recall": neg_modelo["recall"],
                "precision": neg_modelo["precision"],
                "costo_modelo_usd": neg_modelo["costo_errores_usd"],
                "balance_modelo_usd": neg_modelo["balance_neto_usd"],
                "costo_regla_1_sesion_usd": neg_regla["costo_errores_usd"],
                "balance_regla_1_sesion_usd": neg_regla["balance_neto_usd"],
                "costo_llamar_todos_usd": neg_todos["costo_errores_usd"],
                "balance_llamar_todos_usd": neg_todos["balance_neto_usd"],
                "costo_no_hacer_nada_usd": neg_nada["costo_errores_usd"],
                "psi_max": round(psi_max, 5),
                "feature_psi_max": feature_psi_max,
                "p_valor_top_deriva": float(top_deriva["p_valor"]),
                "pct_faltantes_max": pct_faltantes_max,
                "pct_fuera_rango_max": pct_fuera_rango_max,
                "categorias_no_vistas": cats_no_vistas_total,
                "diagnostico": diag["diagnostico"],
                "n_alertas": len(alertas_lote),
            }
        )

    # Guardar evidencias CSV en docs/ y dataset de lotes para reentrenamiento
    docs_dir = BASE_PROYECTO / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    df_preds = pd.DataFrame(todas_predicciones)
    df_mon = pd.DataFrame(resumen_monitoreo)
    df_alrt = pd.DataFrame(todas_alertas)
    df_lotes_all = pd.concat(lotes_guardados, ignore_index=True)

    df_preds.to_csv(docs_dir / "predicciones.csv", index=False)
    df_mon.to_csv(docs_dir / "monitoreo.csv", index=False)
    df_alrt.to_csv(docs_dir / "alertas.csv", index=False)
    df_lotes_all.to_csv(BASE_MONITOREO / "lotes_simulados.csv", index=False)

    print("\n=== RESUMEN DE SIMULACIÓN POR LOTE ===")
    cols_vista = [
        "lote_id",
        "n_sesiones",
        "pr_auc",
        "psi_max",
        "feature_psi_max",
        "costo_modelo_usd",
        "diagnostico",
        "n_alertas",
    ]
    print(df_mon[cols_vista].to_string(index=False))
    print(f"\n[OK] Predicciones guardadas : {len(df_preds)} filas -> docs/predicciones.csv")
    print(f"[OK] Monitoreo guardado     : {len(df_mon)} lotes -> docs/monitoreo.csv")
    print(f"[OK] Alertas disparadas     : {len(df_alrt)} alertas -> docs/alertas.csv")


if __name__ == "__main__":
    ejecutar_simulacion()