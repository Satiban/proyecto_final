"""
Motor de reglas de alerta para monitoreo de OralSolutions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from monitoreo.deriva import PR_AUC_CV_MEAN, PR_AUC_CV_STD

# Umbrales calibrados (justificados por la simulación en monitoreo/)
UMBRAL_PSI_ADVERTENCIA = 0.20
UMBRAL_PSI_CRITICO = 0.30
UMBRAL_PR_AUC_MIN = 0.96000
UMBRAL_TASA_ERROR_HTTP = 0.05
UMBRAL_LATENCIA_P95_MS = 250.0


def evaluar_alertas_lote(
    lote_id: str,
    psi_max: float,
    feature_psi_max: str,
    pr_auc_lote: float | None,
    tasa_error_http: float = 0.0,
    latencia_p95_ms: float = 0.0,
    pct_faltantes_max: float = 0.0,
) -> list[dict[str, Any]]:
    """Evalúa todas las reglas de alerta sobre un lote y devuelve las alertas disparadas."""
    ts = datetime.now(timezone.utc).isoformat()
    alertas: list[dict[str, Any]] = []

    if psi_max > UMBRAL_PSI_CRITICO:
        alertas.append(
            {
                "timestamp": ts,
                "lote_id": lote_id,
                "metrica": f"PSI de {feature_psi_max}",
                "valor_observado": round(psi_max, 5),
                "umbral": f"> {UMBRAL_PSI_CRITICO}",
                "ventana": "1 lote",
                "severidad": "Alta",
                "destinatario": "Equipo de Datos / MLOps",
                "accion": "Revisar fuente de datos, confirmar impacto en desempeño y evaluar reentrenamiento.",
            }
        )

    if pr_auc_lote is not None and pr_auc_lote < UMBRAL_PR_AUC_MIN:
        alertas.append(
            {
                "timestamp": ts,
                "lote_id": lote_id,
                "metrica": "PR-AUC del lote (deriva conceptual/desempeño)",
                "valor_observado": round(pr_auc_lote, 5),
                "umbral": f"< {UMBRAL_PR_AUC_MIN} (calibrado por simulación)",
                "ventana": "1 lote etiquetado",
                "severidad": "Crítica",
                "destinatario": "Equipo de Datos + Gerencia Clínica",
                "accion": "Disparar flujo de reentrenamiento en Kestra y evaluar modo respaldo.",
            }
        )

    if pct_faltantes_max > 5.0:
        alertas.append(
            {
                "timestamp": ts,
                "lote_id": lote_id,
                "metrica": "Porcentaje máximo de faltantes por feature",
                "valor_observado": round(pct_faltantes_max, 3),
                "umbral": "> 5.0%",
                "ventana": "1 lote",
                "severidad": "Media",
                "destinatario": "Ingeniería de Datos",
                "accion": "Auditar pipeline ELT (Kestra/dbt) y esquema de origen.",
            }
        )

    if tasa_error_http > UMBRAL_TASA_ERROR_HTTP:
        alertas.append(
            {
                "timestamp": ts,
                "lote_id": lote_id,
                "metrica": "Tasa de errores HTTP (4xx/5xx)",
                "valor_observado": round(tasa_error_http, 4),
                "umbral": f"> {UMBRAL_TASA_ERROR_HTTP:.0%}",
                "ventana": "1 lote",
                "severidad": "Alta",
                "destinatario": "Equipo de Plataforma / API",
                "accion": "Revisar contrato Pydantic y salud del contenedor api.",
            }
        )

    if latencia_p95_ms > UMBRAL_LATENCIA_P95_MS:
        alertas.append(
            {
                "timestamp": ts,
                "lote_id": lote_id,
                "metrica": "Latencia p95 del servicio (ms)",
                "valor_observado": round(latencia_p95_ms, 2),
                "umbral": f"> {UMBRAL_LATENCIA_P95_MS} ms",
                "ventana": "1 lote",
                "severidad": "Media",
                "destinatario": "Equipo de Plataforma / API",
                "accion": "Revisar consumo de CPU/memoria del contenedor api.",
            }
        )

    return alertas