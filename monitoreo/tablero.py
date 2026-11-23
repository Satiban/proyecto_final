"""
Tablero de monitoreo operativo MLOps y Panel de Recepción — OralSolutions (PSet 4).
Ejecución desde :
    streamlit run monitoreo/tablero.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

BASE_MONITOREO = Path(__file__).resolve().parent
BASE_PROYECTO = BASE_MONITOREO.parent
DOCS_DIR = BASE_PROYECTO / "docs"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UMBRAL_PR_AUC = 0.960
UMBRAL_PSI = 0.30

st.set_page_config(
    page_title="OralSolutions — Monitoreo MLOps y Recepción",
    layout="wide",
)

st.title("OralSolutions — Plataforma de Monitoreo y Priorización de Pacientes")
st.caption(
    "Pestaña 1: Monitoreo MLOps del servicio FastAPI, deriva y reentrenamiento. "
    "Pestaña 2: Panel operativo de recepción para priorizar pacientes en riesgo de abandono."
)


def consultar_api(endpoint: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


@st.cache_data(ttl=60)
def cargar_predicciones_expandidas(ruta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv)
    entradas = df["entradas_json"].apply(json.loads).apply(pd.Series)
    df_exp = pd.concat([df.drop(columns=["entradas_json"]), entradas], axis=1)
    return df_exp


tab_monitoreo, tab_recepcion = st.tabs(
    ["Monitoreo MLOps", "Panel de Recepción (Pacientes)"]
)

# ===========================================================================
# PESTAÑA 1: MONITOREO MLOPS (SERVICIO, DERIVA, NEGOCIO Y REENTRENAMIENTO)
# ===========================================================================
with tab_monitoreo:
    # -----------------------------------------------------------------------
    # 1. Estado en vivo del servicio (FastAPI + MLflow @champion)
    # -----------------------------------------------------------------------
    st.header("1. Estado en vivo del servicio")

    salud = consultar_api("/salud")
    modelo = consultar_api("/modelo")
    metricas = consultar_api("/metricas")
    alertas_api = consultar_api("/alertas")

    if salud and modelo and metricas:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estado API", salud["estado"].upper())
        c2.metric("Modelo activo", f"v{modelo['version']} (@{modelo['alias']})")
        c3.metric("Predicciones servidas", metricas["modelo"]["total_predicciones"])
        c4.metric("Latencia p95 (ms)", f"{metricas['sistema']['latencia_p95_ms']:.2f}")
        c5.metric("Tasa contactar", f"{metricas['modelo']['tasa_contactar']:.1%}")

        st.caption(
            f"run_id: `{modelo['run_id']}` | Umbral: `{modelo['umbral_decision']}` | "
            f"Modo respaldo: `{modelo['modo_respaldo']}` | "
            f"Errores HTTP (4xx/5xx): `{metricas['sistema']['errores_4xx']}/{metricas['sistema']['errores_5xx']}`"
        )

        if alertas_api and alertas_api["total_alertas"] > 0:
            st.error(f"Alertas activas en el servicio: {alertas_api['total_alertas']}")
            st.dataframe(pd.DataFrame(alertas_api["alertas"]), use_container_width=True)
        else:
            st.success("Sin alertas operativas activas en el servicio en vivo.")
    else:
        st.warning(
            f"No se pudo conectar a la API en `{API_URL}`. "
            "Mostrando únicamente los resultados históricos guardados en `docs/`."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # 2. Monitoreo por lote: Desempeño (PR-AUC) y Deriva de datos (PSI)
    # -----------------------------------------------------------------------
    ruta_mon = DOCS_DIR / "monitoreo.csv"
    ruta_alertas = DOCS_DIR / "alertas.csv"
    ruta_reent_json = BASE_MONITOREO / "ultimo_reentrenamiento.json"

    if not ruta_mon.exists():
        st.error(
            "No existe `docs/monitoreo.csv`. Ejecuta primero `python monitoreo/simular_produccion.py`."
        )
    else:
        df_mon = pd.read_csv(ruta_mon)
        orden_lotes = df_mon["lote_id"].tolist()

        st.header("2. Evolución por lote: Desempeño y Deriva")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.subheader("PR-AUC por lote (Deriva conceptual)")
            linea_pr = (
                alt.Chart(df_mon)
                .mark_line(point=True)
                .encode(
                    x=alt.X("lote_id:N", sort=orden_lotes, title="Lote"),
                    y=alt.Y("pr_auc:Q", scale=alt.Scale(domain=[0.45, 1.02]), title="PR-AUC"),
                    tooltip=["lote_id", "pr_auc", "diagnostico", "n_alertas"],
                )
            )
            regla_pr = (
                alt.Chart(pd.DataFrame({"y": [UMBRAL_PR_AUC]}))
                .mark_rule(color="red", strokeDash=[6, 4])
                .encode(y="y:Q")
            )
            st.altair_chart(linea_pr + regla_pr, use_container_width=True)
            st.caption(
                f"Línea roja punteada: umbral mínimo calibrado `PR-AUC = {UMBRAL_PR_AUC:.3f}`."
            )

        with col_der:
            st.subheader("PSI máximo por lote (Deriva de datos)")
            barras_psi = (
                alt.Chart(df_mon)
                .mark_bar()
                .encode(
                    x=alt.X("lote_id:N", sort=orden_lotes, title="Lote"),
                    y=alt.Y("psi_max:Q", title="PSI máximo"),
                    color=alt.condition(
                        alt.datum.psi_max > UMBRAL_PSI,
                        alt.value("#d62728"),
                        alt.value("#1f77b4"),
                    ),
                    tooltip=["lote_id", "psi_max", "feature_psi_max", "diagnostico"],
                )
            )
            regla_psi = (
                alt.Chart(pd.DataFrame({"y": [UMBRAL_PSI]}))
                .mark_rule(color="red", strokeDash=[6, 4])
                .encode(y="y:Q")
            )
            st.altair_chart(barras_psi + regla_psi, use_container_width=True)
            st.caption(
                f"Línea roja punteada: umbral crítico calibrado `PSI = {UMBRAL_PSI:.2f}`."
            )

        # -------------------------------------------------------------------
        # 3. Impacto de negocio: Modelo vs. Líneas base
        # -------------------------------------------------------------------
        st.header("3. Impacto de negocio por lote (USD)")

        df_costos = df_mon.melt(
            id_vars=["lote_id"],
            value_vars=[
                "costo_modelo_usd",
                "costo_regla_1_sesion_usd",
                "costo_llamar_todos_usd",
            ],
            var_name="estrategia",
            value_name="costo_usd",
        )
        etiquetas = {
            "costo_modelo_usd": "Modelo @champion",
            "costo_regla_1_sesion_usd": "Regla 1 sesión restante",
            "costo_llamar_todos_usd": "Llamar a todos",
        }
        df_costos["estrategia"] = df_costos["estrategia"].map(etiquetas)

        chart_costos = (
            alt.Chart(df_costos)
            .mark_line(point=True)
            .encode(
                x=alt.X("lote_id:N", sort=orden_lotes, title="Lote"),
                y=alt.Y("costo_usd:Q", title="Costo de errores (USD)"),
                color=alt.Color("estrategia:N", title="Estrategia"),
                tooltip=["lote_id", "estrategia", "costo_usd"],
            )
        )
        st.altair_chart(chart_costos, use_container_width=True)

        with st.expander("Ver tabla completa de monitoreo por lote"):
            st.dataframe(df_mon, use_container_width=True)

        st.divider()

        # -------------------------------------------------------------------
        # 4. Alertas disparadas y último ciclo de reentrenamiento
        # -------------------------------------------------------------------
        col_al, col_re = st.columns(2)

        with col_al:
            st.header("4. Alertas registradas")
            if ruta_alertas.exists() and ruta_alertas.stat().st_size > 0:
                df_al = pd.read_csv(ruta_alertas)
                st.dataframe(
                    df_al[
                        ["lote_id", "metrica", "valor_observado", "umbral", "severidad", "accion"]
                    ],
                    use_container_width=True,
                )
            else:
                st.info("No se registraron alertas en `docs/alertas.csv`.")

        with col_re:
            st.header("5. Último ciclo de reentrenamiento")
            if ruta_reent_json.exists():
                res = json.loads(ruta_reent_json.read_text(encoding="utf-8"))
                champ = res["champion_actual"]
                chall = res["challenger_nuevo"]

                tabla_comp = pd.DataFrame(
                    [
                        {
                            "Rol": "@champion (activo)",
                            "Versión": f"v{champ['version']}",
                            "PR-AUC": champ["pr_auc"],
                            "Costo (USD)": champ["costo_errores_usd"],
                            "Balance (USD)": champ["balance_neto_usd"],
                        },
                        {
                            "Rol": "@challenger (candidato)",
                            "Versión": f"v{chall['version']}",
                            "PR-AUC": chall["pr_auc"],
                            "Costo (USD)": chall["costo_errores_usd"],
                            "Balance (USD)": chall["balance_neto_usd"],
                        },
                    ]
                )
                st.dataframe(tabla_comp, use_container_width=True)
                st.markdown(
                    f"- **Lote de evaluación:** `{res['lote_evaluacion']}`\n"
                    f"- **Ventana de entrenamiento:** {res['ventana_entrenamiento']}\n"
                    f"- **¿Cumple regla de promoción?** `{res['cumple_regla_promocion']}`\n"
                    f"- **Decisión humana:** `{res['decision_humana']}` — {res['motivo_decision']}"
                )
            else:
                st.info("Ejecuta `python src/reentrenar.py` para visualizar el reentrenamiento.")


# ===========================================================================
# PESTAÑA 2: PANEL DE RECEPCIÓN (PRIORIZACIÓN DE PACIENTES — MVP PSET 1)
# ===========================================================================
with tab_recepcion:
    st.header("Lista de prioridad para seguimiento preventivo")
    st.caption(
        "Ordenada de mayor a menor riesgo de abandono a 60 días para apoyar las llamadas y "
        "recordatorios del equipo de recepción."
    )

    ruta_preds = DOCS_DIR / "predicciones.csv"
    if not ruta_preds.exists():
        st.error(
            "No existe `docs/predicciones.csv`. Ejecuta primero `python monitoreo/simular_produccion.py`."
        )
    else:
        df_preds = cargar_predicciones_expandidas(ruta_preds)

        f1, f2, f3 = st.columns(3)
        lotes_disponibles = df_preds["lote_id"].unique().tolist()
        indice_default = lotes_disponibles.index("L08") if "L08" in lotes_disponibles else 0
        lote_sel = f1.selectbox("Lote de atención", lotes_disponibles, index=indice_default)
        decision_sel = f2.multiselect(
            "Acción recomendada",
            ["contactar", "no_contactar"],
            default=["contactar"],
        )
        procs_disp = sorted(df_preds["codigo_procedimiento"].dropna().unique().tolist())
        proc_sel = f3.multiselect("Procedimiento", procs_disp, default=procs_disp)

        df_fil = df_preds[
            (df_preds["lote_id"] == lote_sel)
            & (df_preds["decision"].isin(decision_sel))
            & (df_preds["codigo_procedimiento"].isin(proc_sel))
        ].sort_values("probabilidad", ascending=False)

        total_lote = int((df_preds["lote_id"] == lote_sel).sum())
        total_contactar = int(
            ((df_preds["lote_id"] == lote_sel) & (df_preds["decision"] == "contactar")).sum()
        )
        riesgo_medio = float(df_fil["probabilidad"].mean()) if not df_fil.empty else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("Sesiones en el lote", total_lote)
        k2.metric("Pacientes a contactar", total_contactar)
        k3.metric("Riesgo promedio filtrado", f"{riesgo_medio:.1%}")

        cols_vista = [
            "fecha_realizada",
            "probabilidad",
            "decision",
            "codigo_procedimiento",
            "numero_sesion",
            "sesiones_planificadas",
            "dias_desde_inicio_plan",
            "cancelaciones_previas",
            "inasistencias_previas",
            "edad_en_atencion",
            "id_prediccion",
        ]
        cols_presentes = [c for c in cols_vista if c in df_fil.columns]
        st.dataframe(df_fil[cols_presentes], use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Evaluación en vivo de un paciente nuevo (POST /predecir)
    # -----------------------------------------------------------------------
    st.header("Evaluar riesgo de un paciente en vivo")
    st.caption("Envía una consulta directa a `POST /predecir` usando el modelo `@champion` activo.")

    with st.form("form_consulta_paciente"):
        c_a, c_b, c_c = st.columns(3)
        codigo_proc = c_a.selectbox(
            "Procedimiento",
            ["CROWN", "ROOT", "WHITEN", "EXTRACT", "FILL1", "FILL2"],
            index=0,
        )
        duracion_min = c_b.number_input("Duración estimada (min)", min_value=15, max_value=240, value=75)
        odontologo_key = c_c.text_input("ID Odontólogo", value="2")

        c_d, c_e, c_f = st.columns(3)
        sesiones_plan = c_d.number_input("Sesiones planificadas", min_value=2, max_value=12, value=3)
        numero_sesion = c_e.number_input("Número de sesión actual", min_value=1, max_value=11, value=2)
        dias_plan = c_f.number_input("Días desde inicio del plan", min_value=0, max_value=365, value=35)

        c_g, c_h, c_i = st.columns(3)
        cancelaciones = c_g.number_input("Cancelaciones previas", min_value=0, max_value=20, value=1)
        inasistencias = c_h.number_input("Inasistencias previas", min_value=0, max_value=20, value=1)
        edad = c_i.number_input("Edad del paciente", min_value=5, max_value=100, value=42)

        enviar = st.form_submit_button("Calcular riesgo de abandono", type="primary")

    if enviar:
        payload = {
            "duracion_estimada_min": int(duracion_min),
            "numero_sesion": int(numero_sesion),
            "sesiones_planificadas": int(sesiones_plan),
            "cancelaciones_previas": int(cancelaciones),
            "inasistencias_previas": int(inasistencias),
            "dias_desde_inicio_plan": int(dias_plan),
            "edad_en_atencion": int(edad),
            "codigo_procedimiento": codigo_proc,
            "odontologo_key": str(odontologo_key),
        }
        try:
            resp = requests.post(f"{API_URL}/predecir", json=payload, timeout=5)
            if resp.status_code == 200:
                res = resp.json()
                r1, r2, r3 = st.columns(3)
                r1.metric("Probabilidad de abandono (60d)", f"{res['probabilidad']:.2%}")
                r2.metric("Decisión", res["decision"].upper())
                r3.metric("Modelo usado", f"v{res['version_modelo']}")

                if res["decision"] == "contactar":
                    st.warning(f"Acción recomendada: {res['accion']}")
                else:
                    st.success(f"Acción recomendada: {res['accion']}")

                st.caption(
                    f"ID predicción: `{res['id_prediccion']}` | Umbral: `{res['umbral_aplicado']}` | "
                    f"Respaldo activo: `{res['uso_respaldo']}` | Latencia: `{res['latencia_ms']} ms`"
                )
            else:
                st.error(f"Error de validación ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"No se pudo conectar con la API en `{API_URL}`: {exc}")