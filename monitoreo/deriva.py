"""
Módulo de monitoreo de calidad de datos, deriva de datos (PSI, KS, Chi²)
y deriva conceptual para OralSolutions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

# Referencias de validación cruzada temporal del PSet 3 (@champion MLP)
PR_AUC_CV_MEAN = 0.98330
PR_AUC_CV_STD = 0.00342

EPSILON = 1e-6


def calcular_psi_numerica(
    referencia: pd.Series,
    actual: pd.Series,
    n_bins: int = 10,
) -> float:
    """Calcula el Population Stability Index (PSI) para una variable numérica."""
    ref = pd.to_numeric(referencia, errors="coerce").dropna().to_numpy()
    act = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(act) == 0:
        return 0.0

    # Cortes por cuantiles sobre referencia (eliminando duplicados)
    cortes = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(cortes) < 3:
        # Variable numérica discreta con pocos valores únicos (ej. numero_sesion)
        return calcular_psi_categorica(referencia.astype(str), actual.astype(str))

    cortes[0] = -np.inf
    cortes[-1] = np.inf

    freq_ref = np.histogram(ref, bins=cortes)[0] / len(ref)
    freq_act = np.histogram(act, bins=cortes)[0] / len(act)

    freq_ref = np.clip(freq_ref, EPSILON, None)
    freq_act = np.clip(freq_act, EPSILON, None)

    psi = float(np.sum((freq_act - freq_ref) * np.log(freq_act / freq_ref)))
    return round(psi, 6)


def calcular_psi_categorica(referencia: pd.Series, actual: pd.Series) -> float:
    """Calcula el PSI para una variable categórica o discreta."""
    ref_s = referencia.astype(str).dropna()
    act_s = actual.astype(str).dropna()
    if len(ref_s) == 0 or len(act_s) == 0:
        return 0.0

    categorias = sorted(set(ref_s.unique()).union(set(act_s.unique())))
    prop_ref = ref_s.value_counts(normalize=True)
    prop_act = act_s.value_counts(normalize=True)

    psi = 0.0
    for cat in categorias:
        p_r = max(float(prop_ref.get(cat, 0.0)), EPSILON)
        p_a = max(float(prop_act.get(cat, 0.0)), EPSILON)
        psi += (p_a - p_r) * np.log(p_a / p_r)
    return round(float(psi), 6)


def evaluar_deriva_lote(
    df_referencia: pd.DataFrame,
    df_lote: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
) -> pd.DataFrame:
    """
    Evalúa calidad de datos y deriva por variable entre el conjunto de entrenamiento
    (referencia) y un lote de producción:
        - Numéricas: PSI + prueba de Kolmogorov-Smirnov (KS) + % faltantes + % fuera de rango.
        - Categóricas: PSI + prueba de Chi-cuadrado + % faltantes + categorías no vistas.
    """
    filas: list[dict[str, Any]] = []

    for col in features_num:
        ref = df_referencia[col]
        act = df_lote[col]
        pct_nulos = round(float(act.isna().mean() * 100.0), 3)

        min_ref, max_ref = float(ref.min()), float(ref.max())
        fuera_rango = float(((act < min_ref) | (act > max_ref)).mean() * 100.0)

        psi = calcular_psi_numerica(ref, act)
        ks_stat, ks_p = ks_2samp(ref.dropna(), act.dropna())

        filas.append(
            {
                "feature": col,
                "tipo": "numerica",
                "psi": psi,
                "prueba_estadistica": "KS",
                "estadistico": round(float(ks_stat), 5),
                "p_valor": round(float(ks_p), 6),
                "pct_faltantes": pct_nulos,
                "pct_fuera_rango": round(fuera_rango, 3),
                "categorias_no_vistas": 0,
            }
        )

    for col in features_cat:
        ref = df_referencia[col].astype(str)
        act = df_lote[col].astype(str)
        pct_nulos = round(float(df_lote[col].isna().mean() * 100.0), 3)

        no_vistas = len(set(act.unique()) - set(ref.unique()))
        psi = calcular_psi_categorica(ref, act)

        # Tabla de contingencia para prueba Chi-cuadrado
        cats = sorted(set(ref.unique()).union(set(act.unique())))
        c_ref = [int((ref == c).sum()) for c in cats]
        c_act = [int((act == c).sum()) for c in cats]
        tabla = np.array([c_ref, c_act])
        # Filtrar columnas totalmente en cero si existieran
        tabla = tabla[:, tabla.sum(axis=0) > 0]
        if tabla.shape[1] >= 2:
            chi2_stat, chi2_p, _, _ = chi2_contingency(tabla)
        else:
            chi2_stat, chi2_p = 0.0, 1.0

        filas.append(
            {
                "feature": col,
                "tipo": "categorica",
                "psi": psi,
                "prueba_estadistica": "Chi2",
                "estadistico": round(float(chi2_stat), 5),
                "p_valor": round(float(chi2_p), 6),
                "pct_faltantes": pct_nulos,
                "pct_fuera_rango": 0.0,
                "categorias_no_vistas": int(no_vistas),
            }
        )

    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def diagnosticar_deriva(
    psi_max: float,
    pr_auc_lote: float | None,
    umbral_psi: float = 0.30,
    umbral_pr_auc: float = 0.96000,
) -> dict[str, Any]:
    """
    Clasifica el estado del lote según la matriz de diagnóstico:
        - Entradas cambian + Desempeño se mantiene -> Deriva de datos sin impacto
        - Entradas cambian + Desempeño cae         -> Deriva de datos con impacto
        - Entradas estables + Desempeño cae        -> Deriva conceptual
        - Entradas estables + Desempeño estable    -> Normal (sin deriva)
    """
    entradas_cambian = psi_max > umbral_psi
    piso_pr_auc = umbral_pr_auc
    desempeno_cae = (pr_auc_lote is not None) and (pr_auc_lote < piso_pr_auc)

    if entradas_cambian and not desempeno_cae:
        diagnostico = "Deriva de datos sin impacto"
        accion = "Investigar fuente de datos y vigilar siguiente lote; NO reentrenar todavía."
    elif entradas_cambian and desempeno_cae:
        diagnostico = "Deriva de datos con impacto"
        accion = "Revisar fuente/calidad y disparar reentrenamiento con lotes etiquetados recientes."
    elif (not entradas_cambian) and desempeno_cae:
        diagnostico = "Deriva conceptual"
        accion = "Disparar reentrenamiento prioritario y evaluar cambio en política clínica/umbral."
    else:
        diagnostico = "Normal (sin deriva)"
        accion = "Mantener modelo @champion en operación estándar."

    return {
        "entradas_cambian": entradas_cambian,
        "desempeno_cae": desempeno_cae,
        "piso_pr_auc": piso_pr_auc,
        "diagnostico": diagnostico,
        "accion": accion,
    }