"""Estadísticas de deriva; no consulta targets ni dispara reentrenamientos."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

UMBRAL_PSI_ADVERTENCIA = 0.10
UMBRAL_PSI_CRITICO = 0.25
EPSILON = 1e-6


def _psi(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return None
    a = np.maximum(a/a.sum(), EPSILON)
    b = np.maximum(b/b.sum(), EPSILON)
    a, b = a/a.sum(), b/b.sum()
    return float(np.sum((b-a)*np.log(b/a)))


def _categorias(ref, act):
    ref, act = ref.dropna().astype(str), act.dropna().astype(str)
    cats = sorted(set(ref) | set(act))
    return np.array([[int((s == c).sum()) for c in cats] for s in (ref, act)])


def calcular_psi_categorica(referencia, actual):
    tabla = _categorias(referencia, actual)
    return _psi(*tabla)


def _numerica(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def calcular_psi_numerica(referencia, actual, n_bins=10):
    ref, act = _numerica(referencia).dropna(), _numerica(actual).dropna()
    if ref.empty or act.empty:
        return None
    # Variables discretas: conservar explícitamente el cero y los otros valores.
    if ref.nunique() <= n_bins:
        return calcular_psi_categorica(ref, act)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins+1)))
    if len(cortes) < 3:
        return calcular_psi_categorica(ref, act)
    cortes = np.r_[-np.inf, cortes[1:-1], np.inf]
    return _psi(np.histogram(ref, cortes)[0], np.histogram(act, cortes)[0])


def evaluar_deriva_lote(df_referencia, df_lote, features_num, features_cat):
    filas = []
    for col in features_num + features_cat:
        numerica = col in features_num
        ref, act = df_referencia[col], df_lote[col]
        if numerica:
            ref, act = _numerica(ref), _numerica(act)
        r, a = ref.dropna(), act.dropna()
        stat = p = None
        valida = False
        fuera = None
        nuevas = 0
        pct_nuevas = 0.0
        if numerica:
            psi = calcular_psi_numerica(ref, act)
            if not r.empty and not a.empty:
                stat, p = ks_2samp(r, a)
                valida = True
                fuera = float(((a < r.min()) | (a > r.max())).sum()/len(act)*100)
        else:
            psi = calcular_psi_categorica(ref, act)
            r, a = r.astype(str), a.astype(str)
            nuevas = len(set(a)-set(r))
            pct_nuevas = float((~a.isin(set(r))).sum()/len(act)*100) if len(act) else None
            tabla = _categorias(r, a)
            if not r.empty and not a.empty and tabla.shape[1] >= 2:
                stat, p, _, esperadas = chi2_contingency(tabla, correction=False)
                valida = bool((esperadas >= 1).all() and (esperadas >= 5).mean() >= .8)
            fuera = 0.0
        filas.append(dict(feature=col, tipo='numerica' if numerica else 'categorica',
            psi=psi, prueba_estadistica='KS' if numerica else 'Chi2',
            estadistico=None if stat is None else float(stat),
            p_valor=None if p is None else float(p), prueba_valida=valida,
            pct_faltantes=float(act.isna().mean()*100) if len(act) else None,
            pct_fuera_rango=fuera, categorias_no_vistas=nuevas,
            pct_categorias_no_vistas=pct_nuevas))
    return pd.DataFrame(filas).sort_values('psi', ascending=False, na_position='last').reset_index(drop=True)


def diagnosticar_deriva(psi_max, pr_auc_lote, umbral_psi=UMBRAL_PSI_CRITICO, *, media_cv, std_cv):
    piso = max(0.0, media_cv-std_cv)
    cambio = None if psi_max is None else psi_max >= umbral_psi
    cae = None if pr_auc_lote is None else pr_auc_lote < piso
    if cambio is None or cae is None:
        diagnostico = 'Evaluación parcial: desempeño o deriva no disponible'
    elif cambio and cae:
        diagnostico = 'Deriva de datos con deterioro observado'
    elif cambio:
        diagnostico = 'Deriva de datos sin deterioro observado'
    elif cae:
        diagnostico = 'Posible deriva conceptual u otra causa'
    else:
        diagnostico = 'Sin deterioro ni deriva significativa detectada'
    return dict(entradas_cambian=cambio, desempeno_cae=cae, piso_pr_auc=piso,
                diagnostico=diagnostico, accion='Revisar evidencia; no reentrenar automáticamente por un solo lote.')