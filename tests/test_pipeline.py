"""
Pruebas unitarias reproducibles para src/pipeline.py.
Usan datos escritos a mano para ejecutarse limpiamente en GitHub Actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline import (
    COLUMNAS_PROHIBIDAS,
    FECHA_CORTE,
    FEATURES_FINALES,
    TARGET,
    construir_features,
    crear_preprocesador,
    dividir_datos_temporal,
)


def _crear_muestra_manual() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sesion_key": 1,
                "fecha_realizada": pd.Timestamp("2025-06-15"),
                "fecha_inicio_plan": pd.Timestamp("2025-06-01"),
                "duracion_estimada_min": 60,
                "numero_sesion": 1,
                "sesiones_planificadas": 4,
                "cancelaciones_previas": 1,
                "inasistencias_previas": 2,
                "dias_desde_inicio_plan": 14,
                "edad_en_atencion": 35,
                "codigo_procedimiento": "FILL1",
                "odontologo_key": "1",
                TARGET: 0,
            },
            {
                "sesion_key": 2,
                "fecha_realizada": pd.Timestamp("2025-07-20"),
                "fecha_inicio_plan": pd.Timestamp("2025-07-01"),
                "duracion_estimada_min": 90,
                "numero_sesion": 2,
                "sesiones_planificadas": 3,
                "cancelaciones_previas": 0,
                "inasistencias_previas": 1,
                "dias_desde_inicio_plan": 20,
                "edad_en_atencion": 50,
                "codigo_procedimiento": "CROWN",
                "odontologo_key": "3",
                TARGET: 1,
            },
            {
                "sesion_key": 3,
                "fecha_realizada": pd.Timestamp("2025-08-10"),
                "fecha_inicio_plan": pd.Timestamp("2025-08-01"),
                "duracion_estimada_min": 45,
                "numero_sesion": 1,
                "sesiones_planificadas": 2,
                "cancelaciones_previas": 0,
                "inasistencias_previas": 0,
                "dias_desde_inicio_plan": 9,
                "edad_en_atencion": 28,
                "codigo_procedimiento": "ROOT",
                "odontologo_key": "2",
                TARGET: 1,
            },
        ]
    )


def test_construir_features_calculos_deterministas() -> None:
    df = _crear_muestra_manual()
    out = construir_features(df)

    assert np.isclose(out.loc[0, "progreso_plan"], 0.25)
    assert np.isclose(out.loc[0, "ritmo_dias_por_sesion"], 14.0)
    assert out.loc[0, "friccion_previa"] == 3
    assert out.loc[0, "complejidad_proc"] == "restaurativo_corto"
    assert out.loc[1, "complejidad_proc"] == "extenso_especializado"


def test_sin_fuga_en_features_finales() -> None:
    fugas = set(FEATURES_FINALES).intersection(COLUMNAS_PROHIBIDAS | {TARGET})
    assert not fugas, f"Columnas prohibidas detectadas: {fugas}"


def test_dividir_datos_temporal_respeta_corte() -> None:
    df = _crear_muestra_manual()
    X_train, X_test, y_train, y_test = dividir_datos_temporal(df, fecha_corte=FECHA_CORTE)

    assert len(X_train) == 2
    assert len(X_test) == 1
    assert list(X_train.columns) == FEATURES_FINALES
    assert list(X_test.columns) == FEATURES_FINALES
    assert set(y_train.tolist()).issubset({0, 1})
    assert set(y_test.tolist()).issubset({0, 1})


def test_preprocesador_imputa_nulos_y_tolera_categoria_nueva() -> None:
    df = _crear_muestra_manual()
    X_train, X_test, _, _ = dividir_datos_temporal(df, fecha_corte=FECHA_CORTE)

    X_test_sucio = X_test.copy()
    X_test_sucio.loc[X_test_sucio.index[0], "duracion_estimada_min"] = np.nan
    X_test_sucio.loc[X_test_sucio.index[0], "codigo_procedimiento"] = "PROCEDIMIENTO_NUEVO"

    pre = crear_preprocesador(escalar_numericas=True)
    pre.fit(X_train)
    matriz = pre.transform(X_test_sucio)

    assert matriz.shape[0] == 1
    assert not np.isnan(matriz).any()