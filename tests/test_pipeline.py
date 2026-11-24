"""
Pruebas unitarias reproducibles para src/pipeline.py.

Verifican:
1. Ingeniería de características determinista.
2. Exclusión de variables con fuga de información.
3. División temporal con maduración de 60 días.
4. Imputación y tratamiento de categorías desconocidas.
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
    """Genera registros para train, gap temporal y hold-out."""

    return pd.DataFrame(
        [
            # TRAIN: 2025-05-15 <= 2025-06-02
            {
                "sesion_key": 1,
                "fecha_realizada": pd.Timestamp("2025-05-15"),
                "fecha_inicio_plan": pd.Timestamp("2025-05-01"),
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
            # GAP: 2025-07-20 se excluye
            {
                "sesion_key": 2,
                "fecha_realizada": pd.Timestamp("2025-07-20"),
                "fecha_inicio_plan": pd.Timestamp("2025-07-01"),
                "duracion_estimada_min": 90,
                "numero_sesion": 2,
                "sesiones_planificadas": 3,
                "cancelaciones_previas": 0,
                "inasistencias_previas": 1,
                "dias_desde_inicio_plan": 19,
                "edad_en_atencion": 50,
                "codigo_procedimiento": "CROWN",
                "odontologo_key": "3",
                TARGET: 1,
            },
            # TEST: 2025-08-10 >= 2025-08-01
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
    """Verifica el cálculo de las cuatro features derivadas."""

    df = _crear_muestra_manual()
    out = construir_features(df)

    assert np.isclose(out.loc[0, "progreso_plan"], 0.25)
    assert np.isclose(out.loc[0, "ritmo_dias_por_sesion"], 14.0)
    assert out.loc[0, "friccion_previa"] == 3

    assert (
        out.loc[0, "complejidad_proc"]
        == "restaurativo_corto"
    )

    assert (
        out.loc[1, "complejidad_proc"]
        == "extenso_especializado"
    )


def test_sin_fuga_en_features_finales() -> None:
    """Comprueba que el modelo no utilice columnas prohibidas."""

    fugas = set(FEATURES_FINALES).intersection(
        COLUMNAS_PROHIBIDAS | {TARGET}
    )

    assert not fugas, (
        f"Columnas prohibidas detectadas: {fugas}"
    )


def test_dividir_datos_temporal_respeta_corte() -> None:
    """Verifica train, gap y hold-out con horizonte de 60 días."""

    df = _crear_muestra_manual()

    X_train, X_test, y_train, y_test = dividir_datos_temporal(
        df,
        fecha_corte=FECHA_CORTE,
    )

    # Una observación para entrenamiento y otra para prueba.
    # La observación intermedia queda excluida.
    assert len(X_train) == 1
    assert len(X_test) == 1

    assert len(df) - len(X_train) - len(X_test) == 1

    # Verificar que las features sean las esperadas.
    assert list(X_train.columns) == FEATURES_FINALES
    assert list(X_test.columns) == FEATURES_FINALES

    # Verificar las observaciones seleccionadas.
    assert X_train["duracion_estimada_min"].iloc[0] == 60
    assert X_test["duracion_estimada_min"].iloc[0] == 45

    # La observación del gap no debe aparecer.
    assert 90 not in X_train["duracion_estimada_min"].values
    assert 90 not in X_test["duracion_estimada_min"].values

    # Comprobar las etiquetas.
    assert y_train.tolist() == [0]
    assert y_test.tolist() == [1]

    # Verificar las condiciones temporales.
    corte = pd.Timestamp(FECHA_CORTE)
    limite_train = corte - pd.Timedelta(days=60)

    fechas_train = df.loc[y_train.index, "fecha_realizada"]
    fechas_test = df.loc[y_test.index, "fecha_realizada"]

    assert (fechas_train <= limite_train).all()
    assert (fechas_test >= corte).all()


def test_preprocesador_imputa_nulos_y_tolera_categoria_nueva() -> None:
    """Verifica imputación y OneHotEncoder con categorías nuevas."""

    df = _crear_muestra_manual()

    X_train, X_test, _, _ = dividir_datos_temporal(
        df,
        fecha_corte=FECHA_CORTE,
    )

    X_test_sucio = X_test.copy()

    # Introducir un valor faltante en una variable numérica.
    X_test_sucio.loc[
        X_test_sucio.index[0],
        "duracion_estimada_min"
    ] = np.nan

    # Introducir una categoría no vista durante entrenamiento.
    X_test_sucio.loc[
        X_test_sucio.index[0],
        "codigo_procedimiento"
    ] = "PROCEDIMIENTO_NUEVO"

    pre = crear_preprocesador(escalar_numericas=True)

    # El preprocesador se ajusta solo con entrenamiento.
    pre.fit(X_train)

    # Transformar el conjunto de prueba sin reajustar.
    matriz = pre.transform(X_test_sucio)

    assert matriz.shape[0] == 1
    assert matriz.shape[1] > 0

    # La imputación debe eliminar los valores faltantes.
    assert np.isfinite(matriz).all()
