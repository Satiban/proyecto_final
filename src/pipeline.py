from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Constantes del esquema y corte temporal
# ---------------------------------------------------------------------------
SEMILLA = 42
TARGET = "abandono_60d"
FECHA_EVENTO = "fecha_realizada"
FECHAS = ["fecha_realizada", "fecha_inicio_plan"]

# División temporal con horizonte de 60 días:
# Train: hasta 2025-06-02 (1195 observaciones).
# Gap: 2025-06-03 a 2025-07-31 (356 observaciones).
# Test: desde 2025-08-01 (826 observaciones).
FECHA_CORTE = "2025-08-01"

FEATURES_NUM = [
    "duracion_estimada_min",
    "numero_sesion",
    "sesiones_planificadas",
    "cancelaciones_previas",
    "inasistencias_previas",
    "dias_desde_inicio_plan",
    "edad_en_atencion",
    "progreso_plan",
    "ritmo_dias_por_sesion",
    "friccion_previa",
]

FEATURES_CAT = [
    "codigo_procedimiento",
    "odontologo_key",
    "complejidad_proc",
]

FEATURES_FINALES = FEATURES_NUM + FEATURES_CAT

COLUMNAS_PROHIBIDAS = {
    "sesion_key",
    "paciente_key",
    "plan_tratamiento_id",
    "cita_id",
    "procedimiento_key",
    "nombre_procedimiento",
    "fecha_realizada",
    "fecha_inicio_plan",
    "sesiones_previas",
    "dia_semana_atencion",
    "mes_atencion",
    "es_ultima_observada",
    "es_ultima_planificada",
    "ventana_incompleta",
}


def resolver_ruta_datos(ruta: str | Path | None = None) -> Path:
    """Localiza obt_riesgo_abandono.csv ya sea ejecutando desde modelado/ o proyecto/."""
    if ruta is not None:
        p = Path(ruta)
        if p.exists():
            return p
        raise FileNotFoundError(f"No se encontró el archivo especificado: {p}")

    base_script = Path(__file__).resolve().parent.parent
    candidatas = [
        Path("../data/processed/obt_riesgo_abandono.csv"),
        Path("data/processed/obt_riesgo_abandono.csv"),
        base_script / "data" / "processed" / "obt_riesgo_abandono.csv",
    ]
    for p in candidatas:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró data/processed/obt_riesgo_abandono.csv")


def obtener_hash_datos(ruta: str | Path | None = None) -> str:
    """Calcula la huella SHA-256 del archivo de datos limpio para trazabilidad."""
    archivo = resolver_ruta_datos(ruta)
    return hashlib.sha256(archivo.read_bytes()).hexdigest()


def cargar_datos(ruta: str | Path | None = None) -> pd.DataFrame:
    """Carga la OBT limpia y la ordena cronológicamente por sesión."""
    archivo = resolver_ruta_datos(ruta)
    df = pd.read_csv(archivo, parse_dates=FECHAS)
    df = df.sort_values([FECHA_EVENTO, "sesion_key"]).reset_index(drop=True)
    return df


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las 4 variables derivadas sin fuga temporal (solo datos conocidos
    al cierre de fecha_realizada) y valida que no existan columnas prohibidas.
    """
    datos = df.copy()

    # 1. Fracción del plan completada al cierre de la sesión actual (Hallazgo 1)
    datos["progreso_plan"] = datos["numero_sesion"] / datos["sesiones_planificadas"]

    # 2. Ritmo promedio de días transcurridos por sesión atendida (Hallazgo 2)
    datos["ritmo_dias_por_sesion"] = datos["dias_desde_inicio_plan"] / datos["numero_sesion"]

    # 3. Historial acumulado de incumplimiento previo (Hallazgo 2)
    datos["friccion_previa"] = datos["cancelaciones_previas"] + datos["inasistencias_previas"]

    # 4. Agrupación clínica de complejidad del procedimiento (Hallazgo 3)
    datos["complejidad_proc"] = np.where(
        datos["codigo_procedimiento"].isin(["EXTRACT", "FILL1", "FILL2"]),
        "restaurativo_corto",
        "extenso_especializado",
    )

    # Asegurar tipo string en categóricas nominales
    for col in FEATURES_CAT:
        datos[col] = datos[col].astype("string").astype(object)
    # Verificación estricta contra fuga de datos
    fugas = set(FEATURES_FINALES).intersection(COLUMNAS_PROHIBIDAS | {TARGET})
    assert not fugas, f"Fuga de datos detectada en FEATURES_FINALES: {fugas}"

    return datos

def dividir_datos_temporal(
    df: pd.DataFrame,
    fecha_corte: str = FECHA_CORTE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    División temporal con maduración del target a 60 días.

    Entrenamiento:
        fecha_realizada + 60 días <= fecha_corte.

    Hold-out:
        fecha_realizada >= fecha_corte.

    Las observaciones intermedias quedan excluidas para
    evitar entrenar con etiquetas no disponibles al corte.
    """

    datos = construir_features(df)

    corte_ts = pd.Timestamp(fecha_corte)
    horizonte = pd.Timedelta(days=60)
    limite_train = corte_ts - horizonte

    mask_train = datos[FECHA_EVENTO] <= limite_train
    mask_test = datos[FECHA_EVENTO] >= corte_ts

    train_df = datos.loc[mask_train].copy()
    test_df = datos.loc[mask_test].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "La fecha de corte produce un conjunto vacío."
        )

    if train_df[FECHA_EVENTO].max() > limite_train:
        raise ValueError(
            "Existen etiquetas de entrenamiento sin madurar."
        )

    if train_df[FECHA_EVENTO].max() >= test_df[FECHA_EVENTO].min():
        raise ValueError(
            "Existe solapamiento temporal entre train y test."
        )

    X_train = train_df[FEATURES_FINALES].copy()
    y_train = train_df[TARGET].astype(int).copy()

    X_test = test_df[FEATURES_FINALES].copy()
    y_test = test_df[TARGET].astype(int).copy()

    print("=== DIVISIÓN TEMPORAL ===")
    print(f"Fecha de corte       : {corte_ts.date()}")
    print(f"Horizonte            : 60 días")
    print(f"Límite entrenamiento : {limite_train.date()}")
    print(f"Train                : {len(train_df):,}")
    print(f"Hold-out             : {len(test_df):,}")
    print(
        f"Gap excluido         : "
        f"{len(datos) - len(train_df) - len(test_df):,}"
    )

    return X_train, X_test, y_train, y_test

def obtener_datos_divididos(
    ruta: str | Path | None = None,
    fecha_corte: str = FECHA_CORTE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Punto de entrada único para que todos los notebooks obtengan exactamente la misma división."""
    df = cargar_datos(ruta)
    return dividir_datos_temporal(df, fecha_corte=fecha_corte)


def crear_preprocesador(escalar_numericas: bool = True) -> ColumnTransformer:
    """
    Construye el ColumnTransformer que se ajustará únicamente con X_train:
    - Numéricas: imputación por mediana + RobustScaler (robusto a colas largas).
    - Categóricas: imputación por moda + OneHotEncoder(handle_unknown='ignore').
    """
    pasos_num = [("imputador", SimpleImputer(strategy="median"))]
    if escalar_numericas:
        pasos_num.append(("escalador", RobustScaler()))

    pipe_num = Pipeline(steps=pasos_num)

    pipe_cat = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", pipe_num, FEATURES_NUM),
            ("cat", pipe_cat, FEATURES_CAT),
        ],
        remainder="drop",
    )


def construir_pipeline(
    modelo: BaseEstimator,
    escalar_numericas: bool = True,
) -> Pipeline:
    """Encapsula preprocesador + estimador en un único Pipeline reproducible."""
    return Pipeline(
        steps=[
            ("preprocesador", crear_preprocesador(escalar_numericas=escalar_numericas)),
            ("modelo", modelo),
        ]
    )


if __name__ == "__main__":
    ruta_csv = resolver_ruta_datos()
    sha256 = obtener_hash_datos(ruta_csv)
    df_raw = cargar_datos(ruta_csv)
    X_train, X_test, y_train, y_test = dividir_datos_temporal(df_raw)

    preprocesador = crear_preprocesador(escalar_numericas=True)
    X_train_proc = preprocesador.fit_transform(X_train)
    X_test_proc = preprocesador.transform(X_test)

    print("=== VERIFICACIÓN PIPELINE PROYECTO (SIN FUGA) ===")
    print(f"Archivo fuente   : {ruta_csv.resolve()}")
    print(f"SHA-256 dataset  : {sha256[:16]}...{sha256[-8:]}")
    print(f"Fecha de corte   : {FECHA_CORTE}")
    print(f"Train (histórico): {X_train.shape[0]:,} filas | tasa abandono = {y_train.mean():.2%}")
    print(f"Test  (hold-out) : {X_test.shape[0]:,} filas | tasa abandono = {y_test.mean():.2%}")
    print(f"Matriz tras OHE  : train={X_train_proc.shape} | test={X_test_proc.shape}")
    print("[OK] Pipeline ejecutado limpiamente sin fuga de datos.")