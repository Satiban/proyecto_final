import base64
import hashlib
import os
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

OUTPUT_CSV = BASE_DIR / "data" / "processed" / "obt_riesgo_abandono.csv"

DATABASE = "ORALSOLUTIONS_DW"
TABLA_OBT = f"{DATABASE}.OBT.OBT_RIESGO_ABANDONO"
TABLA_GOLD = f"{DATABASE}.GOLD.FACT_SESIONES_TRATAMIENTO"

HORIZONTE_DIAS = 60
TARGET = "abandono_60d"


# ============================================================
# CONEXIÓN A SNOWFLAKE
# ============================================================

def conectar_snowflake():
    load_dotenv(ENV_PATH)

    password = base64.b64decode(
        os.environ["SECRET_SNOWFLAKE_PASSWORD"],
        validate=True
    ).decode("utf-8")

    return snowflake.connector.connect(
        account=os.environ["SF_ACCOUNT"],
        user=os.environ["SF_USER"],
        password=password,
        warehouse=os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SF_DATABASE", DATABASE),
        schema=os.getenv("SF_OBT_SCHEMA", "OBT"),
        role=os.getenv("SF_ROLE", "ACCOUNTADMIN"),
    )


def consultar_dataframe(conn, query):
    """Ejecuta una consulta y devuelve un DataFrame."""

    cur = conn.cursor()

    try:
        cur.execute(query)
        columnas = [col[0].lower() for col in cur.description]
        filas = cur.fetchall()

        return pd.DataFrame(filas, columns=columnas)

    finally:
        cur.close()


# ============================================================
# EXTRACCIÓN DE DATOS
# ============================================================

def extraer_datos():
    """
    Recupera:
    1. OBT: puntos candidatos de predicción.
    2. Gold: historial completo de sesiones atendidas.

    Gold se utiliza exclusivamente para construir el target.
    """

    print("1. Conectando a Snowflake...")

    conn = conectar_snowflake()

    try:
        print("2. Descargando OBT_RIESGO_ABANDONO...")

        df_obt = consultar_dataframe(
            conn,
            f"SELECT * FROM {TABLA_OBT}"
        )

        print(
            f"   OBT: {len(df_obt):,} registros | "
            f"{len(df_obt.columns)} columnas"
        )

        print("3. Descargando sesiones completadas desde Gold...")

        df_gold = consultar_dataframe(
            conn,
            f"""
            SELECT
                SESION_KEY,
                PLAN_TRATAMIENTO_ID,
                NUMERO_SESION,
                FECHA_REALIZADA
            FROM {TABLA_GOLD}
            WHERE INDICADOR_COMPLETADA = 1
              AND FECHA_REALIZADA IS NOT NULL
            """
        )

        print(f"   Gold: {len(df_gold):,} sesiones completadas")

    finally:
        conn.close()

    return df_obt, df_gold


# ============================================================
# VALIDACIÓN DE FUENTES
# ============================================================

def validar_fuentes(df_obt, df_gold):
    """Comprueba integridad y coherencia de las fuentes."""

    if df_obt.empty or df_gold.empty:
        raise ValueError("La OBT o Gold no contienen datos.")

    for nombre, df in [("OBT", df_obt), ("Gold", df_gold)]:

        if df["sesion_key"].isna().any():
            raise ValueError(f"{nombre}: existen SESION_KEY nulos.")

        if df["sesion_key"].duplicated().any():
            raise ValueError(f"{nombre}: existen SESION_KEY duplicados.")

        if df["plan_tratamiento_id"].isna().any():
            raise ValueError(
                f"{nombre}: existen PLAN_TRATAMIENTO_ID nulos."
            )

    if df_obt["fecha_realizada"].isna().any():
        raise ValueError("La OBT contiene fechas de atención nulas.")

    if df_obt["fecha_inicio_plan"].isna().any():
        raise ValueError("La OBT contiene fechas de inicio nulas.")

    if not df_obt["sesion_key"].isin(df_gold["sesion_key"]).all():
        raise ValueError(
            "Hay sesiones de referencia de la OBT "
            "que no aparecen como completadas en Gold."
        )

    print("[OK] Integridad de fuentes verificada.")


# ============================================================
# CONSTRUCCIÓN DEL TARGET
# ============================================================

def construir_dataset(df_obt, df_gold):
    """
    Construye abandono_60d.

    0 = existe al menos una sesión posterior completada
        del mismo tratamiento dentro de 60 días.

    1 = no existe continuidad dentro de 60 días.

    Las sesiones posteriores se utilizan solo para etiquetar.
    Nunca se incorporan como predictores.
    """

    print("4. Preparando fechas y controles temporales...")

    obt = df_obt.copy()
    gold = df_gold.copy()

    obt["fecha_realizada"] = pd.to_datetime(
        obt["fecha_realizada"]
    )

    obt["fecha_inicio_plan"] = pd.to_datetime(
        obt["fecha_inicio_plan"]
    )

    gold["fecha_realizada"] = pd.to_datetime(
        gold["fecha_realizada"]
    )

    # Fecha máxima del historial de referencia
    fecha_maxima = obt["fecha_realizada"].max()

    # Comprobar cobertura temporal de Gold
    fecha_maxima_gold = gold["fecha_realizada"].max()

    if fecha_maxima_gold < fecha_maxima:
        raise ValueError(
            "Gold no alcanza la fecha máxima de la OBT. "
            "No es seguro construir etiquetas a 60 días."
        )

    # Límite conservador de observación
    limite_observacion = (
        fecha_maxima - pd.Timedelta(days=HORIZONTE_DIAS)
    )

    print(f"   Última fecha OBT   : {fecha_maxima.date()}")
    print(f"   Última fecha Gold  : {fecha_maxima_gold.date()}")
    print(f"   Horizonte          : {HORIZONTE_DIAS} días")
    print(f"   Última fecha válida: {limite_observacion.date()}")

    # --------------------------------------------------------
    # 1. Excluir observaciones sin horizonte completo
    # --------------------------------------------------------

    sin_horizonte = (
        obt["fecha_realizada"] > limite_observacion
    ).sum()

    obt = obt[
        obt["fecha_realizada"] <= limite_observacion
    ].copy()

    # --------------------------------------------------------
    # 2. Excluir fechas inconsistentes
    # --------------------------------------------------------

    fechas_invalidas = (
        obt["fecha_realizada"] < obt["fecha_inicio_plan"]
    )

    n_fechas_invalidas = int(fechas_invalidas.sum())

    obt = obt.loc[~fechas_invalidas].copy()

    # --------------------------------------------------------
    # 3. Obtener siguiente sesión completada desde Gold
    # --------------------------------------------------------

    print("5. Identificando continuidad real desde Gold...")

    candidatos = obt[
        ["sesion_key", "plan_tratamiento_id", "fecha_realizada"]
    ].merge(
        gold[
            ["sesion_key", "plan_tratamiento_id", "fecha_realizada"]
        ],
        on="plan_tratamiento_id",
        how="left",
        suffixes=("_referencia", "_gold")
    )

    # Solo sesiones del mismo tratamiento en fecha posterior
    posteriores = candidatos[
        (candidatos["sesion_key_gold"].notna())
        & (
            candidatos["fecha_realizada_gold"]
            > candidatos["fecha_realizada_referencia"]
        )
    ].copy()

    siguiente = (
        posteriores.groupby("sesion_key_referencia")[
            "fecha_realizada_gold"
        ]
        .min()
        .rename("siguiente_sesion_fecha")
    )

    obt = obt.join(
        siguiente,
        on="sesion_key"
    )

    # --------------------------------------------------------
    # 4. Calcular días hasta la siguiente sesión
    # --------------------------------------------------------

    obt["dias_hasta_siguiente_sesion"] = (
        obt["siguiente_sesion_fecha"]
        - obt["fecha_realizada"]
    ).dt.days

    # --------------------------------------------------------
    # 5. Construir variable objetivo
    # --------------------------------------------------------

    obt[TARGET] = (
        obt["dias_hasta_siguiente_sesion"].isna()
        | (
            obt["dias_hasta_siguiente_sesion"]
            > HORIZONTE_DIAS
        )
    ).astype(int)

    # --------------------------------------------------------
    # 6. Eliminar información futura
    # --------------------------------------------------------

    obt = obt.drop(
        columns=[
            "siguiente_sesion_fecha",
            "dias_hasta_siguiente_sesion",
        ]
    )

    # --------------------------------------------------------
    # 7. Orden reproducible
    # --------------------------------------------------------

    obt = obt.sort_values(
        ["fecha_realizada", "sesion_key"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 8. Validaciones finales
    # --------------------------------------------------------

    if obt.empty:
        raise ValueError("No quedaron registros para modelado.")

    assert obt["sesion_key"].is_unique, (
        "Existen sesiones duplicadas."
    )

    assert obt[TARGET].notna().all(), (
        "El target contiene valores nulos."
    )

    assert set(obt[TARGET].unique()).issubset({0, 1}), (
        "El target contiene valores diferentes de 0 y 1."
    )

    print("6. Validaciones del dataset final...")
    print(f"   Sin horizonte completo : {sin_horizonte:,}")
    print(f"   Fechas inconsistentes  : {n_fechas_invalidas:,}")
    print(f"   Observaciones finales  : {len(obt):,}")
    print(f"   Continuidad (0)        : {(obt[TARGET] == 0).sum():,}")
    print(f"   Abandono (1)           : {(obt[TARGET] == 1).sum():,}")
    print(f"   Tasa de abandono       : {obt[TARGET].mean():.2%}")

    return obt


# ============================================================
# GUARDADO Y TRAZABILIDAD
# ============================================================

def guardar_dataset(df):
    """Guarda el dataset y calcula su huella SHA-256."""

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    sha256 = hashlib.sha256(
        OUTPUT_CSV.read_bytes()
    ).hexdigest()

    print("-" * 70)
    print("[OK] DATASET GENERADO CORRECTAMENTE")
    print(f"Archivo         : {OUTPUT_CSV}")
    print(f"Filas           : {len(df):,}")
    print(f"Columnas        : {len(df.columns)}")
    print(f"SHA-256         : {sha256}")
    print(f"Target          : {TARGET}")
    print(f"Horizonte       : {HORIZONTE_DIAS} días")
    print("-" * 70)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    df_obt, df_gold = extraer_datos()

    validar_fuentes(
        df_obt,
        df_gold
    )

    df_modelado = construir_dataset(
        df_obt,
        df_gold
    )

    guardar_dataset(df_modelado)


if __name__ == "__main__":
    main()