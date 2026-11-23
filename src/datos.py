import base64
import os
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "obt_riesgo_abandono.csv"


def conectar_snowflake():
    load_dotenv(ENV_PATH)
    password = base64.b64decode(
        os.environ["SECRET_SNOWFLAKE_PASSWORD"], validate=True
    ).decode("utf-8")

    return snowflake.connector.connect(
        account=os.environ["SF_ACCOUNT"],
        user=os.environ["SF_USER"],
        password=password,
        warehouse=os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SF_DATABASE", "ORALSOLUTIONS_DW"),
        schema=os.getenv("SF_OBT_SCHEMA", "OBT"),
        role=os.getenv("SF_ROLE", "ACCOUNTADMIN"),
    )


def main() -> None:
    print("1. Conectando a Snowflake y descargando OBT_RIESGO_ABANDONO...")
    conn = conectar_snowflake()
    try:
        query = "SELECT * FROM ORALSOLUTIONS_DW.OBT.OBT_RIESGO_ABANDONO"
        cur = conn.cursor()
        cur.execute(query)
        columnas = [col[0].lower() for col in cur.description]
        filas = cur.fetchall()
        df = pd.DataFrame(filas, columns=columnas)
    finally:
        conn.close()

    print(f"   Filas descargadas desde Snowflake: {len(df):,} | Columnas: {len(df.columns)}")

    col_fecha = "fecha_realizada"
    col_plan = "plan_tratamiento_id"

    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df["fecha_inicio_plan"] = pd.to_datetime(df["fecha_inicio_plan"])

    fecha_corte_max = df[col_fecha].max()
    limite_observacion = fecha_corte_max - pd.Timedelta(days=60)

    # Ordenar por plan y fecha de realización para medir días hasta la siguiente sesión completada
    df = df.sort_values([col_plan, col_fecha, "numero_sesion"]).reset_index(drop=True)
    df["siguiente_sesion_fecha"] = df.groupby(col_plan)[col_fecha].shift(-1)
    df["dias_hasta_siguiente_sesion"] = (
        df["siguiente_sesion_fecha"] - df[col_fecha]
    ).dt.days

    # Excluir sesiones de los últimos 60 días (ventana de observación incompleta / censura a la derecha)
    df_modelado = df[df[col_fecha] <= limite_observacion].copy()

    # Target: 1 si no volvió en <= 60 días (o no tiene siguiente sesión), 0 si continuó dentro de 60 días
    df_modelado["abandono_60d"] = (
        df_modelado["dias_hasta_siguiente_sesion"].isna()
        | (df_modelado["dias_hasta_siguiente_sesion"] > 60)
    ).astype(int)

    # Eliminar columnas auxiliares futuras para evitar cualquier fuga de datos
    df_modelado = df_modelado.drop(
        columns=["siguiente_sesion_fecha", "dias_hasta_siguiente_sesion"]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_modelado.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    tasa_abandono = df_modelado["abandono_60d"].mean() * 100
    print("-" * 70)
    print(f"[OK] Dataset guardado en: {OUTPUT_CSV}")
    print(f"     Fecha mínima en OBT : {df[col_fecha].min().date()}")
    print(f"     Fecha máxima en OBT : {fecha_corte_max.date()}")
    print(f"     Corte sin censura   : <= {limite_observacion.date()}")
    print(f"     Filas finales       : {len(df_modelado):,} | Columnas: {len(df_modelado.columns)}")
    print(f"     Distribución target : {df_modelado['abandono_60d'].value_counts().to_dict()}")
    print(f"     Tasa abandono_60d   : {tasa_abandono:.2f}% (clase 1)")


if __name__ == "__main__":
    main()