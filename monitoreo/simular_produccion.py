"""OralSolutions: primera etapa de simulación cronológica en Snowflake.

Ejecutar desde la raíz: python -m monitoreo.simular_produccion --lote 1
No calcula ni consulta etiquetas durante la inferencia. El piloto queda
esperando_target. No ejecutar en paralelo ni reintentar lotes incompletos
sin reconciliar las predicciones ya escritas por la API.
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import date, timedelta

import pandas as pd
import requests
from mlflow.tracking import MlflowClient

from src.datos import conectar_snowflake, consultar_dataframe, TABLA_OBT, TABLA_GOLD
from src.pipeline import FECHA_CORTE, FECHA_EVENTO

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
SIMULACION_ID = "SIM_2025_V1"
SNAPSHOT_ID = "SNAP_SIM_2025_V1"
SCHEMA = "ORALSOLUTIONS_DW.MLOPS"
PRED = f"{SCHEMA}.PREDICCIONES"
LOTES = f"{SCHEMA}.LOTES_PRODUCCION"
HTTP_LOG = f"{SCHEMA}.SOLICITUDES_HTTP"
SNAP_OBT = f"{SCHEMA}.{SNAPSHOT_ID}_OBT"
SNAP_GOLD = f"{SCHEMA}.{SNAPSHOT_ID}_GOLD"
HORIZONTE = 60
OBSERVACIONES_ESPERADAS = {
    1: 12,
    2: 37,
    3: 53,
    4: 61,
    5: 54,
    6: 68,
    7: 59,
    8: 63,
    9: 67,
    10: 66,
    11: 66,
    12: 76,
    13: 63,
    14: 81,
}
LOTES_DERIVA = {7, 8}
ESCENARIO_DERIVA = "incremento_sintetico_cancelaciones_e_inasistencias_1"

COLS_NUM = ["duracion_estimada_min", "numero_sesion", "sesiones_planificadas",
            "cancelaciones_previas", "inasistencias_previas", "dias_desde_inicio_plan",
            "edad_en_atencion"]
COLS_CAT = ["codigo_procedimiento", "odontologo_key"]


def ejecutar_sql(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def consultar_filas(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def preparar_snapshot(conn):
    """
    Crea snapshots persistentes de las fuentes de simulación.

    Admite tablas de origen TRANSIENT y permite recuperar
    la creación parcial de los snapshots iniciales.
    """

    existentes = consultar_filas(conn, """
        SELECT TABLE_NAME
        FROM ORALSOLUTIONS_DW.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'MLOPS'
          AND TABLE_NAME IN (%s, %s)
    """, (
        SNAP_OBT.split(".")[-1],
        SNAP_GOLD.split(".")[-1],
    ))

    nombres = {fila[0] for fila in existentes}

    nombre_obt = SNAP_OBT.split(".")[-1]
    nombre_gold = SNAP_GOLD.split(".")[-1]

    if nombre_obt not in nombres:

        ejecutar_sql(
            conn,
            f"CREATE TRANSIENT TABLE {SNAP_OBT} CLONE {TABLA_OBT}"
        )

        print(f"[OK] Snapshot OBT creado: {SNAP_OBT}")

    else:
        print(f"[OK] Snapshot OBT existente: {SNAP_OBT}")

    if nombre_gold not in nombres:

        ejecutar_sql(
            conn,
            f"CREATE TRANSIENT TABLE {SNAP_GOLD} CLONE {TABLA_GOLD}"
        )

        print(f"[OK] Snapshot Gold creado: {SNAP_GOLD}")

    else:
        print(f"[OK] Snapshot Gold existente: {SNAP_GOLD}")

    print(f"Snapshot: {SNAPSHOT_ID}")


def obtener_lote(conn, numero):
    corte = date.fromisoformat(FECHA_CORTE)
    primer_domingo = corte + timedelta(days=(6 - corte.weekday()) % 7)
    fin = primer_domingo + timedelta(weeks=numero - 1)
    inicio = corte if numero == 1 else fin - timedelta(days=6)
    # La fecha de cobertura es un parámetro histórico del experimento,
    # no una feature. No se leen resultados futuros ni etiquetas aquí.
    ultima_elegible = date(2025, 11, 1)
    df = consultar_dataframe(conn, f"""
        SELECT * FROM {SNAP_OBT}
        WHERE FECHA_REALIZADA BETWEEN '{inicio.isoformat()}' AND '{min(fin, ultima_elegible).isoformat()}'
          AND FECHA_REALIZADA >= FECHA_INICIO_PLAN
        ORDER BY FECHA_REALIZADA, SESION_KEY
    """)
    if df.empty or df['sesion_key'].isna().any() or not df['sesion_key'].is_unique:
        raise RuntimeError("Lote vacío o sesiones inválidas.")
    df[FECHA_EVENTO] = pd.to_datetime(df[FECHA_EVENTO])
    if df[COLS_NUM + COLS_CAT + ['paciente_key', 'plan_tratamiento_id']].isna().any().any():
        raise RuntimeError("El lote contiene entradas o identificadores nulos.")
    esperado = OBSERVACIONES_ESPERADAS[numero]
    if len(df) != esperado:
        raise RuntimeError(
            f"L{numero:02d}: se esperaban {esperado} "
            f"observaciones, llegaron {len(df)}."
        )

    return inicio, fin, df


def payload_api(fila, numero=1):
    payload = {c: int(fila[c]) for c in COLS_NUM}
    payload.update({c: str(fila[c]) for c in COLS_CAT})
    if numero in LOTES_DERIVA:
        for campo in ("cancelaciones_previas", "inasistencias_previas"):
            payload[campo] = min(50, payload[campo] + 1)
    return payload


def modelo_activo(http):
    respuesta = http.get(f"{API_URL}/modelo", timeout=120)
    respuesta.raise_for_status()
    info = respuesta.json()
    if info['modo_respaldo'] or info['alias'] != 'champion':
        raise RuntimeError("No está activo el champion ML.")
    cliente = MlflowClient(tracking_uri=TRACKING_URI)
    version = cliente.get_model_version_by_alias(info['nombre_modelo'], 'champion')
    if str(version.version) != str(info['version']) or version.run_id != info['run_id']:
        raise RuntimeError("La API y MLflow no coinciden.")
    info['hash_dataset'] = cliente.get_run(info['run_id']).data.params['hash_dataset']
    return info


def solicitar_prediccion(conn, http, lote_id, sesion, fecha, payload):
    """Registra cada intento POST, incluso errores HTTP y de transporte.

    No reintenta solicitudes: un timeout puede haber generado una predicción.
    La latencia se mide hasta recibir la respuesta, antes de escribir telemetría.
    """
    solicitud = str(uuid.uuid4())
    inicio = time.perf_counter()
    respuesta = None
    error_transporte = False
    try:
        respuesta = http.post(f"{API_URL}/predecir", json=payload, timeout=120)
    except requests.RequestException:
        error_transporte = True
        raise
    finally:
        latencia = (time.perf_counter() - inicio) * 1000
        codigo = respuesta.status_code if respuesta is not None else None
        prediccion = None
        if codigo == 200:
            try:
                prediccion = respuesta.json().get('id_prediccion')
            except (ValueError, AttributeError):
                pass
        ejecutar_sql(conn, f"""
            INSERT INTO {HTTP_LOG} (ID_SOLICITUD, SIMULACION_ID, LOTE_ID,
                SESION_KEY, FECHA_SIMULADA, CODIGO_HTTP, LATENCIA_HTTP_MS,
                ERROR_TRANSPORTE, ID_PREDICCION)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (solicitud, SIMULACION_ID, lote_id, sesion, fecha, codigo,
              latencia, error_transporte, prediccion))
    respuesta.raise_for_status()
    return respuesta.json()


def ejecutar_lote(numero=1):
    if numero not in OBSERVACIONES_ESPERADAS:
        raise ValueError(
            "El número de lote debe estar entre 1 y 14."
        )
    lote_id = f"L{numero:02d}"
    conn = conectar_snowflake()
    conn.autocommit(True)
    try:
        previos = consultar_filas(conn, f"""
            SELECT ESTADO, TOTAL_OBSERVACIONES FROM {LOTES}
            WHERE SIMULACION_ID=%s AND LOTE_ID=%s
        """, (SIMULACION_ID, lote_id))
        if previos:
            estado, esperado = previos[0]
            if len(previos) == 1 and estado in ('esperando_target', 'evaluado'):
                conteos = consultar_filas(conn, f"""
                    SELECT COUNT(*), COUNT(DISTINCT SESION_KEY) FROM {PRED}
                    WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND ORIGEN='simulacion'
                """, (SIMULACION_ID, lote_id))[0]
                if tuple(conteos) == (esperado, esperado):
                    print(f"[OK] {lote_id} ya procesado; no se repiten predicciones.")
                    return
            raise RuntimeError("Lote previo incompleto o inconsistente. Reconciliar antes de reintentar.")

        preparar_snapshot(conn)
        inicio, fin, df = obtener_lote(conn, numero)
        maduracion = fin + timedelta(days=HORIZONTE)
        deriva = numero in LOTES_DERIVA
        escenario = ESCENARIO_DERIVA if deriva else "sin_deriva"
        with requests.Session() as http:
            info = modelo_activo(http)
            print(f"{lote_id}: {inicio} a {fin} | {len(df)} observaciones")
            print(f"Champion v{info['version']} | run_id={info['run_id']}")
            print(f"Escenario: {escenario}")
            print(f"Fecha virtual: {fin} | Evaluación completa desde: {maduracion}")
            ejecutar_sql(conn, f"""
                INSERT INTO {LOTES} (SIMULACION_ID, LOTE_ID, NUMERO_LOTE,
                    FECHA_INICIO, FECHA_FIN, FECHA_PROCESAMIENTO_SIMULADA,
                    FECHA_MADURACION, TOTAL_OBSERVACIONES, ESTADO,
                    VERSION_MODELO, RUN_ID, HASH_DATASET, SNAPSHOT_ID,
                    DERIVA_INDUCIDA, ESCENARIO_DERIVA)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'procesando',%s,%s,%s,%s,%s,%s)
            """, (SIMULACION_ID, lote_id, numero, inicio, fin, fin, maduracion,
                  len(df), str(info['version']), info['run_id'], info['hash_dataset'], SNAPSHOT_ID, deriva, escenario))
            try:
                for i, (_, fila) in enumerate(df.iterrows(), 1):
                    payload = payload_api(fila, numero)
                    # No se reintenta POST automáticamente: la API asigna UUID.
                    r = solicitar_prediccion(conn, http, lote_id, int(fila['sesion_key']), fin, payload)
                    if (r['uso_respaldo'] or str(r['version_modelo']) != str(info['version'])
                            or r['run_id'] != info['run_id']):
                        raise RuntimeError(f"Modelo inesperado en predicción {r['id_prediccion']}.")
                    referencia = fila[FECHA_EVENTO].date()
                    # La API ya insertó la fila. Se enriquece por UUID, no se duplica.
                    n = ejecutar_sql(conn, f"""
                        UPDATE {PRED} SET ORIGEN='simulacion', SIMULACION_ID=%s,
                            LOTE_ID=%s, SESION_KEY=%s, PACIENTE_KEY=%s,
                            PLAN_TRATAMIENTO_ID=%s, FECHA_REFERENCIA=%s,
                            FECHA_MADURACION=%s, FECHA_SIMULADA=%s,
                            UMBRAL_APLICADO=%s, ACCION=%s, HASH_DATASET=%s,
                            SNAPSHOT_ID=%s, DERIVA_INDUCIDA=%s
                        WHERE ID_PREDICCION=%s AND ORIGEN IS NULL
                          AND VERSION_MODELO=%s AND RUN_ID=%s
                    """, (SIMULACION_ID, lote_id, int(fila['sesion_key']), int(fila['paciente_key']),
                          int(fila['plan_tratamiento_id']), referencia, referencia+timedelta(days=HORIZONTE),
                          fin, float(r['umbral_aplicado']), r['accion'], info['hash_dataset'],
                          SNAPSHOT_ID, deriva, r['id_prediccion'], str(info['version']), info['run_id']))
                    if n != 1:
                        raise RuntimeError(f"Se esperaba actualizar 1 fila para {r['id_prediccion']}; resultado={n}.")
                    print(f"  [{i:02d}/{len(df)}] sesión={int(fila['sesion_key'])} | id={r['id_prediccion']} | {r['decision']}")
                total, unicas = consultar_filas(conn, f"""
                    SELECT COUNT(*), COUNT(DISTINCT SESION_KEY) FROM {PRED}
                    WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND ORIGEN='simulacion'
                """, (SIMULACION_ID, lote_id))[0]
                if (total, unicas) != (len(df), len(df)):
                    raise RuntimeError("El conteo de predicciones no coincide con el lote.")
                ejecutar_sql(conn, f"""
                    UPDATE {LOTES} SET ESTADO='esperando_target', TOTAL_PREDICCIONES=%s,
                        TOTAL_ERRORES=0 WHERE SIMULACION_ID=%s AND LOTE_ID=%s
                """, (total, SIMULACION_ID, lote_id))
            except Exception:
                total = consultar_filas(conn, f"""
                    SELECT COUNT(*) FROM {PRED} WHERE SIMULACION_ID=%s AND LOTE_ID=%s
                """, (SIMULACION_ID, lote_id))[0][0]
                ejecutar_sql(conn, f"""
                    UPDATE {LOTES} SET ESTADO='error', TOTAL_PREDICCIONES=%s,
                        TOTAL_ERRORES=1 WHERE SIMULACION_ID=%s AND LOTE_ID=%s
                """, (total, SIMULACION_ID, lote_id))
                raise
        print(f"[OK] {lote_id}: {total} predicciones persistidas y asociadas.")
        print("Estado: esperando_target. No se han calculado métricas supervisadas.")
    finally:
        conn.close()


def verificar_lotes():
    """Verificación de solo lectura: no genera predicciones ni modifica tablas."""
    from api.esquemas import SolicitudPrediccion

    conn = conectar_snowflake()
    try:
        total = 0
        sesiones = set()
        for numero in OBSERVACIONES_ESPERADAS:
            inicio, fin, df = obtener_lote(conn, numero)
            claves = set(df["sesion_key"].astype(int))
            if sesiones.intersection(claves):
                raise RuntimeError("Sesiones repetidas entre lotes.")
            sesiones.update(claves)
            for _, fila in df.iterrows():
                SolicitudPrediccion.model_validate(payload_api(fila, numero))
            total += len(df)
            print(f"[OK] L{numero:02d}: {inicio} a {fin} | {len(df)} entradas validas | deriva={numero in LOTES_DERIVA}")
        if total != 826 or len(sesiones) != 826:
            raise RuntimeError("Conteo global inesperado.")
        print(f"[OK] {total} entradas validas y sesiones unicas. No se generaron predicciones.")
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lote', type=int, default=1)
    parser.add_argument('--verificar', action='store_true')
    args = parser.parse_args()
    if args.verificar:
        verificar_lotes()
    else:
        ejecutar_lote(args.lote)
