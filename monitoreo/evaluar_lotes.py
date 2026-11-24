"""Evalúa lotes completos únicamente cuando madura su target en el reloj virtual.
Ejecutar secuencialmente, sin otros evaluadores concurrentes.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from src.datos import conectar_snowflake
from monitoreo.simular_produccion import (
    SIMULACION_ID, SNAPSHOT_ID, SNAP_OBT, SNAP_GOLD, PRED, LOTES,
    SCHEMA, consultar_filas, ejecutar_sql, HORIZONTE,
)

MONITOREO = f"{SCHEMA}.MONITOREO"
COSTO_FN = 120.0
COSTO_FP = 3.0


def costo(y, decisiones):
    tn, fp, fn, tp = confusion_matrix(y, decisiones, labels=[0, 1]).ravel()
    return float(fn * COSTO_FN + fp * COSTO_FP)


def calcular_metricas(filas):
    # Columnas: UUID, sesión, probabilidad, decisión, latencia,
    # target observado, sesiones pendientes originales.
    y = np.array([int(f[5]) for f in filas], dtype=int)
    p = np.array([float(f[2]) for f in filas])
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades inválidas.")
    if any(f[3] not in ('contactar', 'no_contactar') for f in filas):
        raise ValueError("Decisión desconocida.")
    if not set(y).issubset({0, 1}):
        raise ValueError("Target inválido.")
    d = np.array([int(f[3] == 'contactar') for f in filas])
    lat = np.array([float(f[4]) for f in filas])
    if not np.isfinite(lat).all() or (lat < 0).any():
        raise ValueError("Latencias inválidas.")
    regla = np.array([int(f[6] == 1) for f in filas])
    tn, fp, fn, tp = map(int, confusion_matrix(y, d, labels=[0, 1]).ravel())
    return dict(
        TOTAL_OBSERVACIONES=len(y), ABANDONOS=int(y.sum()),
        TASA_ABANDONO=float(y.mean()),
        PR_AUC=float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
        PRECISION=tp / (tp + fp) if tp + fp else 0.0,
        RECALL=tp / (tp + fn) if tp + fn else None,
        TP=tp, FP=fp, FN=fn, TN=tn,
        TASA_CONTACTAR=float(d.mean()), PROBABILIDAD_PROMEDIO=float(p.mean()),
        LATENCIA_INFERENCIA_P50_MS=float(np.percentile(lat, 50)),
        LATENCIA_INFERENCIA_P95_MS=float(np.percentile(lat, 95)),
        COSTO_MODELO_USD=costo(y, d),
        COSTO_REGLA_1_SESION_USD=costo(y, regla),
        COSTO_LLAMAR_TODOS_USD=costo(y, np.ones_like(y)),
        COSTO_NO_HACER_NADA_USD=costo(y, np.zeros_like(y)),
    )


def evaluar_lotes(fecha_virtual: date):
    conn = conectar_snowflake()
    conn.autocommit(False)
    try:
        lotes = consultar_filas(conn, f"""
            SELECT LOTE_ID, FECHA_MADURACION, TOTAL_OBSERVACIONES,
                   TOTAL_PREDICCIONES, VERSION_MODELO, RUN_ID,
                   SNAPSHOT_ID, HASH_DATASET, DERIVA_INDUCIDA
            FROM {LOTES}
            WHERE SIMULACION_ID=%s AND ESTADO='esperando_target'
            ORDER BY NUMERO_LOTE
        """, (SIMULACION_ID,))
        evaluados = 0
        for lote, madura, esperado, total, version, run, snapshot, hash_datos, deriva in lotes:
            if fecha_virtual < madura:
                print(f"[PENDIENTE] {lote}: madura el {madura}")
                continue
            if total != esperado or esperado <= 0 or snapshot != SNAPSHOT_ID:
                raise RuntimeError(f"Lote {lote} incompleto o snapshot inesperado.")
            # El cierre del lote puede madurar después del último día de cobertura.
            # Lo que debe estar cubierto es el horizonte de cada sesión individual.
            cobertura = consultar_filas(conn, f"""
                SELECT LEAST((SELECT MAX(FECHA_REALIZADA) FROM {SNAP_OBT}),
                    (SELECT MAX(FECHA_REALIZADA) FROM {SNAP_GOLD}
                     WHERE INDICADOR_COMPLETADA=1))
            """)[0][0]
            if cobertura is None:
                raise RuntimeError("Snapshots sin cobertura temporal.")
            control = consultar_filas(conn, f"""
                SELECT COUNT(*), COUNT(DISTINCT ID_PREDICCION),
                       COUNT(DISTINCT SESION_KEY),
                       COALESCE(COUNT_IF(FECHA_MADURACION IS NULL OR FECHA_REFERENCIA IS NULL
                         OR FECHA_MADURACION <> DATEADD(day, {HORIZONTE}, FECHA_REFERENCIA)
                         OR FECHA_MADURACION > %s OR FECHA_MADURACION > %s
                         OR COALESCE(VERSION_MODELO,'') <> %s OR COALESCE(RUN_ID,'') <> %s
                         OR COALESCE(SNAPSHOT_ID,'') <> %s OR COALESCE(HASH_DATASET,'') <> %s
                         OR COALESCE(DERIVA_INDUCIDA, NOT %s) <> %s
                         OR USO_RESPALDO OR COALESCE(ORIGEN,'') <> 'simulacion'),0)
                FROM {PRED} WHERE SIMULACION_ID=%s AND LOTE_ID=%s
            """, (fecha_virtual, cobertura, version, run, snapshot, hash_datos,
                  deriva, deriva, SIMULACION_ID, lote))[0]
            if tuple(control) != (esperado, esperado, esperado, 0):
                raise RuntimeError(f"Predicciones inconsistentes de {lote}: {control}")
            # Misma definición que src/datos.py: continuidad en fecha estrictamente
            # posterior, mismo tratamiento, dentro de 60 días inclusive.
            filas = consultar_filas(conn, f"""
                SELECT p.ID_PREDICCION, p.SESION_KEY, p.PROBABILIDAD,
                       p.DECISION, p.LATENCIA_MS,
                       IFF(COUNT(g.SESION_KEY)>0,0,1) AS ABANDONO_60D,
                       o.SESIONES_PLANIFICADAS-o.NUMERO_SESION AS PENDIENTES
                FROM {PRED} p
                JOIN {SNAP_OBT} o ON p.SESION_KEY=o.SESION_KEY
                   AND p.PLAN_TRATAMIENTO_ID=o.PLAN_TRATAMIENTO_ID
                   AND p.PACIENTE_KEY=o.PACIENTE_KEY
                   AND p.FECHA_REFERENCIA=o.FECHA_REALIZADA
                LEFT JOIN {SNAP_GOLD} g
                   ON g.PLAN_TRATAMIENTO_ID=o.PLAN_TRATAMIENTO_ID
                   AND g.INDICADOR_COMPLETADA=1
                   AND g.FECHA_REALIZADA>o.FECHA_REALIZADA
                   AND g.FECHA_REALIZADA<=p.FECHA_MADURACION
                   AND g.FECHA_REALIZADA<=%s
                WHERE p.SIMULACION_ID=%s AND p.LOTE_ID=%s
                GROUP BY p.ID_PREDICCION,p.SESION_KEY,p.PROBABILIDAD,
                         p.DECISION,p.LATENCIA_MS,o.SESIONES_PLANIFICADAS,o.NUMERO_SESION
            """, (fecha_virtual, SIMULACION_ID, lote))
            if len(filas) != esperado or len({f[1] for f in filas}) != esperado:
                raise RuntimeError(f"No coinciden las sesiones con el snapshot de {lote}.")
            metricas = calcular_metricas(filas)
            registro = dict(SIMULACION_ID=SIMULACION_ID, LOTE_ID=lote,
                            FECHA_EVALUACION_SIMULADA=fecha_virtual,
                            VERSION_MODELO=version, RUN_ID=run, SNAPSHOT_ID=snapshot,
                            HASH_DATASET=hash_datos, DERIVA_INDUCIDA=deriva, **metricas)
            cols = list(registro)
            # No sobrescribir evidencias previas. La transacción une métricas y estado.
            if consultar_filas(conn, f"SELECT COUNT(*) FROM {MONITOREO} WHERE SIMULACION_ID=%s AND LOTE_ID=%s", (SIMULACION_ID,lote))[0][0]:
                raise RuntimeError(f"Ya existen métricas para {lote}; reconciliar estado.")
            ejecutar_sql(conn, f"INSERT INTO {MONITOREO} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))})", tuple(registro.values()))
            n = ejecutar_sql(conn, f"""
                UPDATE {LOTES} SET ESTADO='evaluado', FECHA_EVALUACION_SIMULADA=%s
                WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND ESTADO='esperando_target'
            """, (fecha_virtual, SIMULACION_ID, lote))
            if n != 1:
                raise RuntimeError(f"No se pudo actualizar el estado de {lote}.")
            conn.commit()
            evaluados += 1
            print(f"[OK] {lote} evaluado | fecha virtual={fecha_virtual}")
            for k,v in metricas.items():
                print(f"  {k}: {v}")
        print(f"Lotes evaluados en esta ejecución: {evaluados}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fecha-virtual', required=True, type=date.fromisoformat)
    evaluar_lotes(parser.parse_args().fecha_virtual)
