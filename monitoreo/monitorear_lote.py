"""Monitoreo histórico de un lote, sin volver a inferir. Ejecutar secuencialmente."""
from __future__ import annotations
import argparse
import json
from datetime import date
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from src.datos import conectar_snowflake
from src.pipeline import (cargar_datos, construir_features, obtener_hash_datos,
                            FEATURES_NUM, FEATURES_CAT, FECHA_EVENTO)
from monitoreo.simular_produccion import (SIMULACION_ID, SCHEMA, PRED, LOTES,
                                        TRACKING_URI, HTTP_LOG, consultar_filas, ejecutar_sql)
from monitoreo.deriva import evaluar_deriva_lote, diagnosticar_deriva
from monitoreo.alertas import evaluar_alertas_lote

INMEDIATO = f'{SCHEMA}.MONITOREO_INMEDIATO'
DETALLE = f'{SCHEMA}.DERIVA_FEATURES'
ALERTAS = f'{SCHEMA}.ALERTAS'


def escalar(v):
    if pd.isna(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def insertar(conn, tabla, datos):
    cols = ','.join(datos)
    placeholders = ','.join(['%s'] * len(datos))
    ejecutar_sql(conn, f'INSERT INTO {tabla} ({cols}) VALUES ({placeholders})',
                 tuple(escalar(v) for v in datos.values()))


def monitorear(numero, fecha_virtual):
    lote = f'L{numero:02d}'
    clave = dict(SIMULACION_ID=SIMULACION_ID, LOTE_ID=lote)
    params = tuple(clave.values())
    conn = conectar_snowflake()
    conn.autocommit(False)
    try:
        filas = consultar_filas(conn, f'''SELECT FECHA_PROCESAMIENTO_SIMULADA, FECHA_MADURACION,
            ESTADO, TOTAL_OBSERVACIONES, VERSION_MODELO, RUN_ID, HASH_DATASET, SNAPSHOT_ID,
            DERIVA_INDUCIDA FROM {LOTES} WHERE SIMULACION_ID=%s AND LOTE_ID=%s''', params)
        if len(filas) != 1:
            raise RuntimeError('Lote inexistente o duplicado.')
        cierre, madura, estado, n, version, run, hash_datos, snapshot, inducida = filas[0]
        if estado not in ('esperando_target', 'evaluado') or fecha_virtual < cierre:
            raise RuntimeError('Lote incompleto o todavía no disponible en el reloj virtual.')
        cliente = MlflowClient(tracking_uri=TRACKING_URI)
        corrida = cliente.get_run(run).data
        if corrida.params['hash_dataset'] != hash_datos or obtener_hash_datos() != hash_datos:
            raise RuntimeError('La referencia local no coincide con el dataset histórico del modelo.')
        # Referencia original PSet03; para un modelo reentrenado deberá reconstruirse su ventana exacta.
        if str(version) != '1':
            raise RuntimeError('Referencia de entrenamiento de esta versión aún no implementada.')
        corte = pd.Timestamp(corrida.params['fecha_corte'])
        horizonte = int(corrida.params['horizonte_dias'])
        base = construir_features(cargar_datos())
        ref = base.loc[pd.to_datetime(base[FECHA_EVENTO]) <= corte-pd.Timedelta(days=horizonte)]
        if len(ref) != 1195:
            raise RuntimeError('Referencia de entrenamiento inesperada.')
        pred = consultar_filas(conn, f'''SELECT SESION_KEY, ENTRADAS_JSON, PROBABILIDAD, DECISION,
            LATENCIA_MS, VERSION_MODELO, RUN_ID, USO_RESPALDO FROM {PRED}
            WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND ORIGEN='simulacion' ''', params)
        if len(pred) != n or len({r[0] for r in pred}) != n or any(r[0] is None or str(r[5]) != str(version) or r[6] != run or r[7] for r in pred):
            raise RuntimeError('Predicciones incompletas o inconsistentes.')
        entradas = construir_features(pd.DataFrame([json.loads(r[1]) for r in pred]))
        detalle = evaluar_deriva_lote(ref, entradas, FEATURES_NUM, FEATURES_CAT)
        peor = detalle.iloc[0]
        scores = np.array([float(r[2]) for r in pred]); lat = np.array([float(r[4]) for r in pred])
        if not np.isfinite(scores).all() or not np.isfinite(lat).all() or ((scores<0)|(scores>1)).any() or (lat<0).any():
            raise RuntimeError('Scores o latencias inválidos.')
        if any(r[3] not in ('contactar','no_contactar') for r in pred):
            raise RuntimeError('Decisión inválida.')
        maximo = lambda c: escalar(detalle[c].max())
        resumen = dict(**clave, FECHA_SIMULADA=cierre, VERSION_MODELO=str(version), RUN_ID=run,
            HASH_DATASET=hash_datos, SNAPSHOT_ID=snapshot, DERIVA_INDUCIDA=inducida, TOTAL_PREDICCIONES=n,
            PSI_MAX=escalar(peor.psi), FEATURE_PSI_MAX=peor.feature, MUESTRA_PEQUENA=n<30,
            PCT_FALTANTES_MAX=maximo('pct_faltantes'), PCT_FUERA_RANGO_MAX=maximo('pct_fuera_rango'),
            PCT_CATEGORIAS_NO_VISTAS_MAX=maximo('pct_categorias_no_vistas'),
            PROBABILIDAD_PROMEDIO=scores.mean(), PROBABILIDAD_P05=np.percentile(scores,5),
            PROBABILIDAD_P50=np.percentile(scores,50), PROBABILIDAD_P95=np.percentile(scores,95),
            TASA_CONTACTAR=sum(r[3]=='contactar' for r in pred)/n,
            LATENCIA_INFERENCIA_P50_MS=np.percentile(lat,50), LATENCIA_INFERENCIA_P95_MS=np.percentile(lat,95))
        # Telemetría exclusiva de POST /predecir, no contadores globales de API.
        intentos = consultar_filas(conn, f"""SELECT SESION_KEY, CODIGO_HTTP,
            LATENCIA_HTTP_MS, ERROR_TRANSPORTE, ID_PREDICCION
            FROM {HTTP_LOG} WHERE SIMULACION_ID=%s AND LOTE_ID=%s""", params)
        if intentos:
            exitos = [r for r in intentos if r[1] == 200 and not r[3]]
            ids_pred = consultar_filas(conn, f"""SELECT SESION_KEY, ID_PREDICCION
                FROM {PRED} WHERE SIMULACION_ID=%s AND LOTE_ID=%s
                AND ORIGEN='simulacion'""", params)
            if len(exitos) != n or {(r[0],r[4]) for r in exitos} != set(ids_pred):
                raise RuntimeError('Telemetría incompleta o no conciliada con las predicciones.')
            tiempos = np.array([float(r[2]) for r in intentos])
            if not np.isfinite(tiempos).all() or (tiempos < 0).any():
                raise RuntimeError('Latencia HTTP inválida.')
            volumen = len(intentos)
            e4 = sum(r[1] is not None and 400 <= r[1] < 500 for r in intentos)
            e5 = sum(r[1] is not None and 500 <= r[1] < 600 for r in intentos)
            et = sum(bool(r[3]) for r in intentos)
            resumen.update(VOLUMEN_HTTP=volumen, ERRORES_4XX=e4, ERRORES_5XX=e5,
                ERRORES_TRANSPORTE=et, TASA_ERROR_4XX=e4/volumen,
                TASA_ERROR_5XX=e5/volumen,
                TASA_ERROR_HTTP=sum(r[1] != 200 or r[3] for r in intentos)/volumen,
                LATENCIA_HTTP_P50_MS=np.percentile(tiempos,50),
                LATENCIA_HTTP_P95_MS=np.percentile(tiempos,95))
        eventos = evaluar_alertas_lote(lote, resumen['PSI_MAX'], peor.feature, n_observaciones=n,
            pct_faltantes_max=resumen['PCT_FALTANTES_MAX'], pct_fuera_rango_max=resumen['PCT_FUERA_RANGO_MAX'],
            pct_categorias_no_vistas_max=resumen['PCT_CATEGORIAS_NO_VISTAS_MAX'],
            tasa_error_http=resumen.get('TASA_ERROR_HTTP'),
            latencia_p95_ms=resumen.get('LATENCIA_HTTP_P95_MS'))
        # Reemplazo transaccional por clave; no ejecutar en paralelo.
        for tabla in (INMEDIATO, DETALLE):
            ejecutar_sql(conn, f'DELETE FROM {tabla} WHERE SIMULACION_ID=%s AND LOTE_ID=%s', params)
        insertar(conn, INMEDIATO, resumen)
        for registro in detalle.to_dict('records'):
            insertar(conn, DETALLE, dict(**clave, **{k.upper():v for k,v in registro.items()}))
        def guardar(fase, fecha, eventos):
            ejecutar_sql(conn, f'DELETE FROM {ALERTAS} WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND FASE=%s', (*params,fase))
            for e in eventos:
                insertar(conn, ALERTAS, dict(**clave, FASE=fase, FECHA_SIMULADA=fecha,
                    **{k.upper():v for k,v in e.items() if k not in ('timestamp','lote_id')}))
        guardar('inmediato', cierre, eventos)
        # No consultar métricas futuras aunque ya estén físicamente almacenadas.
        pr = None
        if fecha_virtual >= madura:
            sup = consultar_filas(conn, f'''SELECT PR_AUC, FECHA_EVALUACION_SIMULADA FROM {SCHEMA}.MONITOREO
                WHERE SIMULACION_ID=%s AND LOTE_ID=%s AND FECHA_EVALUACION_SIMULADA<=%s''', (*params,fecha_virtual))
            if len(sup)>1: raise RuntimeError('Evaluaciones duplicadas.')
            if sup:
                pr, fecha_eval = sup[0]
                if fecha_eval < madura: raise RuntimeError('Evaluación prematura.')
                ev = evaluar_alertas_lote(lote, None, None, pr_auc_lote=pr,
                    media_cv=corrida.metrics['pr_auc_cv_mean'], std_cv=corrida.metrics['pr_auc_cv_std'], n_observaciones=n)
                guardar('supervisado', fecha_eval, ev)
                eventos += ev
        diagnostico = diagnosticar_deriva(resumen['PSI_MAX'], pr,
            media_cv=corrida.metrics['pr_auc_cv_mean'], std_cv=corrida.metrics['pr_auc_cv_std'])
        conn.commit()
        print(f'[OK] {lote}: {n} predicciones | 13 features | PSI máximo={peor.psi:.6f} ({peor.feature})')
        print('Diagnóstico:', diagnostico['diagnostico'])
        if intentos:
            print(f"HTTP: {volumen} solicitudes | 4xx={e4} | 5xx={e5} | transporte={et} | p95={resumen['LATENCIA_HTTP_P95_MS']:.3f} ms")
        else:
            print('Telemetría HTTP histórica: no disponible; no se sustituye por latencia de inferencia.')
        for e in eventos: print(e['severidad'], e['metrica'], e['valor_observado'])
        print(f'[OK] {len(eventos)} alertas visibles en esta evaluación.')
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lote', type=int, required=True, choices=range(1,15))
    p.add_argument('--fecha-virtual', type=date.fromisoformat, required=True)
    a=p.parse_args(); monitorear(a.lote,a.fecha_virtual)
