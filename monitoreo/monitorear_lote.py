"""Monitoreo histórico de un lote, sin volver a inferir. Ejecutar secuencialmente."""
from __future__ import annotations
import argparse
import json
import os
from datetime import date
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from src.datos import conectar_snowflake
from src.pipeline import (cargar_datos, construir_features, obtener_hash_datos,
                            FEATURES_NUM, FEATURES_CAT, FECHA_EVENTO)
from src.reentrenar import dataset as dataset_snapshot, huella, SNAP
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


def resolver_referencia(cliente, version, run, hash_datos, snapshot=None):
    """Reconstruye la referencia del run exacto; no entrena ni escribe datos."""
    nombre = os.getenv('MODEL_NAME', 'oralsolutions-modelo')
    mv = cliente.get_model_version(nombre, str(version))
    if mv.run_id != run:
        raise RuntimeError('Version y run_id no coinciden en MLflow.')
    registro = cliente.get_run(run)
    corrida = registro.data
    p = corrida.params
    if registro.info.status != 'FINISHED' or p.get('hash_dataset') != hash_datos:
        raise RuntimeError('Corrida no finalizada o huella del lote incompatible.')
    media, std = (float(corrida.metrics[k]) for k in ('pr_auc_cv_mean', 'pr_auc_cv_std'))
    if not np.isfinite([media, std]).all() or not 0 <= media <= 1 or not 0 <= std <= 1:
        raise RuntimeError('Referencia CV invalida.')
    horizonte = int(p['horizonte_dias'])
    if horizonte != 60:
        raise RuntimeError('Horizonte incompatible con la simulacion.')
    corte = pd.Timestamp(p['fecha_corte'])
    if corrida.tags.get('fase') == 'reentrenamiento':
        if p.get('snapshot_id') != SNAP or (snapshot is not None and snapshot != SNAP):
            raise RuntimeError('Snapshot de entrenamiento no soportado.')
        base = dataset_snapshot()
        limite = pd.Timestamp(p['limite_train'])
        if limite + pd.Timedelta(days=horizonte) >= corte:
            raise RuntimeError('Separacion temporal de reentrenamiento invalida.')
        ref = base.loc[pd.to_datetime(base[FECHA_EVENTO]) <= limite].copy()
        if (len(ref) != int(p['n_filas_entrenamiento']) or huella(base) != hash_datos
                or huella(ref) != p['hash_train']):
            raise RuntimeError('La referencia reconstruida no coincide con las huellas historicas.')
    elif str(version) == '1' and corrida.tags.get('fase') == 'final':
        if obtener_hash_datos() != hash_datos:
            raise RuntimeError('La referencia local no coincide con el dataset historico.')
        base = construir_features(cargar_datos())
        ref = base.loc[pd.to_datetime(base[FECHA_EVENTO]) <= corte-pd.Timedelta(days=horizonte)].copy()
        if len(ref) != 1195:
            raise RuntimeError('Referencia de entrenamiento original inesperada.')
    else:
        raise RuntimeError('Procedencia de entrenamiento no soportada; no usar una referencia ajena.')
    if ref.empty or not set(FEATURES_NUM + FEATURES_CAT).issubset(ref.columns):
        raise RuntimeError('Referencia vacia o incompleta.')
    return ref, corrida


def verificar_referencia(version):
    cliente = MlflowClient(tracking_uri=TRACKING_URI)
    mv = cliente.get_model_version(os.getenv('MODEL_NAME', 'oralsolutions-modelo'), str(version))
    corrida = cliente.get_run(mv.run_id).data
    ref, corrida = resolver_referencia(cliente, version, mv.run_id, corrida.params['hash_dataset'])
    print(f'[OK] Referencia v{version}: {len(ref)} observaciones | {len(FEATURES_NUM + FEATURES_CAT)} features')
    print(f'Run ID: {mv.run_id}')
    print(f"PR-AUC CV: {corrida.metrics['pr_auc_cv_mean']:.6f} +/- {corrida.metrics['pr_auc_cv_std']:.6f}")
    print('[OK] Solo lectura: sin predicciones, entrenamiento ni escritura de metricas.')


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
        ref, corrida = resolver_referencia(cliente, version, run, hash_datos, snapshot)
        if corrida.tags.get('fase') == 'reentrenamiento' and pd.Timestamp(cierre) < pd.Timestamp(corrida.params['fecha_virtual']):
            raise RuntimeError('El lote es anterior a la disponibilidad virtual del modelo reentrenado.')
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
    p.add_argument('--lote', type=int, choices=range(1,15))
    p.add_argument('--fecha-virtual', type=date.fromisoformat)
    p.add_argument('--verificar-version', type=int)
    a=p.parse_args()
    if a.verificar_version is not None:
        if a.lote is not None or a.fecha_virtual is not None:
            p.error('--verificar-version no admite lote ni fecha virtual')
        verificar_referencia(a.verificar_version)
    else:
        if a.lote is None or a.fecha_virtual is None:
            p.error('indique --lote y --fecha-virtual')
        monitorear(a.lote,a.fecha_virtual)
