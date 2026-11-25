"""Reentrenamiento por calendario, evaluación temporal y aprobación separada."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from src.datos import conectar_snowflake, consultar_dataframe, validar_fuentes, construir_dataset
from src.pipeline import construir_features, construir_pipeline, FEATURES_FINALES, TARGET, FECHA_EVENTO

ROOT = Path(__file__).resolve().parent.parent
URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
NAME = os.getenv('MODEL_NAME', 'oralsolutions-modelo')
SCHEMA = 'ORALSOLUTIONS_DW.MLOPS'
SNAP = 'SNAP_SIM_2025_V1'
FECHA_EVAL_INICIO = pd.Timestamp('2025-10-27')
FECHA_EVAL_FIN = pd.Timestamp('2025-11-01')
LIMITE_TRAIN = FECHA_EVAL_INICIO - pd.Timedelta(days=61)
FECHA_CALENDARIO = pd.Timestamp('2026-01-01')


def evaluar(modelo, df, umbral):
    y = df[TARGET].astype(int).to_numpy()
    p = modelo.predict_proba(df[FEATURES_FINALES])[:, list(modelo.classes_).index(1)]
    d = (p >= umbral).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, d, labels=[0, 1]).ravel())
    return dict(pr_auc=float(average_precision_score(y,p)), precision=float(precision_score(y,d,zero_division=0)),
                recall=float(recall_score(y,d,zero_division=0)), costo_errores_usd=float(120*fn+3*fp),
                tp=tp, fp=fp, fn=fn, tn=tn, tasa_contactar=float(d.mean()))


def dataset():
    conn = conectar_snowflake()
    try:
        obt = consultar_dataframe(conn, f'SELECT * FROM {SCHEMA}.{SNAP}_OBT')
        gold = consultar_dataframe(conn, f'''SELECT SESION_KEY, PLAN_TRATAMIENTO_ID, NUMERO_SESION, FECHA_REALIZADA
          FROM {SCHEMA}.{SNAP}_GOLD WHERE INDICADOR_COMPLETADA=1 AND FECHA_REALIZADA IS NOT NULL''')
    finally:
        conn.close()
    validar_fuentes(obt,gold)
    df = construir_features(construir_dataset(obt,gold))
    df[FECHA_EVENTO] = pd.to_datetime(df[FECHA_EVENTO])
    return df.sort_values([FECHA_EVENTO,'sesion_key']).reset_index(drop=True)


def huella(df):
    return hashlib.sha256(df[['sesion_key',FECHA_EVENTO,*FEATURES_FINALES,TARGET]].to_csv(index=False, lineterminator="\n").encode('utf-8')).hexdigest()


def cliente():
    mlflow.set_tracking_uri(URI)
    return MlflowClient(tracking_uri=URI)


def revision_previa(c, train, test):
    """No repetir una ventana histórica ni abrir otra revisión pendiente."""
    exp = c.get_experiment_by_name('OralSolutions_Reentrenamiento')
    if exp is None:
        return None
    ht, he = huella(train), huella(test)
    # MLflow search_runs devuelve páginas; revisar todo el experimento.
    token = None
    while True:
        runs = c.search_runs([exp.experiment_id], max_results=1000, page_token=token)
        for run in runs:
            p, t = run.data.params, run.data.tags
            if t.get('decision_humana') == 'pendiente':
                return f'Revision pendiente: {run.info.run_id}'
            if t.get('fase') != 'reentrenamiento':
                continue
            misma_ventana = (
                p.get('snapshot_id') == SNAP
                and p.get('fecha_corte') == str(FECHA_EVAL_INICIO.date())
                and p.get('limite_train') == str(LIMITE_TRAIN.date())
                and p.get('lote_evaluacion') == 'L14'
            )
            if misma_ventana:
                if (run.info.status == 'FINISHED'
                    and t.get('estado_promocion') in ('rechazado', 'promovido', 'bloqueado_por_regla')
                    and p.get('hash_train') == ht and p.get('hash_eval') == he):
                    return f'Ventana ya procesada: {run.info.run_id}'
                return (f'Ventana historica existente: {run.info.run_id}; '
                        'revision manual requerida (huellas o estado no coinciden). '
                        'No se entrenara automaticamente.')
        token = runs.token
        if not token:
            return None


def entrenar(fecha_virtual, motivo, solo_verificar=False):
    fecha = pd.Timestamp(fecha_virtual)
    if fecha < FECHA_CALENDARIO:
        raise ValueError('El disparador por calendario requiere fecha virtual >= 2026-01-01.')
    c = cliente()
    mv = c.get_model_version_by_alias(NAME,'champion')
    corrida = c.get_run(mv.run_id).data
    umbral = float(mv.tags.get('umbral_decision', corrida.params.get('umbral_negocio','0.01')))
    std = float(corrida.metrics['pr_auc_cv_std'])
    df = dataset()
    train = df.loc[df[FECHA_EVENTO] <= LIMITE_TRAIN].copy()
    test = df.loc[df[FECHA_EVENTO].between(FECHA_EVAL_INICIO,FECHA_EVAL_FIN)].copy()
    if train.empty or len(test)!=81 or train[TARGET].nunique()!=2 or test[TARGET].nunique()!=2:
        raise ValueError('Ventanas inesperadas o sin ambas clases.')
    if train[FECHA_EVENTO].max()+pd.Timedelta(days=60)>=test[FECHA_EVENTO].min() or test[FECHA_EVENTO].max()+pd.Timedelta(days=60)>fecha:
        raise ValueError('Horizonte temporal inválido.')
    if set(train.sesion_key)&set(test.sesion_key):
        raise ValueError('Sesiones repetidas entre ventanas.')
    omitir = revision_previa(c, train, test)
    if omitir or solo_verificar:
        print(f'[OK] {omitir or "Ventana disponible; verificacion sin entrenamiento."}')
        print('::'+json.dumps({'outputs': {'entrenado': False, 'motivo': omitir or 'solo_verificar'}})+'::')
        return
    champion = mlflow.sklearn.load_model(f'models:/{NAME}/{mv.version}')
    # Mismo estimador e hiperparámetros, mismo preprocesador del champion.
    # Se usa el constructor compartido y se conserva su configuración exacta.
    pipe = construir_pipeline(clone(champion.named_steps['modelo']))
    pipe.set_params(preprocesador=clone(champion.named_steps['preprocesador']))
    X=train[FEATURES_FINALES]; y=train[TARGET].astype(int)
    # CV temporal por fechas, con purga de 60 días antes de cada validación.
    fechas=sorted(train[FECHA_EVENTO].unique())
    bloques=np.array_split(fechas,5)
    scores=[]
    for bloque in bloques[1:]:
        ini=pd.Timestamp(bloque[0]); fin=pd.Timestamp(bloque[-1])
        a=train.loc[train[FECHA_EVENTO]+pd.Timedelta(days=60)<ini]
        b=train.loc[train[FECHA_EVENTO].between(ini,fin)]
        if a[TARGET].nunique()!=2 or b[TARGET].nunique()!=2: continue
        p=clone(pipe).fit(a[FEATURES_FINALES],a[TARGET].astype(int))
        scores.append(evaluar(p,b,umbral)['pr_auc'])
    if len(scores)<2: raise ValueError('CV temporal insuficiente.')
    mlflow.set_experiment('OralSolutions_Reentrenamiento')
    with mlflow.start_run(run_name='REENTRENAMIENTO_L14') as run:
        pipe.fit(X,y)
        a=evaluar(champion,test,umbral); b=evaluar(pipe,test,umbral)
        cumple=bool(b['costo_errores_usd']<a['costo_errores_usd'] and b['pr_auc']>=a['pr_auc']-std)
        mlflow.log_params(dict(snapshot_id=SNAP, hash_dataset=huella(df), hash_train=huella(train), hash_eval=huella(test),
            fecha_corte=str(FECHA_EVAL_INICIO.date()), horizonte_dias=60, limite_train=str(LIMITE_TRAIN.date()),
            fecha_virtual=str(fecha.date()), lote_evaluacion='L14', n_filas_entrenamiento=len(train),
            n_filas_evaluacion=len(test), champion_previo_version=str(mv.version), champion_previo_run=mv.run_id,
            umbral_negocio=umbral, tolerancia_cv=std, costo_fn=120,costo_fp=3))
        mlflow.log_metrics({**{f'{k}_champion_eval':v for k,v in a.items()}, **{f'{k}_challenger_eval':v for k,v in b.items()},
             'pr_auc_cv_mean':float(np.mean(scores)), 'pr_auc_cv_std':float(np.std(scores)), 'cumple_regla_promocion':int(cumple)})
        mlflow.set_tags(dict(fase='reentrenamiento',motivo_disparo=motivo,decision_humana='pendiente',estado_promocion='pendiente'))
        ejemplo=X.head(5)
        mlflow.sklearn.log_model(pipe,artifact_path='model',signature=infer_signature(ejemplo,pipe.predict(ejemplo)),input_example=ejemplo)
        rid=run.info.run_id
    nuevo=mlflow.register_model(f'runs:/{rid}/model',NAME,await_registration_for=120)
    v=str(nuevo.version)
    c.set_model_version_tag(NAME,v,'umbral_decision',str(umbral))
    c.set_tag(rid,'version_candidata',v)
    c.set_registered_model_alias(NAME,'challenger',v)
    print(json.dumps(dict(champion=dict(version=mv.version,**a),challenger=dict(version=v,**b),cumple_regla=cumple,
          n_train=len(train),n_eval=len(test),limite_train=str(LIMITE_TRAIN.date()),fecha_virtual=str(fecha.date())),indent=2))
    print('::'+json.dumps({'outputs':{'run_id':rid,'version':v,'cumple_regla':cumple,'entrenado':True}})+'::')


def decidir(run_id, decision, motivo):
    c=cliente(); r=c.get_run(run_id).data
    if r.tags.get('fase')!='reentrenamiento' or r.tags.get('decision_humana')!='pendiente':
        raise ValueError('La corrida no está pendiente de decisión.')
    v=r.tags['version_candidata']; nuevo=c.get_model_version(NAME,v)
    actual=c.get_model_version_by_alias(NAME,'champion')
    if nuevo.run_id!=run_id or str(actual.version)!=r.params['champion_previo_version'] or actual.run_id!=r.params['champion_previo_run']:
        raise RuntimeError('Cambió el candidato o champion desde la evaluación; revisar manualmente.')
    cumple=(r.metrics['costo_errores_usd_challenger_eval']<r.metrics['costo_errores_usd_champion_eval'] and
        r.metrics['pr_auc_challenger_eval']>=r.metrics['pr_auc_champion_eval']-float(r.params['tolerancia_cv']))
    promovido=decision=='aprobar' and cumple
    if promovido: c.set_registered_model_alias(NAME,'champion',v)
    estado='promovido' if promovido else ('bloqueado_por_regla' if decision=='aprobar' else 'rechazado')
    for k,val in dict(decision_humana=decision,motivo_decision=motivo,estado_promocion=estado).items(): c.set_tag(run_id,k,val)
    print(f'[OK] {estado}; champion v{c.get_model_version_by_alias(NAME,"champion").version}; candidato v{v}; motivo: {motivo}')


def exportar():
    cliente(); docs=ROOT/'docs'; docs.mkdir(exist_ok=True)
    runs=mlflow.search_runs(search_all_experiments=True)
    runs.to_csv(docs/'corridas.csv',index=False)
    runs.loc[runs['tags.fase'].eq('reentrenamiento')].to_csv(docs/'reentrenamientos.csv',index=False)
    print('[OK] Corridas exportadas.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='accion',required=True)
    t=sub.add_parser('entrenar'); t.add_argument('--fecha-virtual',default='2026-01-01'); t.add_argument('--motivo',default='Calendario: revision programada posterior a simulacion')
    t.add_argument('--solo-verificar', action='store_true')
    d=sub.add_parser('decidir'); d.add_argument('--run-id',required=True); d.add_argument('--decision',choices=['aprobar','rechazar'],required=True); d.add_argument('--motivo',required=True)
    sub.add_parser('exportar'); a=p.parse_args()
    if a.accion=='entrenar': entrenar(a.fecha_virtual,a.motivo,a.solo_verificar)
    elif a.accion=='decidir': decidir(a.run_id,a.decision,a.motivo)
    else: exportar()
