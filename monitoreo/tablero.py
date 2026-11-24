"""Dashboard MLOps y recepción: fuentes Snowflake, FastAPI y MLflow."""
from __future__ import annotations
import os
import json
import base64
from pathlib import Path
import pandas as pd
import altair as alt
import requests
import streamlit as st
import snowflake.connector
from mlflow.tracking import MlflowClient
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass

API_URL=os.getenv('API_URL','http://127.0.0.1:8000')
URI=os.getenv('MLFLOW_TRACKING_URI','http://mlflow:5000' if 'http://api:' in API_URL else 'http://127.0.0.1:5000')
SCHEMA='ORALSOLUTIONS_DW.MLOPS'
SIM='SIM_2025_V1'
st.set_page_config(page_title='OralSolutions — MLOps',layout='wide')
st.title('OralSolutions — Monitoreo y recepción')
st.caption('Simulación histórica con datos sintéticos. No representa impacto clínico medido.')

@st.cache_data(ttl=60,show_spinner='Consultando Snowflake...')
def cargar():
    conn=snowflake.connector.connect(account=os.environ['SF_ACCOUNT'],user=os.environ['SF_USER'],
        password=base64.b64decode(os.environ['SECRET_SNOWFLAKE_PASSWORD'],validate=True).decode('utf-8'),
        warehouse=os.getenv('SF_WAREHOUSE','COMPUTE_WH'),database='ORALSOLUTIONS_DW',schema='MLOPS',role=os.getenv('SF_ROLE','ACCOUNTADMIN'))
    try:
        def query(tabla, extra=''):
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {SCHEMA}.{tabla} WHERE SIMULACION_ID=%s AND LOTE_ID BETWEEN 'L01' AND 'L08' {extra}",(SIM,))
                return pd.DataFrame(cur.fetchall(),columns=[c[0].lower() for c in cur.description])
        return {t:query(t,"AND ORIGEN='simulacion'" if t=='PREDICCIONES' else '') for t in ['LOTES_PRODUCCION','MONITOREO','MONITOREO_INMEDIATO','DERIVA_FEATURES','ALERTAS','PREDICCIONES']}
    finally: conn.close()

def consultar_api(endpoint):
    try:
        r=requests.get(API_URL+endpoint,timeout=10); r.raise_for_status(); return r.json()
    except requests.RequestException: return None

@st.cache_data(ttl=60)
def corridas():
    c=MlflowClient(tracking_uri=URI)
    ex=c.search_experiments()
    if not ex:return []
    runs=c.search_runs([e.experiment_id for e in ex],filter_string="tags.fase = 'reentrenamiento'",order_by=['attributes.start_time DESC'])
    return [{**r.data.tags, **r.data.params, **r.data.metrics, "run_id":r.info.run_id} for r in runs]

if st.button('Actualizar datos'):
    st.cache_data.clear(); st.rerun()
try:
    datos=cargar()
except Exception as e:
    st.error(f'No fue posible consultar Snowflake ({type(e).__name__}). Revisa configuración y conexión.'); st.stop()
m=datos['MONITOREO'].sort_values('lote_id'); i=datos['MONITOREO_INMEDIATO'].sort_values('lote_id'); a=datos['ALERTAS']; p=datos['PREDICCIONES']; d=datos['DERIVA_FEATURES']; l=datos['LOTES_PRODUCCION']
mon,recep=st.tabs(['Monitoreo MLOps','Panel de recepción'])

def linea(df, campos, titulo, dominio=None, umbral=None):
    if df.empty:return
    t=df.melt(id_vars=['lote_id'],value_vars=campos,var_name='indicador',value_name='valor')
    y=alt.Y('valor:Q',title=titulo,scale=alt.Scale(domain=dominio)) if dominio else alt.Y('valor:Q',title=titulo)
    chart=alt.Chart(t).mark_line(point=True).encode(x=alt.X('lote_id:N',title='Lote',sort=sorted(df.lote_id.unique())),y=y,color='indicador:N',tooltip=['lote_id','indicador','valor'])
    if umbral is not None: chart=chart+alt.Chart(pd.DataFrame({'umbral':[umbral]})).mark_rule(color='red',strokeDash=[5,4]).encode(y='umbral:Q')
    st.altair_chart(chart,use_container_width=True)

with mon:
    st.header('1. Servicio en vivo')
    salud=consultar_api('/salud'); modelo=consultar_api('/modelo'); metrics=consultar_api('/metricas'); live=consultar_api('/alertas')
    if salud and modelo:
        c=st.columns(4); c[0].metric('API',salud['estado']); c[1].metric('Versión activa',modelo['version']); c[2].metric('Respaldo',str(modelo['modo_respaldo'])); c[3].metric('Umbral',modelo['umbral_decision']); st.caption(f"Run: {modelo['run_id']}")
        if metrics: st.json(metrics,expanded=False)
        if live and live.get('alertas'): st.dataframe(pd.DataFrame(live['alertas']))
    else: st.warning('API no disponible; se muestran resultados históricos de Snowflake.')
    st.header('2. Simulación L01–L08')
    tp=int(m.tp.sum()); fp=int(m.fp.sum()); fn=int(m.fn.sum()); total=len(p)
    c=st.columns(4); c[0].metric('Lotes evaluados',len(m)); c[1].metric('Predicciones',total); c[2].metric('Recall agregado',f'{tp/(tp+fn):.1%}' if tp+fn else 'N/D'); c[3].metric('Alertas',len(a))
    st.caption(f'Precisión agregada: {tp/(tp+fp):.1%}' if tp+fp else 'Precisión no disponible')
    st.dataframe(l[['lote_id','estado','total_predicciones','fecha_maduracion','fecha_evaluacion_simulada']],hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        st.subheader('PR-AUC por lote')
        linea(m,['pr_auc'],'PR-AUC',[0,1],0.18522254921481657-0.05966934121023218)
        st.caption('Referencia histórica v1: media CV − 1 desviación estándar. No es prueba causal de concept drift.')
    with c2:
        st.subheader('PSI máximo')
        if not i.empty:
            chart=alt.Chart(i).mark_bar().encode(x='lote_id:N',y='psi_max:Q',color='deriva_inducida:N',tooltip=['lote_id','psi_max','feature_psi_max','muestra_pequena'])
            st.altair_chart(chart+alt.Chart(pd.DataFrame({'umbral':[.25]})).mark_rule(color='red',strokeDash=[5,4]).encode(y='umbral:Q'),use_container_width=True)
        st.caption('PSI ≥ 0,25: alerta alta; las muestras pequeñas requieren cautela.')
    st.subheader('Distribución de scores y decisiones')
    linea(i,['probabilidad_p05','probabilidad_p50','probabilidad_p95','tasa_contactar'],'Probabilidad / proporción',[0,1])
    st.subheader('Telemetría HTTP por lote')
    linea(i,['latencia_http_p50_ms','latencia_http_p95_ms'],'Latencia HTTP (ms)',umbral=250)
    st.dataframe(i[['lote_id','volumen_http','errores_4xx','errores_5xx','errores_transporte','tasa_error_4xx','tasa_error_5xx','latencia_http_p50_ms','latencia_http_p95_ms','latencia_inferencia_p95_ms']],hide_index=True)
    st.caption('L01 no tiene telemetría HTTP histórica. NULL no significa cero.')
    st.subheader('Calidad y deriva por variable')
    st.dataframe(d[['lote_id','feature','psi','prueba_estadistica','p_valor','prueba_valida','pct_faltantes','pct_fuera_rango','pct_categorias_no_vistas']],hide_index=True)
    st.subheader('3. Costo estimado de errores (USD)')
    cols=['costo_modelo_usd','costo_regla_1_sesion_usd','costo_llamar_todos_usd','costo_no_hacer_nada_usd']
    linea(m,cols,'USD'); st.dataframe(m[cols].sum().rename('Total USD').to_frame())
    st.caption('Costos simulados: 120 USD por FN y 3 USD por FP; no son ahorros observados.')
    st.subheader('4. Alertas')
    st.dataframe(a.sort_values(['fecha_simulada','lote_id'])[['lote_id','fecha_simulada','fase','metrica','valor_observado','umbral','ventana','severidad','destinatario','accion']],hide_index=True)
    st.subheader('5. Reentrenamiento y decisiones humanas')
    try:
        r=pd.DataFrame(corridas())
        if r.empty: st.info('Todavía no hay corridas de reentrenamiento.')
        else:
            cols=['run_id','fecha_virtual','lote_evaluacion','champion_previo_version','version_candidata','pr_auc_champion_eval','pr_auc_challenger_eval','costo_errores_usd_champion_eval','costo_errores_usd_challenger_eval','cumple_regla_promocion','decision_humana','estado_promocion','motivo_decision']
            st.dataframe(r[[k for k in cols if k in r]],hide_index=True)
    except Exception as e: st.warning(f'MLflow no disponible ({type(e).__name__}).')

with recep:
    st.header('Priorización de seguimiento — simulación')
    st.caption('Apoyo administrativo, no decisión clínica. Revisión humana antes de contactar; los scores no son certezas.')
    if not p.empty:
        entradas=pd.json_normalize(p.entradas_json.map(json.loads)); pp=pd.concat([p.reset_index(drop=True),entradas.loc[:,~entradas.columns.isin(p.columns)]],axis=1)
        lote=st.selectbox('Lote',sorted(pp.lote_id.unique()),index=len(pp.lote_id.unique())-1)
        decisiones=st.multiselect('Acciones',['contactar','no_contactar'],default=['contactar'])
        vista=pp.loc[pp.lote_id.eq(lote)&pp.decision.isin(decisiones)].sort_values('probabilidad',ascending=False)
        st.metric('Sesiones seleccionadas',len(vista)); st.dataframe(vista[[k for k in ['sesion_key','paciente_key','probabilidad','decision','codigo_procedimiento','numero_sesion','sesiones_planificadas','version_modelo','id_prediccion'] if k in vista]],hide_index=True)
    st.divider(); st.subheader('Consultar FastAPI con una entrada de ejemplo')
    with st.form('prediccion'):
        c=st.columns(3)
        proc=c[0].selectbox('Procedimiento',['CROWN','EXTRACT','FILL1','FILL2','ROOT','WHITEN']); odo=c[1].selectbox('Odontólogo',['1','2','3','4','5']); dur=c[2].number_input('Duración (min)',15,240,75)
        plan=c[0].number_input('Sesiones planificadas',2,12,3); ses=c[1].number_input('Sesión actual',1,11,1); dias=c[2].number_input('Días desde inicio',0,365,14)
        can=c[0].number_input('Cancelaciones',0,20,0); ina=c[1].number_input('Inasistencias',0,20,0); edad=c[2].number_input('Edad',5,100,42)
        enviar=st.form_submit_button('Consultar riesgo')
    if enviar:
        if ses>=plan: st.error('La sesión actual debe ser anterior a la última sesión planificada.')
        else:
            try:
                resp=requests.post(API_URL+'/predecir',json=dict(codigo_procedimiento=proc,odontologo_key=odo,duracion_estimada_min=int(dur),numero_sesion=int(ses),sesiones_planificadas=int(plan),dias_desde_inicio_plan=int(dias),cancelaciones_previas=int(can),inasistencias_previas=int(ina),edad_en_atencion=int(edad)),timeout=60)
                if resp.ok: st.json(resp.json())
                else: st.error(f'HTTP {resp.status_code}: {resp.text}')
            except requests.RequestException: st.error('No se pudo completar la consulta a FastAPI.')
