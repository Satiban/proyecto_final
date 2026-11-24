"""Reglas operativas: generan eventos, nunca ejecutan acciones automáticas."""
from __future__ import annotations
from datetime import datetime, timezone
from monitoreo.deriva import UMBRAL_PSI_ADVERTENCIA, UMBRAL_PSI_CRITICO

UMBRAL_TASA_ERROR_HTTP = .05
UMBRAL_LATENCIA_P95_MS = 250.0  # provisional, para latencia HTTP completa


def evaluar_alertas_lote(lote_id, psi_max, feature_psi_max, pr_auc_lote=None,
                        tasa_error_http=None, latencia_p95_ms=None,
                        pct_faltantes_max=None, *, media_cv=None, std_cv=None,
                        n_observaciones=None, pct_fuera_rango_max=None,
                        pct_categorias_no_vistas_max=None):
    alertas = []
    def agregar(codigo, metrica, valor, umbral, severidad, destinatario, accion, ventana='1 lote'):
        alertas.append(dict(codigo=codigo, timestamp=datetime.now(timezone.utc).isoformat(),
            lote_id=lote_id, metrica=metrica, valor_observado=float(valor), umbral=umbral,
            ventana=ventana, severidad=severidad, destinatario=destinatario, accion=accion))
    datos = 'Equipo de Datos / MLOps'
    plataforma = 'Equipo de Plataforma / API'
    if psi_max is not None and psi_max >= UMBRAL_PSI_ADVERTENCIA:
        alta = psi_max >= UMBRAL_PSI_CRITICO
        umbral = UMBRAL_PSI_CRITICO if alta else UMBRAL_PSI_ADVERTENCIA
        nota = ' Muestra pequeña: interpretar con cautela.' if n_observaciones is not None and n_observaciones < 30 else ''
        agregar('psi', f'PSI de {feature_psi_max}', psi_max, f'>= {umbral}',
                'Alta' if alta else 'Media', datos, 'Revisar distribución y calidad; esperar target maduro.'+nota)
    if pr_auc_lote is not None:
        if media_cv is None or std_cv is None:
            raise ValueError('Se requiere referencia CV del run histórico.')
        piso, grave = max(0., media_cv-std_cv), max(0., media_cv-2*std_cv)
        if pr_auc_lote < piso:
            critica = pr_auc_lote < grave
            agregar('pr_auc', 'PR-AUC del lote', pr_auc_lote, f'< {grave if critica else piso:.6f}',
                    'Crítica' if critica else 'Alta', datos,
                    'Revisar desempeño y persistencia en lotes maduros antes de decidir reentrenamiento.', '1 lote etiquetado')
    for codigo, metrica, valor, limite in [
        ('faltantes','Porcentaje máximo de faltantes',pct_faltantes_max,5.),
        ('fuera_rango','Porcentaje fuera del rango de entrenamiento',pct_fuera_rango_max,5.),
        ('categorias_nuevas','Porcentaje de categorías no vistas',pct_categorias_no_vistas_max,0.)]:
        if valor is not None and valor > limite:
            agregar(codigo,metrica,valor,f'> {limite}%', 'Media',datos,'Auditar entradas y fuente de datos.')
    if tasa_error_http is not None and tasa_error_http > UMBRAL_TASA_ERROR_HTTP:
        agregar('http','Tasa errores HTTP',tasa_error_http,'> 5%', 'Alta',plataforma,'Revisar respuestas 4xx/5xx y logs.')
    if latencia_p95_ms is not None and latencia_p95_ms > UMBRAL_LATENCIA_P95_MS:
        agregar('latencia','Latencia HTTP p95 (ms)',latencia_p95_ms,'> 250 ms','Media',plataforma,'Revisar latencia de extremo a extremo y persistencia.')
    return alertas
