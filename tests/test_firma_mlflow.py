"""Contrato HTTP -> features -> firma MLflow, sin servicios externos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.models import Model
from mlflow.models.utils import _enforce_schema
from pydantic import ValidationError

from api.esquemas import SolicitudPrediccion
from src.pipeline import FEATURES_FINALES, construir_features

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
PAYLOAD = dict(duracion_estimada_min=75, numero_sesion=2,
               sesiones_planificadas=3, cancelaciones_previas=1,
               inasistencias_previas=2, dias_desde_inicio_plan=15,
               edad_en_atencion=42, codigo_procedimiento='CROWN',
               odontologo_key='2')
DERIVADAS = {'progreso_plan', 'ritmo_dias_por_sesion',
             'friccion_previa', 'complejidad_proc'}


def test_contrato_transformado_coincide_con_firma_mlflow():
    ruta = FIXTURES / 'modelo_champion' / 'MLmodel'
    modelo = Model.load(str(ruta))
    firma = modelo.signature.inputs
    meta = json.loads((FIXTURES / 'modelo_meta.json').read_text(encoding='utf-8'))
    especificaciones = json.loads(yaml.safe_load(ruta.read_text(encoding='utf-8'))['signature']['inputs'])
    solicitud = SolicitudPrediccion.model_validate(PAYLOAD)
    assert set(solicitud.model_dump()) == set(PAYLOAD)
    assert set(FEATURES_FINALES) - set(PAYLOAD) == DERIVADAS
    X = construir_features(pd.DataFrame([solicitud.model_dump()]))[FEATURES_FINALES]
    assert list(X.columns) == firma.input_names() == meta['features'] == FEATURES_FINALES
    assert len(X.columns) == 13
    for spec in especificaciones:
        assert spec['required'] is True
        serie = X[spec['name']]
        assert not serie.isna().any()
        if spec['type'] == 'long':
            assert serie.dtype == np.dtype('int64')
        elif spec['type'] == 'double':
            assert serie.dtype == np.dtype('float64')
        elif spec['type'] == 'string':
            assert serie.map(lambda v: isinstance(v, str)).all()
        else:
            pytest.fail(f"Tipo inesperado: {spec['type']}")
    assert X.iloc[0]['progreso_plan'] == pytest.approx(2/3)
    assert X.iloc[0]['ritmo_dias_por_sesion'] == pytest.approx(7.5)
    assert X.iloc[0]['friccion_previa'] == 3
    assert X.iloc[0]['complejidad_proc'] == 'extenso_especializado'
    # El validador de MLflow comprueba la compatibilidad efectiva del DataFrame.
    validado = _enforce_schema(X.copy(), firma)
    assert list(validado.columns) == FEATURES_FINALES


@pytest.mark.parametrize('campo,valor', [('progreso_plan', 0.1),
    ('ritmo_dias_por_sesion', 100.0), ('friccion_previa', 0),
    ('complejidad_proc', 'restaurativo_corto')])
def test_contrato_rechaza_derivadas_proporcionadas_por_cliente(campo, valor):
    with pytest.raises(ValidationError) as exc:
        SolicitudPrediccion.model_validate({**PAYLOAD, campo: valor})
    assert any(e['loc'] == (campo,) and e['type'] == 'extra_forbidden'
               for e in exc.value.errors())
