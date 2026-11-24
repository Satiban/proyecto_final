"""
Esquemas Pydantic de entrada y salida para la API de predicción de OralSolutions.
Valida tipos, rangos clínicos, categorías permitidas y consistencia de negocio
(cualquier violación devuelve HTTP 422 con el detalle del campo).
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

CodigoProcedimiento = Literal["CROWN", "EXTRACT", "FILL1", "FILL2", "ROOT", "WHITEN"]
OdontologoKey = Literal["1", "2", "3", "4", "5"]


class SolicitudPrediccion(BaseModel):
    duracion_estimada_min: int = Field(
        ..., ge=15, le=240, description="Duración estimada de la sesión en minutos (30-90 en histórico)."
    )
    numero_sesion: int = Field(
        ..., ge=1, le=20, description="Número ordinal de la sesión que acaba de finalizar."
    )
    sesiones_planificadas: int = Field(
        ..., ge=2, le=20, description="Total de sesiones planificadas en el tratamiento."
    )
    cancelaciones_previas: int = Field(
        ..., ge=0, le=50, description="Cancelaciones acumuladas antes de la sesión actual."
    )
    inasistencias_previas: int = Field(
        ..., ge=0, le=50, description="Inasistencias acumuladas antes de la sesión actual."
    )
    dias_desde_inicio_plan: int = Field(
        ..., ge=0, le=730, description="Días transcurridos desde fecha_inicio_plan hasta fecha_realizada."
    )
    edad_en_atencion: int = Field(
        ..., ge=0, le=120, description="Edad del paciente en años al momento de la consulta."
    )
    codigo_procedimiento: CodigoProcedimiento = Field(
        ..., description="Código del procedimiento odontológico prescrito."
    )
    odontologo_key: OdontologoKey = Field(
        ..., description="Identificador nominal del odontólogo ('1' a '5')."
    )

    @model_validator(mode="after")
    def validar_continuidad_pendiente(self) -> "SolicitudPrediccion":
        if self.numero_sesion >= self.sesiones_planificadas:
            raise ValueError(
                "numero_sesion debe ser estrictamente menor que sesiones_planificadas "
                "(el modelo solo evalúa sesiones con continuidad pendiente)."
            )
        return self

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "duracion_estimada_min": 75,
                "numero_sesion": 1,
                "sesiones_planificadas": 3,
                "cancelaciones_previas": 0,
                "inasistencias_previas": 0,
                "dias_desde_inicio_plan": 14,
                "edad_en_atencion": 42,
                "codigo_procedimiento": "CROWN",
                "odontologo_key": "2",
            }
        }
    )


class RespuestaSalud(BaseModel):
    estado: str
    modelo_cargado: bool
    modo_respaldo: bool
    nombre_modelo: str
    version_modelo: str


class RespuestaModelo(BaseModel):
    nombre_modelo: str
    version: str
    alias: str
    run_id: str
    umbral_decision: float
    modo_respaldo: bool


class RespuestaPrediccion(BaseModel):
    id_prediccion: str
    probabilidad: float
    decision: Literal["contactar", "no_contactar"]
    accion: str
    umbral_aplicado: float
    version_modelo: str
    run_id: str
    uso_respaldo: bool
    latencia_ms: float
    timestamp: str


class SolicitudRespaldo(BaseModel):
    activar: bool = Field(..., description="True activa la regla de negocio de respaldo; False vuelve al modelo.")
    motivo: str = Field(default="Cambio manual operativo", description="Motivo del cambio de modo.")