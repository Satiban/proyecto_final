"""
Coordina el monitoreo inmediato y diferido de OralSolutions.

Permite procesar el cierre de un lote o avanzar el reloj
virtual sin generar nuevas predicciones.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from src.datos import conectar_snowflake

from monitoreo.simular_produccion import (
    SIMULACION_ID,
    LOTES,
    consultar_filas,
)

from monitoreo.monitorear_lote import monitorear
from monitoreo.evaluar_lotes import evaluar_lotes


def obtener_fecha_lote(numero: int) -> date:
    """Devuelve el domingo de cierre del lote semanal."""

    if not 1 <= numero <= 14:
        raise ValueError("El lote debe estar entre 1 y 14.")

    return date(2025, 8, 3) + timedelta(
        weeks=numero - 1
    )


def obtener_lotes_evaluados(fecha_virtual: date) -> list[int]:

    conn = conectar_snowflake()

    try:
        filas = consultar_filas(
            conn,
            f"""
            SELECT NUMERO_LOTE
            FROM {LOTES}
            WHERE SIMULACION_ID = %s
                AND ESTADO = 'evaluado'
                AND FECHA_EVALUACION_SIMULADA = %s
            ORDER BY NUMERO_LOTE
            """,
            (SIMULACION_ID, fecha_virtual),
        )

        return [int(fila[0]) for fila in filas]

    finally:
        conn.close()


def ejecutar_monitoreo(
    numero: int = 0,
    fecha_virtual: date | None = None,
) -> None:

    if numero:
        cierre = obtener_fecha_lote(numero)

        if fecha_virtual is not None and fecha_virtual != cierre:
            raise ValueError(
                "La fecha virtual debe coincidir "
                "con el cierre del lote."
            )

        fecha_virtual = cierre

    if fecha_virtual is None:
        raise ValueError(
            "Debe indicar un lote o una fecha virtual."
        )

    print("=" * 65)
    print("ORALSOLUTIONS - MONITOREO ORQUESTADO")
    print("=" * 65)

    print(f"Simulación: {SIMULACION_ID}")
    print(f"Fecha virtual: {fecha_virtual}")

    # 1. Monitoreo inmediato del lote actual.

    if numero:
        print(f"\n1. MONITOREO INMEDIATO L{numero:02d}")

        monitorear(numero, fecha_virtual)

    else:
        print("\n1. Sin nuevo lote de inferencia.")

    # 2. Evaluación de lotes con target madurado.

    print("\n2. EVALUACIÓN DE TARGETS MADURADOS")

    evaluar_lotes(fecha_virtual)

    # 3. Generar alertas supervisadas para las evaluaciones
    # realizadas en esta fecha virtual.

    print("\n3. ACTUALIZACIÓN DE ALERTAS SUPERVISADAS")

    evaluados = obtener_lotes_evaluados(fecha_virtual)

    for lote_evaluado in evaluados:

        print(f"\nActualizando L{lote_evaluado:02d}")

        monitorear(lote_evaluado, fecha_virtual)

    if not evaluados:
        print("Sin evaluaciones nuevas para esta fecha.")

    print("\n[OK] Monitoreo orquestado completado.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--lote",
        type=int,
        default=0,
        choices=range(0, 15),
    )

    parser.add_argument(
        "--fecha-virtual",
        type=date.fromisoformat,
        default=None,
    )

    args = parser.parse_args()

    ejecutar_monitoreo(
        numero=args.lote,
        fecha_virtual=args.fecha_virtual,
    )