"""Punto de entrada para la limpieza de datos de OralSolutions.

Todas las reglas de limpieza, normalizacion y validacion de calidad se
centralizan exclusivamente en la capa Silver de dbt
(dbt/models/silver/ y dbt/tests/silver/), evitando
duplicar transformaciones en Python.
"""

from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"
ENV_FILE = BASE_DIR / ".env"


def ejecutar_dbt(comando: list[str]) -> None:
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(ENV_FILE),
        "--project-name",
        "proyecto_final",
        "exec",
        "-T",
        "dbt",
        *comando,
    ]
    print(f"[INFO] Ejecutando: {' '.join(comando)}")
    resultado = subprocess.run(cmd, cwd=BASE_DIR, check=False)
    if resultado.returncode != 0:
        sys.exit(resultado.returncode)


def main() -> None:
    print("=" * 75)
    print("ORALSOLUTIONS - LIMPIEZA CENTRALIZADA EN DBT (CAPA SILVER)")
    print("=" * 75)
    ejecutar_dbt(["dbt", "run", "--select", "silver"])
    ejecutar_dbt(["dbt", "test", "--select", "silver"])
    print("[OK] Limpieza y pruebas de calidad centralizadas en dbt Silver.")


if __name__ == "__main__":
    main()