import time
from pathlib import Path
import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1bDd9OOZ1SPTLjnGQld5IIn-x4Gp8KQIh?usp=drive_link"

ARCHIVOS_ESPERADOS = [
    "appointments.csv",
    "dentists.csv",
    "insurers.csv",
    "invoices.csv",
    "missed_appointments.csv",
    "patient_followups.csv",
    "patients.csv",
    "procedures.csv",
    "treatment_lines.csv",
    "treatment_plans.csv",
    "treatment_sessions.csv",
]


def contar_registros(ruta_raw: Path) -> int:
    """Cuenta el total de registros (sin encabezados) en los 11 CSV."""
    total_filas = 0
    for archivo in ARCHIVOS_ESPERADOS:
        with open(ruta_raw / archivo, "r", encoding="utf-8") as f:
            total_filas += max(sum(1 for _ in f) - 1, 0)
    return total_filas


def descargar_desde_drive(ruta_raw: Path) -> None:
    """Descarga los CSV faltantes desde la carpeta pública de Google Drive hacia data/raw/."""
    print(f"[INFO] Descargando archivos desde Google Drive hacia {ruta_raw}...")
    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(ruta_raw),
        quiet=False,
        use_cookies=False,
    )


def verificar_o_descargar_ingesta(
    ruta_raw: Path, max_reintentos: int = 3, espera_seg: int = 5
) -> None:
    """Verifica primero si los 11 CSV existen localmente; si faltan, los descarga con reintentos."""
    ruta_raw.mkdir(parents=True, exist_ok=True)

    # 1. Comprobación inicial: si ya existen todos localmente, no descarga nada
    faltantes = [f for f in ARCHIVOS_ESPERADOS if not (ruta_raw / f).exists()]
    if not faltantes:
        total_filas = contar_registros(ruta_raw)
        print(
            f"[OK] Fuentes locales encontradas: {len(ARCHIVOS_ESPERADOS)} CSV verificados en {ruta_raw} ({total_filas:,} registros)."
        )
        return

    # 2. Si falta algún archivo, intenta descargar desde Google Drive con reintentos
    for intento in range(1, max_reintentos + 1):
        print(f"[Intento {intento}/{max_reintentos}] Faltan {len(faltantes)} archivos en local: {faltantes}")
        try:
            descargar_desde_drive(ruta_raw)
        except Exception as e:
            print(f"[WARN] Error al descargar desde Google Drive: {e}")

        faltantes = [f for f in ARCHIVOS_ESPERADOS if not (ruta_raw / f).exists()]
        if not faltantes:
            total_filas = contar_registros(ruta_raw)
            print(
                f"[OK] Descarga completada: {len(ARCHIVOS_ESPERADOS)} CSV verificados en {ruta_raw} ({total_filas:,} registros)."
            )
            return

        if intento < max_reintentos:
            time.sleep(espera_seg)

    raise FileNotFoundError(
        f"No se pudieron obtener todos los archivos en {ruta_raw}. Faltantes: {faltantes}"
    )


if __name__ == "__main__":
    ruta_datos = Path(__file__).resolve().parent.parent / "data" / "raw"
    verificar_o_descargar_ingesta(ruta_datos)