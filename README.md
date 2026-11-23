# OralSolutions — Predicción de Abandono de Tratamientos Odontológicos

Proyecto de ciencia de datos que prepara y modela los registros clínicos, operativos y financieros de la clínica odontológica **OralSolutions** con el objetivo de predecir la probabilidad de que un paciente activo abandone su tratamiento en los próximos 60 días tras cada consulta atendida. Con esta predicción, el equipo administrativo y clínico prioriza recordatorios personalizados y reprogramaciones preventivas antes de que se pierda la continuidad del tratamiento, reemplazando la gestión puramente reactiva actual.

**Métrica de éxito:** Incrementar en **10 puntos porcentuales** la tasa de continuidad de tratamientos a 60 días frente al seguimiento reactivo habitual (línea base), activando alertas preventivas cuando la probabilidad predicha de abandono sea >= 1% (u = 0.01) (umbral justificado porque el costo operativo de un contacto preventivo es significativamente menor al costo clínico y financiero de perder un tratamiento recurrente).

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| Dataset clínico-operativo OralSolutions (11 archivos CSV) | 38,603 registros distribuidos en: `patients.csv`, `dentists.csv`, `insurers.csv`, `procedures.csv`, `treatment_plans.csv`, `treatment_sessions.csv`, `treatment_lines.csv`, `appointments.csv`, `missed_appointments.csv`, `patient_followups.csv` e `invoices.csv` (citas, planes de tratamiento, sesiones, facturación, aseguradoras y seguimientos). | Descarga automática desde Google Drive mediante `src/ingestar.py` (o automáticamente al ejecutar Kestra en `ingest_raw_oralsolutions.yml`), sin requerir descargas manuales previas. | `data/raw/*.csv` |

Los datos no están en el repositorio (ignorados por `.gitignore`). Para obtener y verificar los archivos desde cero (desde ``):

- **Alternativa 1 — Descarga y verificación automática (recomendada):**
  Ejecutar `src/ingestar.py`, el cual comprueba si los 11 CSV ya existen localmente y, de no estar presentes, los descarga automáticamente desde Google Drive y verifica sus **38,603 registros**:

```powershell
  # En local (con el entorno virtual activado)
  python src/ingestar.py

  # O mediante contenedor Docker aislado
  docker build -t oralsolutions-ingesta .
  docker run --rm -v "${PWD}\data\raw:/app/data/raw" oralsolutions-ingesta
```
- **Alternativa 2 — Descarga manual de respaldo:** Si la descarga automática falla por restricciones de red, descargar manualmente los 11 archivos CSV desde la [carpeta de Google Drive](https://drive.google.com/drive/folders/1bDd9OOZ1SPTLjnGQld5IIn-x4Gp8KQIh?usp=drive_link), pegarlos dentro de `data/raw/` y volver a ejecutar `python src/ingestar.py` para confirmar que todos los archivos y conteos estén íntegros.

## Cómo correr el pipeline de datos

Implementa un pipeline ELT reproducible que procesa los **11 archivos CSV** mediante **Kestra → Snowflake Raw → dbt (Bronze, Silver y Gold) → Apache Spark → Snowflake OBT**, además de incluir una cadena local de calidad y warehouse en **DuckDB** (`calidad/` y `warehouse/construir_warehouse.py`).

### 1. Estructura del proyecto

```text

├── Dockerfile                   # Imagen base de ingesta y validación
├── Dockerfile.dbt               # Imagen de dbt Core con adaptador Snowflake
├── requirements.txt             # Dependencias Python del proyecto
├── .env.example                 # Plantilla de variables de entorno
├── data/
│   ├── raw/                     # 11 CSV de origen (ignorados por Git)
│   └── processed/               # Dataset limpio de modelado (obt_riesgo_abandono.csv)
├── ingesta/
│   └── ingestar.py              # Descarga desde Drive y verificación de los 11 CSV
├── calidad/
│   ├── diagnostico_proyecto.ipynb # Análisis de calidad por tabla y dimensión
│   └── limpiar.py               # Ejecutor de la limpieza centralizada en dbt Silver
├── warehouse/
│   ├── construir_warehouse.py   # Carga y verificación de filas en DuckDB
│   └── dbt/                     # Modelos y pruebas Bronze, Silver y Gold
├── procesamiento/
│   └── job_spark.py             # Construcción y validación de la OBT con Spark
├── kestra/
│   ├── ingest_raw_oralsolutions.yml
│   ├── test_dbt_docker.yml
│   ├── test_spark_docker.yml
│   ├── backfill_oralsolutions.yml  # Backfill parametrizado para las 11 fuentes
│   ├── flow_proyecto.yml           # Flujo maestro ELT (id: flow_proyecto)
│   ├── produccion_lotes.yml        # Inferencia batch nocturna contra la API
│   ├── monitoreo.yml               # Evaluación de deriva y alertas
│   └── reentrenamiento.yml         # Reentrenamiento con pausa de aprobación humana
├── infra/
│   ├── docker-compose.yml          # PostgreSQL, Kestra, Spark, dbt, MLflow, API y Tablero
│   ├── config/dbt/                 # Plantilla y perfil privado de dbt
│   └── sql/                        # Inicialización de Snowflake
├── modelado/
│   ├── preparar_obt.py             # Extrae OBT de Snowflake, calcula abandono_60d y guarda CSV limpio
│   ├── explorar_proyecto.ipynb     # EDA y 3 hallazgos que guían el modelado
│   ├── features_proyecto.ipynb     # Diseño y prueba sin fuga de 4 features derivadas
│   ├── pipeline_proyecto.py        # División temporal, features y preprocesador reutilizable
│   ├── modelos_proyecto.ipynb      # Líneas base, candidatos y resultados_proyecto.csv
│   ├── comparar_proyecto.ipynb     # Ensamblajes vs. red neuronal y prueba de LLM
│   ├── experimentos_proyecto.ipynb # Tuning, tracking MLflow y corridas_proyecto.csv
│   ├── validacion_proyecto.ipynb   # Validación cruzada, costos y calibración de umbral
│   ├── resultados_proyecto.csv     # Tabla comparativa de modelos candidatos
│   ├── corridas_proyecto.csv       # Exportación limpia de las 101 corridas de MLflow
│   ├── test_pipeline_proyecto.py   # Pruebas unitarias reproducibles del pipeline
│   └── reentrenar.py               # Reentrenamiento controlado y exportación MLflow
├── servicio/
│   ├── Dockerfile                  # Imagen de producción de la API FastAPI
│   ├── requirements.txt            # Dependencias fijadas del servicio
│   ├── api.py                      # Endpoints /salud, /modelo, /predecir, /metricas, /alertas
│   ├── esquemas.py                 # Contratos Pydantic de entrada/salida
│   ├── test_api.py                 # Pruebas de contrato, modelo, negocio y respaldo
│   └── fixtures/                   # Modelo @champion y muestra fija para CI
├── monitoreo/
│   ├── deriva.py                   # Calidad de datos, PSI, KS, Chi² y deriva conceptual
│   ├── alertas.py                  # Reglas de alerta calibradas por simulación
│   ├── simular_produccion.py       # Simulación L01-L08 + L09 datos + L10 conceptual
│   └── tablero.py                  # Tablero interactivo en Streamlit
└── docs/                           # Memos, decision_modelo.md, despliegue.md y CSV de evidencias
```

> **Nota de estructura:** El proyecto sigue la estructura oficial del repositorio del curso (``). Su equivalencia con la plantilla del enunciado PDF es: `docker-compose.yml` (`docker-compose.yml`), `kestra/` (`kestra/`), `dbt/` (`dbt/`), `procesamiento/` (`spark/`), `modelado/` (`src/` y `notebooks/`), `servicio/` (`api/` y `tests/`), `monitoreo/` (`monitoreo/`) y `docs/` (`docs/`).

### 2. Requisitos previos y entorno Python
- **Docker Desktop** con Docker Compose y contenedores Linux. Los comandos siguientes están escritos para **PowerShell en Windows**.
- **Python 3.11+** con un entorno virtual dedicado donde se instalen las librerías de requirements.txt (numpy, pandas, python-dotenv, requests, gdown, duckdb, pytest):

```PowerShell
# Crear y activar entorno virtual en PowerShell (desde )
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux / GitHub Codespaces:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- **Cuenta de Snowflake** con permisos `ACCOUNTADMIN` y un método de autenticación compatible con conexiones **usuario/contraseña**.
- Conexión a Internet para descargar los datos desde Google Drive, conectar con Snowflake y descargar las imágenes/dependencias de Docker y Spark.
- Sin descargas manuales: los 11 archivos CSV se descargan y verifican automáticamente en data/raw/ al ejecutar python src/ingestar.py o directamente desde el flujo de Kestra (oralsolutions.flow_proyecto).

> Salvo los scripts SQL (que se ejecutan en **Snowsight**), todos los comandos se ejecutan desde **``**.

#### Configuración requerida en Snowflake
La implementación actual utiliza estos nombres en los scripts SQL y en los flujos YAML; **cambiar las variables `SNOWFLAKE_*` de `.env` no cambia automáticamente los valores escritos en los flujos de Kestra**.

|**Recurso**|**Nombre o requisito utilizado**|
|---|---|
|Base de datos|`ORALSOLUTIONS_DW`|
|Warehouse|`COMPUTE_WH`|
|Rol de los flujos JDBC de Kestra|`ACCOUNTADMIN`|
|Esquemas|`RAW`, `BRONZE`, `SILVER`, `GOLD`, `OBT`|
|Autenticación actual|Usuario y contraseña admitidos por la cuenta|

En una cuenta nueva, utilizar estos nombres y los permisos correspondientes. Si se desea utilizar otro rol, warehouse o nombre de base de datos, habrá que modificar **de manera coherente los YAML, scripts SQL, perfil dbt y configuración de Spark**. La inicialización SQL crea el warehouse, la base, los esquemas, las tablas Raw, el formato CSV y el stage, pero **no aprovisiona el usuario**.

### 3. Obtener los datos y configurar credenciales privadas

1. **Descargar y verificar los 11 archivos CSV en `data/raw/`:**

```PowerShell
python src/ingestar.py
```

_(Confirma que los 11 archivos y sus **38,603 registros** estén disponibles en `data/raw/`)._

2. **Crear los archivos locales de configuración a partir de las plantillas públicas (ignorados por `.gitignore`):**

```PowerShell
Copy-Item .env.example .env
Copy-Item config/dbt/profiles.example.yml config/dbt/profiles.yml
```

En Linux / GitHub Codespaces:

```bash
cp .env.example .env
cp config/dbt/profiles.example.yml config/dbt/profiles.yml
```

- **En `.env`:** completar `KESTRA_DB_PASSWORD`, `KESTRA_ADMIN_EMAIL`, `KESTRA_ADMIN_PASSWORD`, `ENV_SNOWFLAKE_URL`, `ENV_SNOWFLAKE_USER`, `SF_ACCOUNT`, `SF_USER`, `SF_WAREHOUSE=COMPUTE_WH`, `SF_DATABASE=ORALSOLUTIONS_DW`, `SF_GOLD_SCHEMA=GOLD`, `SF_OBT_SCHEMA=OBT`, `SF_ROLE=ACCOUNTADMIN`, `SECRET_SNOWFLAKE_PASSWORD` (contraseña en Base64 UTF-8) y `ENV_KESTRA_HOST_PROJECT_PATH` (ruta absoluta de `` visible para Docker, guardando `.env` en UTF-8 sin BOM con finales de línea `LF`).

```PowerShell
# Generar SECRET_SNOWFLAKE_PASSWORD en Base64:
python -c "import base64,getpass; print(base64.b64encode(getpass.getpass('Contraseña Snowflake: ').encode('utf-8')).decode('ascii'))"
```

- **En `config/dbt/profiles.yml`:** completar `account`, `user`, `password` (texto plano, sin Base64), `role: ACCOUNTADMIN` y `warehouse: COMPUTE_WH` manteniendo el perfil `oralsolutions`, target `dev` y base `ORALSOLUTIONS_DW`.

### 4. Cadena local de calidad y warehouse (DuckDB)
1. **Limpieza y validación de calidad centralizada en dbt (capa Silver):**
Las reglas documentadas en `calidad/diagnostico_proyecto.ipynb` viven exclusivamente en `dbt/models/silver/`. Una vez levantados los contenedores de la sección 5, pueden ejecutarse directamente con:
```PowerShell
  python calidad/limpiar.py
```
(Invoca dbt run --select silver y dbt test --select silver sin duplicar lógica de limpieza en Python).

2. **Carga al warehouse local DuckDB y verificación de filas:**

```PowerShell
python warehouse/construir_warehouse.py
```

(Crea `warehouse/oralsolutions.duckdb` verificando que las **38,603 filas** cargadas coincidan exactamente con los CSV limpios).
*(Nota: En entornos limpios como un Codespace nuevo, la primera ejecución del paso de Spark tarda ~9–10 minutos mientras descarga los conectores Maven/Ivy de Snowflake; en ejecuciones posteriores reutiliza el volumen caché `spark_ivy`).*

### 5. Pipeline End-to-End en Docker + Snowflake (Kestra → dbt → Spark)
1. **Inicializar Snowflake (en Snowsight):** ejecutar en orden:
    - `sql/01_setup_snowflake.sql`: crea el warehouse `COMPUTE_WH`, la base `ORALSOLUTIONS_DW` y los esquemas `RAW`, `BRONZE`, `SILVER`, `GOLD` y `OBT`.
    - `sql/02_create_raw_tables.sql`: crea las 11 tablas en `RAW`, el formato `CSV_RAW_FORMAT` y el stage `RAW_CSV_STAGE`.

2. **Construir imagen de dbt y levantar infraestructura:**

```PowerShell
docker build -f Dockerfile.dbt -t pset_2-dbt:latest .
docker compose --env-file .env --project-name pset_2 config --quiet
docker compose --env-file .env --project-name pset_2 up -d postgres kestra spark dbt
docker compose --env-file .env --project-name pset_2 exec dbt dbt debug
```

3. **Importar flujos en Kestra (`http://localhost:8080`):** crear y guardar los YAML de `kestra/` en el namespace `oralsolutions` en este orden:
    
    1. `kestra/ingest_raw_oralsolutions.yml`
    2. `kestra/test_dbt_docker.yml`
    3. `kestra/test_spark_docker.yml`
    4. `kestra/backfill_oralsolutions.yml` *(recuperación histórica parametrizada para las 11 fuentes)*
    5. `kestra/ingesta.yml` (**maestro**, `id: flow_proyecto`)

4. **Ejecutar el pipeline maestro:** abrir `oralsolutions.flow_proyecto` en Kestra y seleccionar **Execute**. Ejecuta secuencialmente la **descarga automática desde Google Drive** (si faltan archivos en `data/raw/`), la ingesta idempotente a `RAW` (con reintentos exponenciales `maxAttempts: 3`), las transformaciones `dbt run` y `dbt test` (`BRONZE`, `SILVER`, `GOLD`), y la construcción de `OBT.OBT_RIESGO_ABANDONO` mediante `spark/job_spark.py`.

### 6. Verificar resultados y operaciones opcionales
Resultados esperados en Snowflake: **11** tablas cargadas en `RAW` (**38,603** registros), **28** modelos dbt, **177** pruebas aprobadas, **6,794** eventos en `GOLD.FACT_SESIONES_TRATAMIENTO` y **3,435** puntos candidatos únicos en `OBT.OBT_RIESGO_ABANDONO`.

```PowerShell
# Ejecución manual de dbt (desde )
docker compose --env-file .env --project-name pset_2 exec dbt dbt run
docker compose --env-file .env --project-name pset_2 exec dbt dbt test

# Ejecución manual de Spark (desde )
docker compose --env-file .env --project-name pset_2 exec spark /opt/spark/bin/spark-submit --master 'local[*]' --packages 'net.snowflake:spark-snowflake_2.12:3.2.3-spark_3.5,net.snowflake:snowflake-jdbc:4.0.2' /opt/spark/work-dir/job_spark.py

# Detener servicios sin borrar volúmenes
docker compose --env-file .env --project-name pset_2 down
```

### Decisión de procesamiento: Batch vs. Streaming
Este proyecto utiliza procesamiento por lotes (batch) diario (0 2 * * * , America/Guayaquil) y no requiere streaming, debido a que la decisión de negocio que habilita (priorizar llamadas, mensajes de seguimiento y reprogramaciones preventivas para pacientes con riesgo de abandonar su tratamiento en los próximos 60 días) no pierde valor con datos de horas de antigüedad. Durante la jornada diurna y vespertina la clínica registra citas y sesiones de forma progresiva, por lo que consolidar los datos en la madrugada permite que el equipo administrativo disponga cada mañana de las alertas actualizadas sin interferir con la operación clínica ni asumir la complejidad operativa de una arquitectura continua de baja latencia.

## Cómo entrenar y evaluar el modelo

Todos los comandos de esta etapa se ejecutan desde **`modelado/`** con el entorno virtual activado:

1. **Extracción desde Snowflake y generación del dataset limpio (`src/datos.py`):**
   - El flujo de modelado parte de la tabla analítica `ORALSOLUTIONS_DW.OBT.OBT_RIESGO_ABANDONO` (3,435 sesiones candidatas) generada en Snowflake por el pipeline ELT.
   - El script `preparar_obt.py` se conecta a Snowflake con las credenciales definidas en `.env` (`SF_ACCOUNT`, `SF_USER`, `SECRET_SNOWFLAKE_PASSWORD`, `SF_WAREHOUSE`, `SF_DATABASE`, `SF_OBT_SCHEMA`, `SF_ROLE`), descarga la OBT, excluye las sesiones de los últimos 60 días (`> 2025-11-01`) para evitar censura a la derecha, calcula la variable objetivo binaria `abandono_60d` (eliminando columnas auxiliares futuras para evitar fuga) y guarda el archivo limpio `../data/processed/obt_riesgo_abandono.csv` (**2,377 sesiones**, 21 columnas, 0 nulos; huella `SHA-256 = 937af6f343e5de34e3e6b07cd211ebde6732221e73c53d2ec47d0e26a767cf60`):

```powershell
python src/datos.py
```

2. **Archivo de entrada y división de datos (`src/pipeline.py`):**
   - Lee `../data/processed/obt_riesgo_abandono.csv`, ordena cronológicamente por `(fecha_realizada, sesion_key)` y aplica una **división temporal única** con corte **`FECHA_CORTE = 2025-08-01`**: **`1,551`** sesiones históricas para entrenamiento (`65.25%`) y **`826`** sesiones recientes (`2025-08-01` a `2025-11-01`) como prueba *hold-out* (`34.75%`).
   - Construye las 4 variables derivadas sin fuga (`progreso_plan`, `ritmo_dias_por_sesion`, `friccion_previa` y `complejidad_proc`) y encapsula la imputación, `RobustScaler` y `OneHotEncoder(handle_unknown="ignore")` dentro de un `ColumnTransformer` / `Pipeline` ajustado **exclusivamente sobre entrenamiento** (23 columnas finales).
   - Puede verificarse de forma independiente en una terminal nueva con:

```powershell
python src/pipeline.py
```

3. **Levantar el servidor de tracking MLflow:** Antes de correr los experimentos, asegúrate de tener activo el servicio de **MLflow** (`http://127.0.0.1:5000`) definido en `docker-compose.yml`:

```PowerShell
# Desde 
docker compose --env-file .env --project-name pset_2 up -d mlflow
```

4. **Orden de ejecución de los notebooks (`modelado/`):** Todos los notebooks de modelado importan directamente `pipeline_proyecto.py` y deben ejecutarse con **Restart & Run All**:
    1. `explorar_proyecto.ipynb` — Análisis exploratorio (tamaño, faltantes, distribución de `abandono_60d`, atípicos IQR/Z-score, correlación de Spearman y 3 hallazgos con impacto en el modelo).
    2. `features_proyecto.ipynb` — Evidencia estadística y verificación temporal sin fuga de las 4 variables derivadas.
    3. `modelos_proyecto.ipynb` — Entrenamiento de 3 líneas base (`Dummy_Mayoritaria`, `Regla_Negocio_14d` y `Regla_1_Sesion`) y candidatos por defecto; exporta `resultados_proyecto.csv`.
    4. `comparar_proyecto.ipynb` — Contraste de ensamblajes contra red neuronal (`MLPClassifier`), diagnóstico de regularización y evaluación de LLM (_few-shot prompting_ sobre 30 filas reales).
    5. `experimentos_proyecto.ipynb` — Ajuste fino (`RandomizedSearchCV` con 30 configuraciones × 3 familias sobre `TimeSeriesSplit(5)`), registro de las 101 corridas limpias en MLflow (`OralSolutions`), registro en Model Registry (`oralsolutions-modelo`) y exportación de `corridas_proyecto.csv`.
    6. `validacion_proyecto.ipynb` — Recarga de `@champion` desde el Model Registry, comparación de esquemas de validación cruzada (`TimeSeriesSplit`, `GroupKFold` por paciente y `KFold` aleatorio), cálculo de costo de errores (`FN = $120 USD`, `FP = $3 USD`) y calibración del umbral óptimo.

## Modelo final
- **Modelo elegido (`@champion`):** Pipeline completo de scikit-learn con **`MLPClassifier`** (`alpha = 0.00338`, `learning_rate_init = 3.49e-4`, `hidden_layer_sizes = (128, 64)`, `max_iter = 400`, `random_state = 42`).
- **Identificador en MLflow (`run_id`):** `169e8821e2c54761bf602b21f86aec93` (Versión 7, alias `@champion`).
- **Model Registry:** `oralsolutions-modelo` — **Versión 7**, alias **`@champion`** (segundo mejor: `Tuning_LogisticRegression`, `run_id = 92446a9861994d22a0a7431b389f1eb1`, **Versión 8**, alias **`@challenger`** inicial; tras el ciclo de reentrenamiento controlado, la **Versión 9** (`run_id = ee684c683fc04b54845826bc76ce3b32`) quedó registrada como **`@challenger`**).
- **Métricas que lo sostienen:**
    - **Validación cruzada temporal (`TimeSeriesSplit(5)`):** `PR-AUC = 0.98330 ± 0.00342` (superando en `+0.32` puntos a las líneas base operativas y con menor varianza entre pliegues que el `@challenger` `0.98298 ± 0.00461`).
    - **Conjunto de prueba (_hold-out_ `>= 2025-08-01`, 826 sesiones):** **`PR-AUC = 0.98269`** (reproducido exactamente al recargar `models:/oralsolutions-modelo@champion`), con tiempo de entrenamiento de `3.48 s` e inferencia `< 6 ms` para todo el trimestre.
    - **Impacto financiero con umbral calibrado (`u = 0.01`):** `Recall = 98.64 %`, `Precisión = 69.21 %`, reduciendo el costo de errores a **`$1,518 USD`** (`-72.1 %` frente al umbral inicial `0.60`) y alcanzando un balance neto de **`$18,294 USD`**.
- **Documentación completa de la decisión:** `docs/decision_modelo.md` y `docs/PSet3_memo_Tiban.pdf`.

## Cómo correr el servicio

El modelo ganador (`oralsolutions-modelo` **versión 7**, alias `@champion`, `run_id = 169e8821e2c54761bf602b21f86aec93`, umbral calibrado `u = 0.01`) se sirve mediante una API en **FastAPI** contenedorizada (`api/Dockerfile`) conectada al Model Registry de **MLflow** (`http://mlflow:5000`). El patrón de despliegue es **batch diario nocturno (`02:00`, `America/Guayaquil`)** orquestado por Kestra (`kestra/produccion_lotes.yml`). La documentación completa de despliegue, rollback y métricas está en `docs/despliegue.md`.

### 1. Levantar MLflow y el servicio de predicción (`api`)

Desde ``:

```powershell
# Construir y levantar MLflow + API en Docker
docker compose --env-file .env --project-name pset_2 up -d --build mlflow api tablero

# Verificar estado del servicio y versión activa desde el Model Registry
Invoke-RestMethod http://127.0.0.1:8000/salud
Invoke-RestMethod http://127.0.0.1:8000/modelo
```

_(En Linux / GitHub Codespaces, reemplaza `Invoke-RestMethod` por `curl -s http://127.0.0.1:8000/salud` y `curl -s http://127.0.0.1:8000/modelo`)._

### 2. Ejemplo de petición y respuesta (`POST /predecir`)

```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "duracion_estimada_min": 75,
    "numero_sesion": 1,
    "sesiones_planificadas": 3,
    "cancelaciones_previas": 0,
    "inasistencias_previas": 0,
    "dias_desde_inicio_plan": 14,
    "edad_en_atencion": 42,
    "codigo_procedimiento": "CROWN",
    "odontologo_key": "2"
  }'
```

Respuesta esperada (`200 OK`):
```json
{
  "id_prediccion": "a3f9c2e1-...",
  "probabilidad": 0.00412,
  "decision": "no_contactar",
  "accion": "Mantener seguimiento administrativo estándar",
  "umbral_aplicado": 0.01,
  "version_modelo": "7",
  "run_id": "169e8821e2c54761bf602b21f86aec93",
  "uso_respaldo": false,
  "latencia_ms": 6.98,
  "timestamp": "2026-11-19T16:15:00+00:00"
}
```

### 3. Pruebas automáticas, monitoreo, tablero y reentrenamiento

Desde `` (con `.venv` activo e instaladas las dependencias de `requirements.txt`):

```powershell
# 1. Ejecutar las 14 pruebas automáticas (pipeline + contrato, modelo, negocio y respaldo de API)
pytest -v

# 2. Simular producción por lotes (L01-L08 reales + L09 deriva de datos + L10 deriva conceptual)
python monitoreo/simular_produccion.py

# 3. Levantar el tablero interactivo de monitoreo (http://localhost:8501)
streamlit run monitoreo/tablero.py

# 4. Ejecutar ciclo de reentrenamiento controlado contra MLflow
python src/reentrenar.py --decision rechazar --motivo-decision "Revision humana tras deriva conceptual en L10"
```

Los flujos automatizados de inferencia batch, monitoreo y reentrenamiento con pausa de aprobación humana se encuentran en `kestra/produccion_lotes.yml`, `kestra/monitoreo.yml` y `kestra/reentrenamiento.yml`. El CI/CD está automatizado en `.github/workflows/ci.yml` y `.github/workflows/cd.yml` (publicando la imagen en GHCR en cada _push_ a `master`).
