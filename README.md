OralSolutions — Predicción de abandono de tratamientos odontológicos
Proyecto de ciencia de datos para preparar, analizar y modelar información clínica, operativa y financiera de una clínica odontológica simulada, OralSolutions. Su objetivo es estimar el riesgo de que un paciente no continúe su tratamiento dentro de los 60 días posteriores a una cita atendida, para apoyar la priorización de contactos preventivos y reprogramaciones.
Objetivo de negocio: incrementar en 10 puntos porcentuales la tasa de continuidad de tratamientos a 60 días respecto al seguimiento reactivo habitual. Esta es una meta del proyecto, no un resultado observado. Como política operativa propuesta, se contempla activar alertas cuando la probabilidad estimada de abandono sea al menos 1 % (`umbral = 0.01`); su eficacia y costo deben evaluarse con los resultados del modelo.
> **Alcance de esta sección:** se documentan la fuente de datos, el entorno, el pipeline ELT del PSet2 y su integración en la estructura definitiva `proyecto_final/`. La preparación de etiquetas, el modelado predictivo y la puesta en producción corresponden a las fases posteriores.
Datos
El dataset sintético de OralSolutions comprende 11 archivos CSV y 38.603 registros de origen. Se creó para simular operaciones odontológicas y permitir el análisis de calidad, la transformación y el modelado sin emplear historias clínicas reales.
Archivo	Registros	Contenido principal
`appointments.csv`	7.788	Citas odontológicas
`dentists.csv`	5	Odontólogos
`insurers.csv`	4	Aseguradoras
`invoices.csv`	7.788	Facturación
`missed_appointments.csv`	1.500	Cancelaciones e inasistencias
`patient_followups.csv`	1.243	Seguimiento y contacto con pacientes
`patients.csv`	2.500	Pacientes
`procedures.csv`	10	Catálogo de procedimientos
`treatment_lines.csv`	7.956	Líneas de tratamiento/procedimientos
`treatment_plans.csv`	3.015	Planes de tratamiento
`treatment_sessions.csv`	6.794	Eventos e intentos de sesiones
Total	38.603	11 fuentes
Ubicación local: `data/raw/*.csv`. En la configuración actual, los archivos de datos están excluidos del repositorio mediante `.gitignore`.
Obtención de los CSV
Desde la raíz de `proyecto_final/`:
```powershell
python src/ingestar.py
```
El script comprueba la presencia de los 11 archivos; si falta alguno, intenta descargarlos desde la carpeta compartida de Google Drive. Si están todos presentes, informa del número de archivos y del total de filas calculado a partir de los CSV. Este conteo no sustituye una auditoría de integridad del contenido.
Como alternativa, descargar manualmente los archivos desde la carpeta de Google Drive de OralSolutions, colocarlos en `data/raw/` y volver a ejecutar el script.
La ingesta también se invoca desde Kestra mediante el subflujo `kestra/ingest_raw_oralsolutions.yml`.
Cómo correr el pipeline de datos
La arquitectura ELT comprobada es:
```text
11 CSV → Kestra → Snowflake RAW → dbt Bronze → dbt Silver
                                      → dbt Gold → Apache Spark → Snowflake OBT
```
Kestra: orquesta la ingesta, las transformaciones, las pruebas y la generación de la OBT.
Snowflake RAW: almacena las 11 fuentes cargadas.
dbt Core: construye Bronze (tipificación), Silver (depuración y normalización) y Gold (esquema dimensional), además de ejecutar pruebas.
Apache Spark: lee Gold, verifica las claves y los joins, calcula variables históricas y escribe `OBT.OBT_RIESGO_ABANDONO`.
Procesamiento batch: el flujo maestro incorpora un horario diario a las 02:00 (`America/Guayaquil`); la ejecución manual completa ya se verificó, mientras que la ejecución automática por horario todavía debe comprobarse de forma separada.
1. Estructura del proyecto
La estructura actual utiliza la raíz `proyecto_final/`, sin las carpetas antiguas `infra/`, `warehouse/`, `procesamiento/` o `modelado/` mencionadas en versiones previas del README.
```text
proyecto_final/
├── README.md
├── docker-compose.yml             # Servicios Docker del proyecto
├── Dockerfile.dbt                 # Imagen de dbt con adaptador Snowflake
├── requirements.txt              # Dependencias Python locales
├── .env.example                  # Plantilla pública de variables
├── .env                          # Credenciales locales: NO versionar
├── .gitignore
├── config/
│   └── dbt/
│       ├── profiles.example.yml
│       └── profiles.yml          # Perfil privado de Snowflake: NO versionar
├── data/
│   └── raw/                      # 11 CSV, excluidos de Git
├── dbt/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── bronze/               # 11 modelos y _sources.yml
│   │   ├── silver/               # 11 modelos y _silver.yml
│   │   └── gold/                 # 6 modelos y schema.yml
│   ├── macros/
│   │   └── generate_schema_name.sql
│   └── tests/                    # Pruebas SQL, si corresponden
├── kestra/
│   ├── ingesta.yml               # Flujo maestro; id interno: flow_proyecto
│   ├── ingest_raw_oralsolutions.yml
│   ├── backfill_oralsolutions.yml
│   ├── test_dbt_docker.yml
│   ├── test_spark_docker.yml
│   ├── produccion_lotes.yml
│   ├── monitoreo.yml
│   └── reentrenamiento.yml
├── spark/
│   ├── job_spark.py
│   └── verificar_conexion.py
├── src/
│   ├── ingestar.py
│   ├── datos.py
│   ├── pipeline.py
│   └── reentrenar.py
└── docs/                        # Documentación y evidencias
```
> El árbol muestra los archivos y directorios relevantes para reproducir el **pipeline de datos**. La aplicación del PSet4 puede incluir, además, otros módulos, servicios, notebooks, pruebas y artefactos que deben documentarse en las secciones correspondientes. El archivo **`kestra/ingesta.yml` no se renombra**: su identificador dentro de Kestra es `oralsolutions.flow_proyecto`.
2. Requisitos previos y entorno Python
Docker Desktop con Docker Compose y contenedores Linux.
Python 3.11 o compatible con las dependencias de `requirements.txt`.
Cuenta de Snowflake con permisos suficientes para crear o utilizar el warehouse, la base de datos, los esquemas, las tablas y el stage. El entorno comprobado utiliza el rol `ACCOUNTADMIN`; para un despliegue real se recomienda un rol de privilegios mínimos.
Internet para Google Drive, Snowflake, imágenes Docker y paquetes Maven utilizados por Spark.
Los siguientes comandos están pensados para PowerShell en Windows, ejecutados desde `proyecto_final/`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
En Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Preparar Snowflake y configurar credenciales
Recursos utilizados en la implementación verificada:
Recurso	Valor
Base de datos	`ORALSOLUTIONS_DW`
Warehouse	`COMPUTE_WH`
Esquemas	`RAW`, `BRONZE`, `SILVER`, `GOLD`, `OBT`
Rol usado en las pruebas	`ACCOUNTADMIN`
Stage CSV	`RAW.RAW_CSV_STAGE`
Formato CSV	`RAW.CSV_RAW_FORMAT`
Inicialización en Snowflake: antes de lanzar una carga en una cuenta vacía, se deben ejecutar los scripts SQL de preparación del entorno (creación de base de datos y esquemas, 11 tablas RAW, formato de CSV y stage). Verificar la ubicación exacta de esos scripts en la copia definitiva del proyecto; las rutas antiguas `infra/sql/` o `sql/` no deben suponerse válidas sin comprobarlas. La inicialización no crea necesariamente un usuario Snowflake.
Crear los archivos privados a partir de sus plantillas:
```powershell
Copy-Item .env.example .env
Copy-Item config/dbt/profiles.example.yml config/dbt/profiles.yml
```
Configurar los valores requeridos por Docker, Kestra y Spark. Entre ellos:
Kestra / Docker: `KESTRA_DB_PASSWORD`, `KESTRA_ADMIN_EMAIL`, `KESTRA_ADMIN_PASSWORD`, `ENV_SNOWFLAKE_URL`, `ENV_SNOWFLAKE_USER`, `SECRET_SNOWFLAKE_PASSWORD` y `ENV_KESTRA_HOST_PROJECT_PATH`.
Spark: `SF_ACCOUNT`, `SF_USER`, `SF_DATABASE=ORALSOLUTIONS_DW`, `SF_WAREHOUSE=COMPUTE_WH`, `SF_ROLE=ACCOUNTADMIN`, `SF_GOLD_SCHEMA=GOLD`, `SF_OBT_SCHEMA=OBT` y `SECRET_SNOWFLAKE_PASSWORD`.
dbt: en `config/dbt/profiles.yml`, mantener el perfil `oralsolutions` y configurar cuenta, usuario, contraseña, warehouse, rol, base de datos y target `dev` según la plantilla local.
La variable `SECRET_SNOWFLAKE_PASSWORD` utilizada por el trabajo de Spark y los flujos correspondientes contiene la contraseña codificada en Base64 UTF-8. Base64 no es cifrado y no debe considerarse un mecanismo de protección. Para generar el valor:
```powershell
python -c "import base64,getpass; print(base64.b64encode(getpass.getpass('Contraseña Snowflake: ').encode('utf-8')).decode('ascii'))"
```
Importante para Windows: guardar `.env` en UTF-8 con saltos de línea LF, porque el flujo actual de Spark lo carga desde Linux mediante `. /config/.env`. Este ajuste solucionó el error de lectura observado con CRLF. El archivo debe contener asignaciones compatibles con shell. No publicar `.env`, `profiles.yml`, contraseñas ni tokens.
Ruta de montaje de Kestra: `ENV_KESTRA_HOST_PROJECT_PATH` debe señalar la ruta absoluta del proyecto visible para el motor Docker; en Docker Desktop para Windows puede ser diferente de `C:\...`. Esta ruta se usa para montar `src/`, `data/raw/`, `dbt/`, `config/dbt/`, `spark/` y `.env` en las tareas temporales de Kestra.
4. Construir y levantar la infraestructura
Desde `proyecto_final/`:
```powershell
# Validar la configuración de Docker Compose
docker compose -p proyecto_final config --quiet

# Construir las imágenes personalizadas (incluida dbt)
docker compose -p proyecto_final build

# Iniciar los servicios necesarios para el pipeline de datos
docker compose -p proyecto_final up -d postgres kestra dbt spark

# Comprobar los contenedores
docker compose -p proyecto_final ps -a

# Verificar la conexión de dbt con Snowflake
docker compose -p proyecto_final exec dbt dbt debug
```
La imagen utilizada por el subflujo dbt debe ser `proyecto_final-dbt:latest`, no la antigua `pset_2-dbt:latest`. El subflujo Spark utiliza `apache/spark:3.5.3`.
5. Importar y ejecutar los flujos en Kestra
Abrir http://localhost:8080. Registrar o actualizar los YAML del directorio `kestra/` en el namespace `oralsolutions`:
`ingest_raw_oralsolutions.yml`: descarga/comprobación de archivos e ingesta idempotente a RAW, con procesamiento de 11 CSV.
`test_dbt_docker.yml`: `dbt run` seguido de `dbt test`.
`test_spark_docker.yml`: lectura de Gold, construcción y escritura de la OBT.
`backfill_oralsolutions.yml`: recuperación histórica parametrizada.
`ingesta.yml`: flujo maestro, con `id: flow_proyecto`.
Luego abrir `oralsolutions.flow_proyecto` y pulsar Execute. El flujo maestro invoca los tres subflujos principales en este orden:
```text
ingesta_raw                 → oralsolutions.ingest_raw_oralsolutions
transformar_y_validar_dbt   → oralsolutions.test_dbt_docker
construir_obt_spark         → oralsolutions.test_spark_docker
```
El maestro utiliza `wait: true` y `transmitFailed: true`: espera a que termine cada subflujo y propaga sus fallos. La tarea de ingesta contiene reintentos exponenciales configurados con `maxAttempts: 3` para la descarga de fuentes.
El maestro incluye un trigger diario:
```yaml
cron: "0 2 * * *"
timezone: America/Guayaquil
```
Se ha verificado la ejecución manual completa; la activación efectiva del trigger programado aún requiere una prueba independiente.
6. Resultados verificados y operaciones opcionales
En la ejecución del 23 de noviembre de 2026, el pipeline maestro `oralsolutions.flow_proyecto` terminó en Success, con sus 3 subflujos completados en 6 min 2,10 s. Los componentes se probaron además por separado.
Componente	Resultado comprobado
Fuentes RAW	11 tablas, 38.603 registros
Verificación de identificadores RAW	`COUNT(*) = COUNT(DISTINCT ID)` en las 11 tablas tras repetir la ingesta
dbt Bronze	11 tablas
dbt Silver	11 tablas
dbt Gold	5 dimensiones + 1 tabla de hechos
Pruebas dbt	177 PASS, 0 WARN, 0 ERROR, 0 SKIP
`GOLD.FACT_SESIONES_TRATAMIENTO`	6.794 eventos
Spark — joins dimensionales	6.794 sesiones únicas conservadas
`OBT.OBT_RIESGO_ABANDONO`	3.435 puntos candidatos
Citas repetidas por paciente-plan-cita	0
Candidatos sin `CITA_ID`	0
Flujo maestro Kestra	Success, 3/3 etapas
La OBT se genera antes de definir la variable objetivo `abandono_60d`. El corte temporal de observabilidad de 60 días y la preparación del dataset para entrenamiento forman parte de una etapa posterior; por ello, 3.435 filas de la OBT no deben confundirse con las 2.377 observaciones elegibles documentadas para modelado.
El flujo de backfill también terminó en Success para las 11 fuentes en una ejecución parametrizada con fechas `2024-01-01` a `2025-12-31`. Esta prueba confirma que el proceso se completa sobre el histórico existente, pero no demuestra por sí sola la recuperación de filas ausentes. Esa prueba sigue pendiente en un entorno controlado.
Comandos de operación
```powershell
# Ejecutar dbt manualmente dentro de su contenedor
docker compose -p proyecto_final exec dbt dbt run
docker compose -p proyecto_final exec dbt dbt test

# Ver estado de servicios
docker compose -p proyecto_final ps -a

# Detener los contenedores sin eliminar volúmenes
docker compose -p proyecto_final down
```
Para reconstruir la OBT se recomienda ejecutar `oralsolutions.test_spark_docker` desde Kestra, ya que esta tarea configura expresamente las variables de entorno, los conectores Maven y el volumen con `spark/job_spark.py`. No se recomienda ejecutar un `spark-submit` aislado sin reproducir esa configuración.
Validación rápida de la OBT en Snowsight
```sql
SELECT
    COUNT(*) AS TOTAL_REGISTROS,
    COUNT(DISTINCT SESION_KEY) AS SESIONES_UNICAS,
    COUNT_IF(SESION_KEY IS NULL) AS CLAVES_NULAS
FROM ORALSOLUTIONS_DW.OBT.OBT_RIESGO_ABANDONO;
```
Para el dataset utilizado, se esperan 3.435 registros, 3.435 sesiones únicas y 0 claves nulas.
Decisión de procesamiento: Batch vs. Streaming
Se seleccionó procesamiento por lotes (batch) diario, con horario propuesto de 02:00 en `America/Guayaquil`, porque el objetivo de negocio consiste en priorizar contactos, recordatorios y reprogramaciones ante el posible abandono durante un horizonte de 60 días. La decisión no requiere una respuesta de milisegundos ni una infraestructura de streaming continuo. Consolidar los registros durante la madrugada facilita que el personal disponga de información actualizada al comenzar la jornada, con una complejidad operativa menor.






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
