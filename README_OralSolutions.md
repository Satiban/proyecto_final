# OralSolutions — Predicción de abandono de tratamientos odontológicos

Proyecto de ciencia de datos para preparar, analizar y modelar información clínica, operativa y financiera de una clínica odontológica simulada, **OralSolutions**. Su objetivo es estimar el riesgo de que un paciente no continúe su tratamiento dentro de los **60 días posteriores a una cita atendida**, para apoyar la priorización de contactos preventivos y reprogramaciones.

**Objetivo de negocio:** incrementar en **10 puntos porcentuales** la tasa de continuidad de tratamientos a 60 días respecto al seguimiento reactivo habitual. Esta es una **meta del proyecto**, no un resultado observado. Como política operativa propuesta, se contempla activar alertas cuando la probabilidad estimada de abandono sea al menos **1 %** (`umbral = 0.01`); su eficacia y costo deben evaluarse con los resultados del modelo.

## Datos

El dataset **sintético** de OralSolutions comprende **11 archivos CSV y 38.603 registros de origen**. Se creó para simular operaciones odontológicas y permitir el análisis de calidad, la transformación y el modelado sin emplear historias clínicas reales.

| Archivo | Registros | Contenido principal |
|---|---:|---|
| `appointments.csv` | 7.788 | Citas odontológicas |
| `dentists.csv` | 5 | Odontólogos |
| `insurers.csv` | 4 | Aseguradoras |
| `invoices.csv` | 7.788 | Facturación |
| `missed_appointments.csv` | 1.500 | Cancelaciones e inasistencias |
| `patient_followups.csv` | 1.243 | Seguimiento y contacto con pacientes |
| `patients.csv` | 2.500 | Pacientes |
| `procedures.csv` | 10 | Catálogo de procedimientos |
| `treatment_lines.csv` | 7.956 | Líneas de tratamiento/procedimientos |
| `treatment_plans.csv` | 3.015 | Planes de tratamiento |
| `treatment_sessions.csv` | 6.794 | Eventos e intentos de sesiones |
| **Total** | **38.603** | **11 fuentes** |

**Ubicación local:** `data/raw/*.csv`. En la configuración actual, los archivos de datos están excluidos del repositorio mediante `.gitignore`.

### Obtención de los CSV

Desde la raíz de `proyecto_final/`:

```powershell
python src/ingestar.py
```

El script comprueba la presencia de los 11 archivos; si falta alguno, intenta descargarlos desde la carpeta compartida de Google Drive. Si están todos presentes, informa del número de archivos y del total de filas calculado a partir de los CSV. **Este conteo no sustituye una auditoría de integridad del contenido**.

Como alternativa, descargar manualmente los archivos desde la [carpeta de Google Drive de OralSolutions](https://drive.google.com/drive/folders/1bDd9OOZ1SPTLjnGQld5IIn-x4Gp8KQIh?usp=drive_link), colocarlos en `data/raw/` y volver a ejecutar el script.

La ingesta también se invoca desde Kestra mediante el subflujo `kestra/ingest_raw_oralsolutions.yml`.

## Cómo correr el pipeline de datos

La arquitectura ELT comprobada es:

```text
11 CSV → Kestra → Snowflake RAW → dbt Bronze → dbt Silver
                                      → dbt Gold → Apache Spark → Snowflake OBT
```

- **Kestra:** orquesta la ingesta, las transformaciones, las pruebas y la generación de la OBT.
- **Snowflake RAW:** almacena las 11 fuentes cargadas.
- **dbt Core:** construye Bronze (tipificación), Silver (depuración y normalización) y Gold (esquema dimensional), además de ejecutar pruebas.
- **Apache Spark:** lee Gold, verifica las claves y los joins, calcula variables históricas y escribe `OBT.OBT_RIESGO_ABANDONO`.
- **Procesamiento batch:** el flujo maestro incorpora un horario diario a las 02:00 (`America/Guayaquil`); la ejecución manual completa ya se verificó, mientras que la ejecución automática por horario todavía debe comprobarse de forma separada.

### 1. Estructura del proyecto

La estructura **actual** utiliza la raíz `proyecto_final/`, sin las carpetas antiguas `infra/`, `warehouse/`, `procesamiento/` o `modelado/` mencionadas en versiones previas del README.

```text
proyecto_final/
├── README.md
├── docker-compose.yml
├── Dockerfile.dbt
├── requirements.txt
├── .env.example
├── .env                         # Credenciales privadas; no versionar
├── .gitignore
├── .github/
│   └── workflows/              # Integración y despliegue continuo
├── config/
│   └── dbt/
│       ├── profiles.example.yml
│       └── profiles.yml        # Configuración privada
├── data/
│   ├── raw/                    # 11 archivos CSV de origen
│   └── processed/              # Dataset preparado para modelado
├── dbt/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── bronze/
│   │   ├── silver/
│   │   └── gold/
│   ├── macros/
│   └── tests/
├── kestra/
│   ├── ingesta.yml
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
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   └── 05_seleccion.ipynb
├── api/
│   └── main.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_modelo.py
│   ├── test_contrato.py
│   ├── test_comportamiento.py
│   └── fixtures/
│       ├── muestra_test.csv
│       ├── modelo_meta.json
│       └── modelo_champion/
├── calidad/
│   └── diagnostico_proyecto.ipynb
└── docs/
    ├── cambios.md
    ├── resultados_modelos.csv
    ├── resultados_tuning.csv
    ├── resultados_seleccion.csv
    └── corridas.csv
```

### 2. Requisitos previos y entorno Python

- Docker Desktop con Docker Compose y contenedores Linux.
- Python 3.11 o compatible con las dependencias de `requirements.txt`.
- Cuenta de Snowflake con permisos suficientes para crear o utilizar el warehouse, la base de datos, los esquemas, las tablas y el stage. El entorno comprobado utiliza el rol `ACCOUNTADMIN`; para un despliegue real se recomienda un rol de privilegios mínimos.
- Internet para Google Drive, Snowflake, imágenes Docker y paquetes Maven utilizados por Spark.

Los siguientes comandos están pensados para **PowerShell en Windows**, ejecutados desde `proyecto_final/`:

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

### 3. Preparar Snowflake y configurar credenciales

**Recursos utilizados en la implementación verificada:**

| Recurso | Valor |
|---|---|
| Base de datos | `ORALSOLUTIONS_DW` |
| Warehouse | `COMPUTE_WH` |
| Esquemas | `RAW`, `BRONZE`, `SILVER`, `GOLD`, `OBT` |
| Rol usado en las pruebas | `ACCOUNTADMIN` |
| Stage CSV | `RAW.RAW_CSV_STAGE` |
| Formato CSV | `RAW.CSV_RAW_FORMAT` |

**Inicialización en Snowflake:** antes de lanzar una carga en una cuenta vacía, se deben ejecutar los scripts SQL de preparación del entorno (creación de base de datos y esquemas, 11 tablas RAW, formato de CSV y stage). **Verificar la ubicación exacta de esos scripts en la copia definitiva del proyecto**; las rutas antiguas `infra/sql/` o `sql/` no deben suponerse válidas sin comprobarlas. La inicialización no crea necesariamente un usuario Snowflake.

Crear los archivos privados a partir de sus plantillas:

```powershell
Copy-Item .env.example .env
Copy-Item config/dbt/profiles.example.yml config/dbt/profiles.yml
```

Configurar los valores requeridos por Docker, Kestra y Spark. Entre ellos:

- **Kestra / Docker:** `KESTRA_DB_PASSWORD`, `KESTRA_ADMIN_EMAIL`, `KESTRA_ADMIN_PASSWORD`, `ENV_SNOWFLAKE_URL`, `ENV_SNOWFLAKE_USER`, `SECRET_SNOWFLAKE_PASSWORD` y `ENV_KESTRA_HOST_PROJECT_PATH`.
- **Spark:** `SF_ACCOUNT`, `SF_USER`, `SF_DATABASE=ORALSOLUTIONS_DW`, `SF_WAREHOUSE=COMPUTE_WH`, `SF_ROLE=ACCOUNTADMIN`, `SF_GOLD_SCHEMA=GOLD`, `SF_OBT_SCHEMA=OBT` y `SECRET_SNOWFLAKE_PASSWORD`.
- **dbt:** en `config/dbt/profiles.yml`, mantener el perfil `oralsolutions` y configurar cuenta, usuario, contraseña, warehouse, rol, base de datos y target `dev` según la plantilla local.

La variable `SECRET_SNOWFLAKE_PASSWORD` utilizada por el trabajo de Spark y los flujos correspondientes contiene la contraseña codificada en **Base64 UTF-8**. Base64 **no es cifrado** y no debe considerarse un mecanismo de protección. Para generar el valor:

```powershell
python -c "import base64,getpass; print(base64.b64encode(getpass.getpass('Contraseña Snowflake: ').encode('utf-8')).decode('ascii'))"
```

**Importante para Windows:** guardar `.env` en **UTF-8 con saltos de línea LF**, porque el flujo actual de Spark lo carga desde Linux mediante `. /config/.env`. Este ajuste solucionó el error de lectura observado con CRLF. El archivo debe contener asignaciones compatibles con shell. No publicar `.env`, `profiles.yml`, contraseñas ni tokens.

**Ruta de montaje de Kestra:** `ENV_KESTRA_HOST_PROJECT_PATH` debe señalar la ruta absoluta del proyecto **visible para el motor Docker**; en Docker Desktop para Windows puede ser diferente de `C:\...`. Esta ruta se usa para montar `src/`, `data/raw/`, `dbt/`, `config/dbt/`, `spark/` y `.env` en las tareas temporales de Kestra.

### 4. Construir y levantar la infraestructura

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

La imagen utilizada por el subflujo dbt debe ser **`proyecto_final-dbt:latest`**, no la antigua `pset_2-dbt:latest`. El subflujo Spark utiliza `apache/spark:3.5.3`.

### 5. Importar y ejecutar los flujos en Kestra

Abrir **http://localhost:8080**. Registrar o actualizar los YAML del directorio `kestra/` en el namespace `oralsolutions`:

1. `ingest_raw_oralsolutions.yml`: descarga/comprobación de archivos e ingesta idempotente a RAW, con procesamiento de 11 CSV.
2. `test_dbt_docker.yml`: `dbt run` seguido de `dbt test`.
3. `test_spark_docker.yml`: lectura de Gold, construcción y escritura de la OBT.
4. `backfill_oralsolutions.yml`: recuperación histórica parametrizada.
5. `ingesta.yml`: **flujo maestro**, con `id: flow_proyecto`.

Luego abrir **`oralsolutions.flow_proyecto`** y pulsar **Execute**. El flujo maestro invoca los tres subflujos principales en este orden:

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

Se ha verificado la **ejecución manual completa**; la activación efectiva del trigger programado aún requiere una prueba independiente.

### 6. Resultados verificados y operaciones opcionales

En la ejecución del **23 de noviembre de 2026**, el pipeline maestro `oralsolutions.flow_proyecto` terminó en **Success**, con sus **3 subflujos completados en 6 min 2,10 s**. Los componentes se probaron además por separado.

| Componente | Resultado comprobado |
|---|---|
| Fuentes RAW | 11 tablas, **38.603** registros |
| Verificación de identificadores RAW | `COUNT(*) = COUNT(DISTINCT ID)` en las 11 tablas tras repetir la ingesta |
| dbt Bronze | 11 tablas |
| dbt Silver | 11 tablas |
| dbt Gold | 5 dimensiones + 1 tabla de hechos |
| Pruebas dbt | **177 PASS, 0 WARN, 0 ERROR, 0 SKIP** |
| `GOLD.FACT_SESIONES_TRATAMIENTO` | **6.794** eventos |
| Spark — joins dimensionales | **6.794** sesiones únicas conservadas |
| `OBT.OBT_RIESGO_ABANDONO` | **3.435** puntos candidatos |
| Citas repetidas por paciente-plan-cita | **0** |
| Candidatos sin `CITA_ID` | **0** |
| Flujo maestro Kestra | **Success**, 3/3 etapas |

La **OBT se genera antes de definir la variable objetivo** `abandono_60d`. El corte temporal de observabilidad de 60 días y la preparación del dataset para entrenamiento forman parte de una etapa posterior; por ello, **3.435** filas de la OBT no deben confundirse con las **2.377** observaciones elegibles documentadas para modelado.

El flujo de backfill también terminó en **Success** para las 11 fuentes en una ejecución parametrizada con fechas `2024-01-01` a `2025-12-31`. Esta prueba confirma que el proceso se completa sobre el histórico existente, **pero no demuestra por sí sola la recuperación de filas ausentes**. Esa prueba sigue pendiente en un entorno controlado.

#### Comandos de operación

```powershell
# Ejecutar dbt manualmente dentro de su contenedor
docker compose -p proyecto_final exec dbt dbt run
docker compose -p proyecto_final exec dbt dbt test

# Ver estado de servicios
docker compose -p proyecto_final ps -a

# Detener los contenedores sin eliminar volúmenes
docker compose -p proyecto_final down
```

Para reconstruir la OBT se recomienda ejecutar **`oralsolutions.test_spark_docker`** desde Kestra, ya que esta tarea configura expresamente las variables de entorno, los conectores Maven y el volumen con `spark/job_spark.py`. No se recomienda ejecutar un `spark-submit` aislado sin reproducir esa configuración.

#### Validación rápida de la OBT en Snowsight

```sql
SELECT
    COUNT(*) AS TOTAL_REGISTROS,
    COUNT(DISTINCT SESION_KEY) AS SESIONES_UNICAS,
    COUNT_IF(SESION_KEY IS NULL) AS CLAVES_NULAS
FROM ORALSOLUTIONS_DW.OBT.OBT_RIESGO_ABANDONO;
```

Para el dataset utilizado, se esperan **3.435 registros**, **3.435 sesiones únicas** y **0 claves nulas**.

### Decisión de procesamiento: Batch vs. Streaming

Se seleccionó **procesamiento por lotes (batch) diario**, con horario propuesto de **02:00 en `America/Guayaquil`**, porque el objetivo de negocio consiste en priorizar contactos, recordatorios y reprogramaciones ante el posible abandono durante un horizonte de 60 días. La decisión no requiere una respuesta de milisegundos ni una infraestructura de streaming continuo. Consolidar los registros durante la madrugada facilita que el personal disponga de información actualizada al comenzar la jornada, con una complejidad operativa menor.



## Cómo entrenar y evaluar el modelo

Todos los comandos se ejecutan desde la raíz de `proyecto_final/`, con el entorno virtual activado. Esta etapa utiliza la OBT generada durante el PSet 2 y los componentes de modelado del PSet 3.

### 1. Extracción desde Snowflake y generación del dataset (`src/datos.py`)

El modelado parte de `ORALSOLUTIONS_DW.OBT.OBT_RIESGO_ABANDONO`, con **3.435 sesiones candidatas**, y consulta adicionalmente `GOLD.FACT_SESIONES_TRATAMIENTO` para identificar la continuidad real de cada tratamiento.

El script utiliza las credenciales Snowflake de `.env`, identifica sesiones completadas posteriores dentro del mismo tratamiento y construye el target `abandono_60d`:

- **0 (continuidad):** existe una sesión posterior completada dentro de 60 días.
- **1 (abandono):** no existe continuidad registrada dentro de 60 días.

Se excluyen las observaciones de los últimos 60 días del histórico (1.058 registros) para evitar censura a la derecha, así como las fechas inconsistentes. Las columnas auxiliares futuras se eliminan antes de guardar el dataset.

```powershell
python -m src.datos
```

**Resultado verificado:** `data/processed/obt_riesgo_abandono.csv`, con **2.377 observaciones, 21 columnas, 321 abandonos (13,50 %) y 2.056 continuidades (86,50 %)**.

Huella del dataset:

```text
SHA-256:
11ae3cc53264511ef486f12d7a3fc0ddf2c6311e3ae7a8076f3c0cf046396dc1
```

El script guarda directamente el CSV y calcula su huella SHA-256, sin generar respaldos automáticos. Si se modifica el dataset, deberán repetirse los experimentos y actualizarse las evidencias.

### 2. Preparación y división temporal (`src/pipeline.py`)

El pipeline carga el CSV, ordena las observaciones por `(fecha_realizada, sesion_key)` y aplica una división temporal con `FECHA_CORTE = 2025-08-01` y horizonte de maduración de 60 días.

| Conjunto | Período | Observaciones | Abandono |
|---|---|---:|---:|
| Entrenamiento | Hasta 2025-06-02 | 1.195 | 13,64 % |
| Gap excluido | 2025-06-03 a 2025-07-31 | 356 | 11,24 % |
| Hold-out | Desde 2025-08-01 | 826 | 14,29 % |

La separación temporal evita utilizar etiquetas que aún no habrían madurado al comenzar el período de evaluación.

Se construyen cuatro características derivadas:

- `progreso_plan`: `numero_sesion / sesiones_planificadas`.
- `ritmo_dias_por_sesion`: `dias_desde_inicio_plan / numero_sesion`.
- `friccion_previa`: cancelaciones e inasistencias acumuladas.
- `complejidad_proc`: agrupación del procedimiento odontológico.

El modelo utiliza **13 predictores** (10 numéricos y 3 categóricos). Un `ColumnTransformer` integrado en un `Pipeline` aplica imputación por mediana y `RobustScaler` a numéricas, y moda más `OneHotEncoder` a categóricas, produciendo **23 columnas transformadas**. El preprocesador se ajusta exclusivamente con datos de entrenamiento.

Verificar independientemente:

```powershell
python -m src.pipeline
```

**Resultado verificado:** 1.195 observaciones de entrenamiento, 826 de hold-out, 356 excluidas y matrices transformadas de `(1195, 23)` y `(826, 23)`.

### 3. Levantar MLflow

La experimentación utiliza MLflow 2.19.0 mediante Docker Compose, con almacenamiento persistente en el volumen `proyecto_final_datos_mlflow`.

Desde `proyecto_final/`:

```powershell
docker compose -p proyecto_final up -d mlflow

docker compose -p proyecto_final ps mlflow

docker compose -p proyecto_final logs --tail=30 mlflow
```

Interfaz web: **http://localhost:5000**

Los notebooks utilizan el experimento `OralSolutions` y registran parámetros, métricas, etiquetas, artefactos y modelos. No es necesario eliminar las corridas anteriores para consultar o reutilizar los modelos existentes.

### 4. Orden de ejecución de los notebooks (`notebooks/`)

Seleccionar el kernel del entorno virtual `.venv` y ejecutar los notebooks en el siguiente orden. Todos reutilizan `src/pipeline.py`.

1. **`01_eda.ipynb` — Análisis exploratorio:** distribución de clases, calidad del dataset, recurrencia de pacientes, valores atípicos, correlaciones de Spearman y hallazgos relevantes para el modelado.

2. **`02_features.ipynb` — Ingeniería y evaluación de características:** analiza las cuatro variables derivadas mediante estadísticas descriptivas, correlaciones y pruebas de asociación. Incluye un estudio de ablación con LogisticRegression y cinco pliegues temporales purgados, comparando nueve variables originales frente a trece variables completas y cuatro configuraciones de eliminación individual. El modelo original obtuvo PR-AUC de 0,16795 frente a 0,16683 del modelo completo (diferencia de −0,00112), con mejoras en solo uno de cinco pliegues. No se demostró un aporte predictivo incremental conjunto de las variables derivadas para este algoritmo. La evaluación no utiliza el hold-out ni modifica MLflow.

3. **`03_baseline_candidatos.ipynb` — Líneas base y modelos candidatos:** compara tres líneas base (Dummy Mayoritaria, Regla de Negocio 14d y Regla 1 Sesión) y cinco candidatos (LogisticRegression, RandomForest, HistGradientBoosting, MLPClassifier y GaussianNB), mediante validación temporal purgada de cinco pliegues. Exporta `docs/resultados_modelos.csv`. Incluye una comparación exploratoria con un LLM local.

4. **`04_tuning.ipynb` — Optimización de hiperparámetros:** ejecuta `RandomizedSearchCV` sobre MLPClassifier, HistGradientBoosting y LogisticRegression, con 30 configuraciones por familia, cinco pliegues temporales purgados y semilla 42. Registra 93 corridas de tuning en MLflow y exporta `docs/resultados_tuning.csv`.

5. **`05_seleccion.ipynb` — Selección y evaluación final:** compara los finalistas mediante predicciones fuera de pliegue (OOF), selecciona el umbral por costo de errores dentro del rango operativo permitido, evalúa una sola vez el hold-out y registra `@champion` y `@challenger` en MLflow Model Registry. Exporta `docs/resultados_seleccion.csv` y `docs/corridas.csv`.

**Importante:** el hold-out de 826 observaciones permanece reservado durante la comparación y optimización de modelos. La evaluación final se realiza en el notebook 05. Las ejecuciones de los notebooks 03–05 crean corridas y pueden registrar nuevas versiones, por lo que repetirlos modificará el historial de MLflow.

**Interpretación del tuning:** Las configuraciones ajustadas incrementaron numéricamente la PR-AUC promedio de MLPClassifier (+0,0111), HistGradientBoosting (+0,0155) y LogisticRegression (+0,0161). Sin embargo, todas las diferencias fueron menores que las desviaciones estándar observadas entre los pliegues temporales. Por ello, el tuning se considera una exploración sistemática y reproducible de hiperparámetros, sin evidencia suficiente de mejoras robustas o estadísticamente significativas.

### 5. Evaluación exploratoria con Ollama (opcional)

El notebook `03_baseline_candidatos.ipynb` incluye una prueba real con el modelo local `qwen2.5:0.5b`. Esta comparación no interviene en la elección del champion.

Instalar Ollama desde su sitio oficial y descargar el modelo:

```powershell
ollama pull qwen2.5:0.5b

ollama list
```

Comprobar que el servicio local esté disponible en `http://localhost:11434` y activar `EJECUTAR_OLLAMA = True` en la sección correspondiente del notebook 03.

**Resultado experimental:** 20 observaciones de validación, PR-AUC de 0,1833 para Ollama frente a 0,2416 para MLPClassifier, con latencia media de 2,91 segundos por caso. La comparación es exploratoria y no sustituye la validación cruzada temporal.

### 6. Resultados, trazabilidad y pruebas

Los resultados se conservan en:

| Archivo | Contenido |
|---|---|
| `docs/resultados_modelos.csv` | Líneas base y candidatos por defecto |
| `docs/resultados_tuning.csv` | Configuraciones y métricas de tuning |
| `docs/resultados_seleccion.csv` | Comparación final de champion y challenger |
| `docs/corridas.csv` | Corridas, métricas e identificadores de MLflow |

La ejecución experimental definitiva registró **102 corridas activas**: 8 iniciales, 93 de tuning y una final.

Para verificar el pipeline y la API sin depender de Snowflake ni del servidor MLflow se incluye un champion exportado en `tests/fixtures/modelo_champion/`, acompañado de su metadata y de una muestra fija de 826 observaciones.

```powershell
python -m pytest tests -v
```

**Resultado local verificado:** 14 pruebas aprobadas, 0 fallidas y 3 advertencias de dependencias, utilizando Python 3.14.6.

Las pruebas cubren construcción de características, división temporal, preprocesamiento, contrato HTTP, reproducción de la PR-AUC, decisiones de negocio y funcionamiento del modo de respaldo.


## Modelo final

- **Champion:** Pipeline completo de scikit-learn con `HistGradientBoostingClassifier`, ajustado mediante `RandomizedSearchCV`.
- **Model Registry:** `oralsolutions-modelo`, **versión 1**, alias `@champion`.
- **Run ID:** `c90a1c37aef940278afc025b1d22dc6b`.
- **Challenger:** `MLPClassifier`, **versión 2**, alias `@challenger`.
- **Umbral operativo:** `0.01` (1 %), establecido como límite inferior para la selección por costos.

### Métricas de selección y evaluación

| Métrica | Champion (HGB) | Challenger (MLP) |
|---|---:|---:|
| PR-AUC CV temporal ajustada | 0,1852 ± 0,0597 | 0,1886 ± 0,0352 |
| Costo estimado OOF (USD) | 3.459 | 5.019 |
| PR-AUC hold-out | **0,159858** | 0,1666 |
| ROC-AUC hold-out | 0,5642 | 0,5619 |
| Recall hold-out | 100 % | 96,61 % |
| Precisión hold-out | 14,36 % | 15,00 % |
| Costo de errores hold-out (USD) | **2.112** | 2.418 |


**Criterio de selección:** HistGradientBoosting se seleccionó como champion experimental por registrar el menor costo de errores entre los modelos finalistas en validación temporal OOF: USD 3.459 frente a USD 5.019 de MLPClassifier. La comparación utilizó costos hipotéticos de USD 120 por falso negativo y USD 3 por falso positivo, con un umbral operativo mínimo de 0,01. Aunque MLPClassifier obtuvo una PR-AUC promedio ligeramente superior, las diferencias fueron pequeñas frente a la variabilidad entre pliegues.

**Comparación con políticas simples:** En validación OOF, contactar a todos presentó un costo estimado de USD 2.847, inferior al de ambos modelos. Por tanto, el champion fue el mejor de los candidatos entrenados, pero no superó la política más simple durante la selección.

En el hold-out de 826 observaciones, HistGradientBoosting obtuvo un costo de USD 2.112, frente a USD 2.124 de contactar a todos. El modelo emitió 822 alertas, evitando únicamente cuatro contactos. La ventaja de USD 12 fue mínima y contrastó con los resultados OOF.

**Conclusión económica:** No existe evidencia suficiente para afirmar que el champion supera de forma robusta a la política de contactar a todos. Los resultados dependen de costos hipotéticos, una tasa de rescate supuesta del 35 % y datos sintéticos. Se conserva el modelo para fines experimentales y de despliegue del prototipo, no como demostración de superioridad clínica o económica.

### Recuperación del modelo registrado

Con MLflow activo, el champion puede recuperarse mediante:

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

modelo = mlflow.sklearn.load_model(
    "models:/oralsolutions-modelo@champion"
)
```

El modelo registrado reproduce una **PR-AUC de 0,159858** sobre las 826 observaciones del hold-out corregido. Esta comprobación también se realiza mediante `tests/test_modelo.py` utilizando el artefacto local exportado.

La API del PSet 4 consume el alias `@champion` y aplica el umbral de decisión de 0,01. El modelo y sus métricas deben interpretarse considerando el origen sintético del dataset, la capacidad discriminativa limitada y la ausencia de validación causal de las intervenciones.

**Documentación y evidencias:** `docs/PSet3_memo_Tiban.pdf`, `docs/corridas.csv`, `docs/resultados_modelos.csv`, `docs/resultados_tuning.csv` y `docs/resultados_seleccion.csv`.