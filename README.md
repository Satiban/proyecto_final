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
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── esquemas.py
│   ├── exportar_modelo.py
│   └── main.py
├── monitoreo/
│   ├── alertas.py
│   ├── deriva.py
│   ├── simular_produccion.py
│   ├── evaluar_lotes.py
│   ├── monitorear_lote.py
│   ├── orquestar_monitoreo.py
│   └── tablero.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   ├── 05_seleccion.ipynb
│   ├── 06_simulacion_produccion.ipynb
│   └── 07_analisis_monitoreo.ipynb
├── scripts/
│   └── demostrar_rollback.py
├── sql/
│   └── mlops/
├── src/
│   ├── ingestar.py
│   ├── datos.py
│   ├── pipeline.py
│   └── reentrenar.py
└── docs/
    ├── historial/
    ├── arquitectura.png
    ├── modelo_dimensional.png
    ├── cambios.md
    ├── alertas.csv
    ├── corridas.csv
    ├── deriva_features.csv
    ├── equidad.csv
    ├── lotes_produccion.csv
    ├── monitoreo.csv
    ├── monitoreo_inmediato.csv
    ├── predicciones.csv
    ├── predicciones_demostracion.csv
    ├── reentrenamientos.csv
    ├── resultados_modelos.csv
    ├── resultados_seleccion.csv
    ├── resultados_tuning.csv
    └── solicitudes_http.csv
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

## Cómo correr el servicio

El modelo seleccionado durante el PSet 3 (`oralsolutions-modelo`, **versión 1**, alias `@champion`, `run_id = c90a1c37aef940278afc025b1d22dc6b`, umbral operativo `u = 0.01`) se sirve mediante una API desarrollada en **FastAPI**, contenedorizada mediante `api/Dockerfile` y conectada al Model Registry de **MLflow** (`http://mlflow:5000`).

La estrategia de despliegue seleccionada es el **procesamiento batch diario nocturno**, con horario operativo propuesto de **02:00 (`America/Guayaquil`)**, debido a que las predicciones apoyan la planificación de contactos preventivos y reprogramaciones, sin requerir respuestas en tiempo real.

FastAPI proporciona adicionalmente una interfaz de inferencia bajo demanda, utilizada por Kestra durante la simulación histórica y por el panel de recepción de Streamlit.

La simulación del PSet 4 utiliza lotes semanales históricos para reproducir la evolución temporal del sistema. El procesamiento diario corresponde a la estrategia operativa propuesta, mientras que los lotes semanales corresponden al diseño experimental verificado.

### 1. Preparar el almacenamiento de MLOps en Snowflake

Además de los esquemas utilizados por el pipeline ELT, el sistema utiliza el esquema `ORALSOLUTIONS_DW.MLOPS` para almacenar las predicciones, métricas, alertas y evidencias de producción.

Antes de ejecutar el servicio en un entorno nuevo, ejecutar en Snowsight los scripts SQL del directorio `sql/mlops/`, siguiendo su orden de numeración.

| Tabla | Responsabilidad |
|---|---|
| `PREDICCIONES` | Predicciones individuales, entradas, decisiones y versiones del modelo |
| `LOTES_PRODUCCION` | Control de ejecución y maduración de los lotes |
| `MONITOREO` | Métricas supervisadas y costos de negocio |
| `MONITOREO_INMEDIATO` | Métricas operativas, distribución y deriva |
| `DERIVA_FEATURES` | PSI, pruebas estadísticas y calidad por variable |
| `ALERTAS` | Alertas generadas durante el monitoreo |
| `SOLICITUDES_HTTP` | Telemetría HTTP, latencias y errores |
| `SNAP_SIM_2025_V1_OBT` | Snapshot de la OBT utilizada en la simulación |
| `SNAP_SIM_2025_V1_GOLD` | Snapshot de la tabla de hechos utilizada para construir el target |

Los snapshots se generan automáticamente durante la preparación de la simulación y permiten conservar las fuentes originales del experimento.

Configurar adicionalmente en `.env`:

```dotenv
SF_MLOPS_SCHEMA=MLOPS
ADMIN_TOKEN=<TOKEN_ADMINISTRATIVO_PRIVADO>
```

El token administrativo protege el mecanismo de activación y desactivación del modo de respaldo. No debe incluirse en el repositorio.

Puede generarse mediante:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Levantar MLflow, FastAPI y Streamlit

Desde la raíz de `proyecto_final/`, con las credenciales configuradas y el champion registrado previamente mediante el notebook `05_seleccion.ipynb`:

```powershell
# Levantar MLflow
docker compose -p proyecto_final up -d mlflow

# Levantar el servicio de predicción y el dashboard
docker compose -p proyecto_final up -d api tablero

# Comprobar el estado de los servicios
docker compose -p proyecto_final ps -a

# Revisar los logs de FastAPI
docker compose -p proyecto_final logs api --tail=50
```

Servicios disponibles:

| Servicio | URL local |
|---|---|
| Kestra | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

La API recupera el modelo mediante el alias:

```text
models:/oralsolutions-modelo@champion
```

Por tanto, el cambio de versión del modelo no requiere reconstruir la imagen Docker ni reiniciar el contenedor.

Para verificar el estado del servicio:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/salud" |
    ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8000/modelo" |
    ConvertTo-Json -Depth 5
```

Respuesta verificada de `/modelo`:

```json
{
    "nombre_modelo": "oralsolutions-modelo",
    "version": "1",
    "alias": "champion",
    "run_id": "c90a1c37aef940278afc025b1d22dc6b",
    "umbral_decision": 0.01,
    "modo_respaldo": false
}
```

En Linux o GitHub Codespaces se pueden utilizar:

```bash
curl -s http://localhost:8000/salud
curl -s http://localhost:8000/modelo
```

### 3. Ejemplo de petición y respuesta (`POST /predecir`)

La API recibe nueve variables originales y reutiliza la lógica de ingeniería de características de `src/pipeline.py` para construir los trece predictores esperados por el modelo.

El contrato de entrada se encuentra definido en `api/esquemas.py` mediante Pydantic e incluye validación estricta de tipos, rangos, categorías y coherencia entre sesiones.

Ejemplo en PowerShell:

```powershell
$payload = @{
    duracion_estimada_min = 75
    numero_sesion = 1
    sesiones_planificadas = 3
    cancelaciones_previas = 0
    inasistencias_previas = 0
    dias_desde_inicio_plan = 14
    edad_en_atencion = 42
    codigo_procedimiento = "CROWN"
    odontologo_key = "2"
}

Invoke-RestMethod `
    -Uri "http://localhost:8000/predecir" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ($payload | ConvertTo-Json) |
    ConvertTo-Json -Depth 5
```

Alternativamente, en Linux:

```bash
curl -X POST http://localhost:8000/predecir \
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

Respuesta verificada (`200 OK`):

```json
{
    "id_prediccion": "4e4362de-c3c3-400e-8ef5-7da7ce6d006a",
    "probabilidad": 0.09417587082633772,
    "decision": "contactar",
    "accion": "Priorizar recordatorio personalizado y seguimiento preventivo de recepción",
    "umbral_aplicado": 0.01,
    "version_modelo": "1",
    "run_id": "c90a1c37aef940278afc025b1d22dc6b",
    "uso_respaldo": false,
    "latencia_ms": 298.493,
    "timestamp": "2026-11-24T04:38:43.605423+00:00"
}
```

El identificador, la fecha y la latencia varían en cada solicitud.

Cuando la probabilidad estimada es igual o superior a `0.01`, el sistema devuelve la decisión `contactar`; en caso contrario, devuelve `no_contactar`.

Las entradas inválidas generan una respuesta HTTP `422` que identifica el campo que incumple el contrato.

Con `REGISTRAR_SNOWFLAKE=true`, las predicciones se registran en `ORALSOLUTIONS_DW.MLOPS.PREDICCIONES`, incluyendo su identificador, entradas, probabilidad, decisión, versión del modelo, `run_id`, latencia y uso del respaldo.

### 4. Ejecutar las pruebas automatizadas

Desde `proyecto_final/`, con `.venv` activo y las dependencias instaladas:

```powershell
python -m pytest tests/ -v --tb=short
```

Las pruebas se organizan en:

| Archivo | Responsabilidad |
|---|---|
| `tests/test_pipeline.py` | Ingeniería de características, división temporal y ausencia de fuga |
| `tests/test_contrato.py` | Validación del contrato HTTP y entradas inválidas |
| `tests/test_modelo.py` | Reproducción de la métrica registrada en MLflow |
| `tests/test_comportamiento.py` | Comportamiento de negocio, respaldo y retirada del modelo |

**Resultado local verificado: 28 pruebas aprobadas, 0 fallidas y 3 advertencias de dependencias.**

El modelo local de pruebas se encuentra en `tests/fixtures/modelo_champion/`, junto con una muestra fija de 826 observaciones y sus metadatos.

La PR-AUC registrada en MLflow (`0.1598578375`) se reproduce mediante el modelo exportado, con una tolerancia absoluta de `0.0005`.

Si se requiere actualizar el artefacto de pruebas desde el champion activo:

```powershell
python -m api.exportar_modelo
```

Este comando descarga el modelo registrado, verifica el dataset, reproduce la métrica y actualiza los artefactos locales utilizados por CI.

No se requiere un servidor MLflow ni una conexión Snowflake para ejecutar las pruebas automatizadas con los fixtures existentes.

### 5. Versionado y rollback del modelo

MLflow Model Registry constituye la fuente de verdad para el versionado del modelo.

La API consulta el alias `@champion` y recupera la versión correspondiente cuando este cambia.

Para reproducir la demostración de rollback con las versiones originales del PSet 3 y los servicios activos:

```powershell
python scripts/demostrar_rollback.py
```

La demostración verificada realizó la siguiente transición:

```text
Champion v1
    ↓
Cambio temporal del alias a v2
    ↓
Predicción utilizando v2
    ↓
Rollback a v1
    ↓
Recuperación de la predicción original
```

| Estado | Versión | Probabilidad |
|---|---:|---:|
| Champion inicial | 1 | 0.0941758708 |
| Cambio temporal | 2 | 0.1032651291 |
| Después del rollback | 1 | 0.0941758708 |

Se comprobó que el identificador del contenedor, la imagen Docker y la fecha de inicio permanecieron sin cambios durante la demostración.

Las tres predicciones quedaron registradas en Snowflake con su versión y `run_id` correspondiente.

### 6. Simulación de producción por lotes

La simulación utiliza las observaciones posteriores al corte temporal del entrenamiento original (`2025-08-01`), recuperadas desde Snowflake.

Se verificó la equivalencia entre el dataset reconstruido desde Snowflake y el CSV utilizado en el PSet 3.

El histórico de evaluación contiene 826 observaciones distribuidas en 14 lotes semanales. Para el experimento del PSet 4 se ejecutaron y evaluaron ocho lotes consecutivos, cumpliendo el mínimo requerido.

| Lote | Período | Observaciones | Escenario |
|---|---|---:|---|
| L01 | 01/08/2025 – 03/08/2025 | 12 | Original |
| L02 | 04/08/2025 – 10/08/2025 | 37 | Original |
| L03 | 11/08/2025 – 17/08/2025 | 53 | Original |
| L04 | 18/08/2025 – 24/08/2025 | 61 | Original |
| L05 | 25/08/2025 – 31/08/2025 | 54 | Original |
| L06 | 01/09/2025 – 07/09/2025 | 68 | Original |
| L07 | 08/09/2025 – 14/09/2025 | 59 | Deriva inducida |
| L08 | 15/09/2025 – 21/09/2025 | 63 | Deriva inducida |
| **Total** | | **407** | |

La deriva de datos se induce sobre `cancelaciones_previas` e `inasistencias_previas` y afecta consecuentemente a la característica derivada `friccion_previa`. Las etiquetas originales se conservan sin modificaciones.

Los snapshots de la OBT y Gold permiten reproducir el experimento sin alterar las tablas originales del pipeline ELT.

#### Ejecución mediante Kestra

Registrar o actualizar los siguientes flujos en el namespace `oralsolutions.mlops`:

| Archivo | Flow ID |
|---|---|
| `kestra/produccion_lotes.yml` | `produccion_lotes` |
| `kestra/monitoreo.yml` | `monitoreo` |
| `kestra/reentrenamiento.yml` | `reentrenamiento` |

Desde la interfaz de Kestra (`http://localhost:8080`), ejecutar `oralsolutions.mlops.produccion_lotes` indicando el número de lote.

El flujo realiza:

1. Verificación de disponibilidad de FastAPI.
2. Recuperación de observaciones desde Snowflake.
3. Generación y persistencia de predicciones.
4. Registro de telemetría HTTP.
5. Invocación del flujo de monitoreo.

Para reproducir la simulación experimental, ejecutar los lotes del 1 al 8 en orden cronológico.

El simulador admite los 14 lotes definidos, aunque los resultados experimentales reportados corresponden exclusivamente a L01–L08.

#### Ejecución manual alternativa

```powershell
# Validar las entradas de los 14 lotes sin generar predicciones
python -m monitoreo.simular_produccion --verificar

# Procesar el primer lote
python -m monitoreo.simular_produccion --lote 1

# Ejecutar el monitoreo inmediato
python -m monitoreo.monitorear_lote --lote 1 --fecha-virtual 2025-08-03

# Evaluar el lote una vez madurado su target
python -m monitoreo.evaluar_lotes --fecha-virtual 2025-10-02
```

Los lotes completados no se vuelven a procesar al repetir la ejecución.

La comprobación de idempotencia se realizó sobre L01, verificando que sus 12 predicciones permanecieran únicas.

### 7. Monitoreo, deriva y alertas

El monitoreo se realiza en dos momentos:

- **Monitoreo inmediato:** distribución de entradas y predicciones, PSI, calidad de datos, volumen, latencias y errores HTTP.
- **Monitoreo supervisado:** PR-AUC, precisión, recall, matriz de confusión y costos de negocio, una vez transcurrido el horizonte de 60 días.

Para evaluar los ocho lotes pendientes desde Kestra, ejecutar `oralsolutions.mlops.monitoreo` con:

```yaml
lote: 0
fecha_virtual: "2025-11-20"
```

Esta fecha permite evaluar todos los targets correspondientes a L01–L08.

Durante el experimento, L01 se evaluó el 2 de octubre de 2025 y los siete lotes restantes se evaluaron retrospectivamente el 20 de noviembre de 2025, respetando sus respectivas fechas de maduración.

#### Umbrales de monitoreo

| Indicador | Umbral | Severidad |
|---|---|---|
| PSI moderado | 0.10 ≤ PSI < 0.25 | Media |
| PSI significativo | PSI ≥ 0.25 | Alta |
| PR-AUC | Inferior a 0.125553 | Deterioro supervisado |
| Latencia HTTP p95 | Superior a 250 ms | Media |
| Valores fuera del rango de entrenamiento | Superior al 5 % | Media |

El umbral de deterioro supervisado corresponde a la PR-AUC promedio de validación cruzada del champion menos una desviación estándar:

```text
0.1852225492 - 0.0596693412 = 0.1255532080
```

El PSI se complementa con pruebas Kolmogorov-Smirnov para variables numéricas y Chi-cuadrado para variables categóricas, considerando las limitaciones de tamaño muestral y los supuestos de las pruebas.

#### Resultados verificados

| Lote | PR-AUC | PSI máximo | Recall |
|---|---:|---:|---:|
| L01 | 0.1833 | 3.6239 | 100 % |
| L02 | 0.1398 | 2.3312 | 100 % |
| L03 | 0.3427 | 0.3090 | 100 % |
| L04 | 0.1478 | 0.1022 | 100 % |
| L05 | 0.1403 | 0.3093 | 100 % |
| L06 | 0.1718 | 0.3259 | 100 % |
| L07 | 0.5185 | 16.5797 | 100 % |
| L08 | 0.1641 | 16.4179 | 100 % |

Se registraron **16 alertas**, incluyendo la detección de deriva inducida en L07 y L08.

Ninguno de los ocho lotes presentó una PR-AUC inferior al umbral de deterioro establecido. Por tanto, se observó deriva de datos sin evidencia de deterioro supervisado según el criterio implementado.

También se generaron alertas en lotes no alterados, asociadas a cambios naturales de distribución y a la sensibilidad del PSI ante muestras pequeñas.

La latencia HTTP p95 superó los 250 ms en los siete lotes con telemetría disponible. Esto representa una limitación operativa del prototipo, aunque resulta menos restrictiva bajo el patrón de procesamiento batch seleccionado.

L01 no dispone de telemetría HTTP histórica debido a que fue procesado antes de incorporar esta funcionalidad. Sus valores permanecen nulos.

### 8. Dashboard interactivo

El dashboard se encuentra implementado en `monitoreo/tablero.py` mediante Streamlit y consulta directamente Snowflake, FastAPI y MLflow.

Para iniciarlo:

```powershell
docker compose -p proyecto_final up -d tablero
```

Acceder a:

http://localhost:8501

El tablero incluye:

- Estado del servicio y versión activa del modelo.
- Volumen de predicciones y estado de los lotes.
- Evolución de PR-AUC y recall.
- PSI por lote y variable.
- Identificación de escenarios con deriva inducida.
- Métricas de calidad de datos y telemetría HTTP.
- Alertas generadas.
- Comparación económica entre estrategias.
- Resultados y decisiones de reentrenamiento.
- Panel de recepción para inferencia bajo demanda.

La ejecución verificada recuperó ocho lotes evaluados, 407 predicciones, recall agregado del 100 %, precisión agregada del 14,2 % y 16 alertas.

### 9. Reentrenamiento y gobernanza del modelo

El reentrenamiento se implementa en `src/reentrenar.py` y se orquesta mediante `kestra/reentrenamiento.yml`.

Como no se observó deterioro supervisado durante los ocho lotes, se utilizó una **revisión por calendario** como disparador del ciclo experimental, en lugar de reentrenar únicamente por la presencia de deriva de datos.

El proceso reutiliza `src/pipeline.py`, los hiperparámetros del champion y los snapshots originales, sin incorporar las alteraciones artificiales de L07 y L08.

Para reproducir el ciclo, ejecutar en Kestra:

```text
oralsolutions.mlops.reentrenamiento
```

Con el parámetro:

```yaml
fecha_virtual: "2026-01-01"
```

El flujo ejecuta tres tareas:

1. `entrenar_challenger`: entrenamiento, evaluación comparativa y registro en MLflow.
2. `pausa_aprobacion_humana`: revisión manual de las métricas.
3. `aplicar_decision_promocion`: aplicación y registro de la decisión.

#### Configuración experimental

| Parámetro | Valor |
|---|---|
| Disparador | Revisión por calendario |
| Fecha virtual | 2026-01-01 |
| Observaciones de entrenamiento | 1.745 |
| Límite de entrenamiento | 2025-08-27 |
| Separación temporal | 60 días |
| Lote de evaluación | L14 |
| Observaciones de evaluación | 81 |
| Algoritmo | HistGradientBoostingClassifier |
| Umbral operativo | 0.01 |

El challenger se compara con el champion sobre el mismo conjunto de evaluación.

L14 no se utiliza para entrenar el challenger. Sin embargo, formó parte del hold-out original del PSet 3, por lo que esta comparación retrospectiva no constituye una validación externa completamente independiente.

#### Regla de promoción

El nuevo modelo únicamente puede promoverse cuando:

- Su costo estimado de errores es inferior al del champion.
- Su PR-AUC no es inferior a la del champion menos la tolerancia de validación cruzada.
- Existe una aprobación humana explícita.

#### Resultado verificado

| Métrica | Champion v1 | Nuevo challenger |
|---|---:|---:|
| PR-AUC | 0.327545 | 0.378327 |
| Precisión | 18.75 % | 18.52 % |
| Recall | 100 % | 100 % |
| Falsos positivos | 65 | 66 |
| Falsos negativos | 0 | 0 |
| Tasa de contacto | 98.77 % | 100 % |
| Costo estimado de errores | USD 195 | USD 198 |

**Decisión: rechazar la promoción del challenger.**

Aunque el nuevo modelo mejora la PR-AUC, incrementa el costo estimado de errores y no reduce la carga administrativa. Por tanto, no satisface el criterio económico de promoción.

La ejecución de Kestra finalizó en `Success`, con sus tres tareas completadas y la decisión humana registrada.

Se verificó mediante `/modelo` que FastAPI conservó el champion v1.

Las corridas de esta fase se almacenan en el experimento `OralSolutions_Reentrenamiento` de MLflow.

### 10. Retirada del modelo y modo de respaldo

La API dispone de un mecanismo de respaldo basado en una regla de negocio, que permite mantener el servicio operativo cuando el modelo no está disponible o debe retirarse temporalmente.

El mecanismo puede activarse mediante el endpoint administrativo protegido:

```text
POST /admin/respaldo
```

Ejemplo en PowerShell:

```powershell
# Recuperar el token sin mostrarlo en consola
$adminToken = (
    docker compose exec -T api python -c "import os; print(os.getenv('ADMIN_TOKEN', ''))"
).Trim()

$headers = @{
    "x-admin-token" = $adminToken
}

# Activar el modo de respaldo
Invoke-RestMethod `
    -Uri "http://localhost:8000/admin/respaldo" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json; charset=utf-8" `
    -Body (@{
        activar = $true
        motivo = "Retirada temporal para revision"
    } | ConvertTo-Json)
```

Para verificar el estado:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/salud"

Invoke-RestMethod -Uri "http://localhost:8000/alertas"
```

Para restaurar el modelo:

```powershell
Invoke-RestMethod `
    -Uri "http://localhost:8000/admin/respaldo" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json; charset=utf-8" `
    -Body (@{
        activar = $false
        motivo = "Fin de la revision"
    } | ConvertTo-Json)
```

Se verificaron el rechazo de solicitudes administrativas sin autorización, la activación del respaldo, la generación de predicciones mediante la regla de negocio y la restauración del champion.

Las predicciones realizadas en respaldo se identifican mediante:

```json
{
    "version_modelo": "fallback-regla-1-sesion",
    "run_id": "fallback",
    "uso_respaldo": true
}
```

Las puntuaciones asignadas por el respaldo corresponden a una regla operativa y no deben interpretarse como probabilidades calibradas.

Las decisiones de contacto constituyen recomendaciones administrativas sujetas a supervisión humana y no sustituyen decisiones clínicas.

### 11. Integración y entrega continua (CI/CD)

El repositorio utiliza GitHub Actions mediante:

| Archivo | Responsabilidad |
|---|---|
| `.github/workflows/ci.yml` | Instalación de dependencias, pruebas automatizadas y construcción de la imagen |
| `.github/workflows/cd.yml` | Validación previa y publicación de la imagen en GHCR |

CI se ejecuta ante los eventos configurados de `push` y `pull_request`. CD publica la imagen ante un `push` a la rama `main`, después de superar las pruebas automatizadas.

Las imágenes se publican en:

```text
ghcr.io/satiban/oralsolutions-api
```

Cada publicación utiliza dos etiquetas:

- Hash completo del commit.
- `latest`.

La imagen publicada y verificada durante el PSet 4 corresponde a:

```text
ghcr.io/satiban/oralsolutions-api:fe1fdcb0108994682d30f876becd8b9ff3804936
```

Con Docker Compose configurado para utilizar la etiqueta publicada:

```powershell
docker compose -p proyecto_final pull api

docker compose -p proyecto_final up -d --no-deps --no-build api

docker compose -p proyecto_final images api
```

Verificar posteriormente:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/salud"

Invoke-RestMethod -Uri "http://localhost:8000/modelo"
```

Se comprobó la ejecución exitosa de los workflows de CI y CD, la publicación en GHCR, la descarga de la imagen y su funcionamiento mediante Docker Compose.

La publicación del código de la API se automatiza mediante CD. En cambio, la promoción de modelos permanece separada y requiere cumplir las reglas de evaluación y aprobación humana.

### 12. Notebooks de producción y análisis final

Una vez completada la simulación, el monitoreo y el ciclo de reentrenamiento, ejecutar los notebooks restantes en el siguiente orden, utilizando el kernel de `.venv`:

1. **`06_simulacion_produccion.ipynb`:** consulta los lotes, predicciones, maduración de targets, trazabilidad y telemetría HTTP almacenados en Snowflake.
2. **`07_analisis_monitoreo.ipynb`:** analiza desempeño, deriva, alertas, costos, equidad, condiciones de retirada y resultados del reentrenamiento.

Ambos notebooks deben ejecutarse mediante `Restart Kernel → Run All` y guardarse con sus salidas visibles.

Estos notebooks consultan y exportan resultados existentes. No vuelven a ejecutar las predicciones ni generan nuevas versiones del modelo.

### 13. Resultados de negocio en producción simulada

Los ocho lotes evaluados contienen 407 observaciones, de las cuales 57 corresponden a abandono.

El champion identificó los 57 abandonos, sin falsos negativos, pero recomendó contactar a 401 observaciones.

| Estrategia | Costo estimado de errores (USD) |
|---|---:|
| Champion v1 | 1.032 |
| Contactar a todos | 1.050 |
| Regla de una sesión restante | 3.312 |
| No realizar intervención | 6.840 |

Los costos corresponden a una función hipotética de penalización:

```text
Costo = 120 × FN + 3 × FP
```

El champion presenta un costo estimado inferior a las estrategias comparadas. Sin embargo, su ventaja frente a contactar a todos es de únicamente USD 18, evitando seis contactos.

La mejora económica frente a esta última estrategia es marginal y no demuestra una superioridad robusta.

Los resultados incluyen dos lotes con deriva inducida y proceden de datos sintéticos. No representan ahorros económicos reales ni demuestran causalmente que los contactos preventivos reduzcan el abandono.

La meta de negocio de incrementar en diez puntos porcentuales la continuidad de los tratamientos requiere una evaluación prospectiva con intervenciones reales y supervisión clínica.

### 14. Evidencias y trazabilidad

Los resultados de los PSet 3 y 4 se conservan en el directorio `docs/`.

| Archivo | Contenido |
|---|---|
| `resultados_modelos.csv` | Comparación inicial de modelos |
| `resultados_tuning.csv` | Optimización de hiperparámetros |
| `resultados_seleccion.csv` | Selección final del modelo |
| `corridas.csv` | Historial de corridas de MLflow |
| `predicciones.csv` | Predicciones de producción simulada |
| `predicciones_demostracion.csv` | Predicciones de demostraciones de la API |
| `lotes_produccion.csv` | Control y maduración de los lotes |
| `solicitudes_http.csv` | Telemetría HTTP |
| `monitoreo.csv` | Métricas supervisadas y económicas |
| `monitoreo_inmediato.csv` | Métricas inmediatas |
| `deriva_features.csv` | Deriva y calidad por variable |
| `alertas.csv` | Alertas registradas |
| `equidad.csv` | Evaluación por subgrupos |
| `reentrenamientos.csv` | Corridas y decisiones de reentrenamiento |
| `cambios.md` | Correcciones y decisiones metodológicas |

La trazabilidad del sistema permite relacionar:

```text
Predicción individual
    → Sesión y plan de tratamiento
    → Lote de producción
    → Versión del modelo
    → Run ID de MLflow
    → Snapshot utilizado
    → OBT y tabla de hechos Gold
    → Fuentes originales del pipeline ELT
```

La documentación histórica de los PSet anteriores se conserva en `docs/historial/`.