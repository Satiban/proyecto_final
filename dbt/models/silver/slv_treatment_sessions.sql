{{ config(
    materialized='table'
) }}

WITH sesiones AS (

    SELECT
        id,
        treatment_plan_id,
        session_number,
        planned_date,
        completed_at,
        LOWER(NULLIF(TRIM(status), '')) AS status,
        appointment_id,
        missed_appointment_id

    FROM {{ ref('brz_treatment_sessions') }}

),

planes AS (

    SELECT
        id AS plan_id,
        start_date

    FROM {{ ref('brz_treatment_plans') }}

)

SELECT
    s.id,
    s.treatment_plan_id,
    s.session_number,
    s.planned_date,
    s.completed_at,
    s.status,
    s.appointment_id,
    s.missed_appointment_id,
    CASE
        WHEN (s.planned_date IS NOT NULL AND p.start_date IS NOT NULL AND CAST(s.planned_date AS DATE) < CAST(p.start_date AS DATE))
        OR (s.completed_at IS NOT NULL AND p.start_date IS NOT NULL AND CAST(s.completed_at AS DATE) < CAST(p.start_date AS DATE))
        THEN TRUE
        ELSE FALSE
    END AS flag_fecha_anterior_inicio

FROM sesiones AS s
LEFT JOIN planes AS p
    ON s.treatment_plan_id = p.plan_id