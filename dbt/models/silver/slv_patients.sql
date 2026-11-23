{{ config(
    materialized='table'
) }}

WITH source AS (

    SELECT
        id,
        first_name,
        last_name,
        email,
        birth_date,
        insurer_id,
        cover_fraction,
        registered_at

    FROM {{ ref('brz_patients') }}

),

cleaned AS (

    SELECT
        id,
        NULLIF(TRIM(first_name), '') AS first_name,
        NULLIF(TRIM(last_name), '') AS last_name,

        CASE
            WHEN email IS NULL THEN NULL
            WHEN TRIM(email) = '' THEN NULL
            WHEN REGEXP_LIKE(
                LOWER(TRIM(email)),
                '[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'
            )
            THEN LOWER(TRIM(email))
            ELSE NULL
        END AS email,

        birth_date,
        insurer_id,

        CASE
            WHEN cover_fraction BETWEEN 0 AND 1 THEN cover_fraction
            ELSE NULL
        END AS cover_fraction,

        registered_at

    FROM source

)

SELECT
    id,
    first_name,
    last_name,
    email,
    birth_date,
    insurer_id,
    cover_fraction,
    registered_at

FROM cleaned