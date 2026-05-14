-- drift/ingestion/athena_queries.sql
-- Run these in order in the AWS Athena console
-- to create the database and table pointing at your S3 data.

-- ============================================================
-- 1. Create database
-- ============================================================
CREATE DATABASE IF NOT EXISTS drift;

-- ============================================================
-- 2. Create external table pointing at S3 Parquet files
--    Replace <YOUR_BUCKET> with your actual bucket name.
-- ============================================================
CREATE EXTERNAL TABLE IF NOT EXISTS drift.events (
    event_time      TIMESTAMP,
    event_type      STRING,
    product_id      INT,
    category_id     BIGINT,
    category_code   STRING,
    brand           STRING,
    price           FLOAT,
    user_id         BIGINT,
    user_session    STRING
)
PARTITIONED BY (year INT, month INT)
STORED AS PARQUET
LOCATION 's3://drift-data-547519648043/raw/'
TBLPROPERTIES ('parquet.compress' = 'SNAPPY');

-- ============================================================
-- 3. Load partitions (run after uploading new Parquet files)
-- ============================================================
MSCK REPAIR TABLE drift.events;

-- ============================================================
-- 4. Sanity check — should return row counts per month
-- ============================================================
SELECT
    year,
    month,
    COUNT(*) AS row_count
FROM drift.events
GROUP BY year, month
ORDER BY year, month;

-- ============================================================
-- 5. Event type distribution
-- ============================================================
SELECT
    event_type,
    COUNT(*)                            AS total_events,
    COUNT(DISTINCT user_id)             AS unique_users,
    ROUND(COUNT(*) * 100.0
          / SUM(COUNT(*)) OVER (), 2)   AS pct
FROM drift.events
GROUP BY event_type
ORDER BY total_events DESC;

-- ============================================================
-- 6. Daily active users — used in EDA notebook
-- ============================================================
SELECT
    DATE(event_time)        AS event_date,
    COUNT(DISTINCT user_id) AS dau
FROM drift.events
GROUP BY DATE(event_time)
ORDER BY event_date;

-- ============================================================
-- 7. Top 20 brands by revenue
-- ============================================================
SELECT
    brand,
    COUNT(*)            AS purchases,
    ROUND(SUM(price), 2) AS total_revenue
FROM drift.events
WHERE event_type = 'purchase'
  AND brand IS NOT NULL
GROUP BY brand
ORDER BY total_revenue DESC
LIMIT 20;

-- ============================================================
-- 8. User-level session summary — input to feature engineering
-- ============================================================
SELECT
    user_id,
    user_session,
    MIN(event_time)                             AS session_start,
    MAX(event_time)                             AS session_end,
    COUNT(*)                                    AS total_events,
    SUM(CASE WHEN event_type = 'view'
             THEN 1 ELSE 0 END)                 AS views,
    SUM(CASE WHEN event_type = 'cart'
             THEN 1 ELSE 0 END)                 AS cart_adds,
    SUM(CASE WHEN event_type = 'purchase'
             THEN 1 ELSE 0 END)                 AS purchases,
    SUM(CASE WHEN event_type = 'purchase'
             THEN price ELSE 0 END)             AS session_revenue
FROM drift.events
GROUP BY user_id, user_session
ORDER BY user_id, session_start;
