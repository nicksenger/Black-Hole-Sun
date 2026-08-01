//! Schema migrations for void_objects tables.
//!
//! This module is re-exported publicly so the offline-schema regeneration test
//! can call the migration against a live postgres container.

use sqlx::postgres::PgPool;

/// Current schema version.
pub const SCHEMA_VERSION: i32 = 0;

/// Run V0 migrations on the given pool.
///
/// Creates `void_schema_metadata` and `void_objects` tables with indexes,
/// matching the pattern used by jungle-migrate.
pub async fn migrate_postgres_v0(pool: &PgPool) -> sqlx::Result<()> {
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS void_schema_metadata (
            id SMALLINT PRIMARY KEY,
            version INTEGER NOT NULL
        )
        "#,
    )
    .execute(&mut *tx)
    .await?;

    let version_row = sqlx::query_scalar::<_, i32>(
        "SELECT version FROM void_schema_metadata WHERE id = 1",
    )
    .fetch_optional(&mut *tx)
    .await?;

    if let Some(version) = version_row {
        if version != SCHEMA_VERSION {
            tracing::warn!(
                expected_schema_version = SCHEMA_VERSION,
                actual_schema_version = version,
                "postgres schema version mismatch"
            );
        }
    }

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS void_objects (
            id UUID PRIMARY KEY,
            bucket TEXT NOT NULL,
            key TEXT NOT NULL,
            size_bytes BIGINT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#,
    )
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_void_objects_bucket
        ON void_objects (bucket)
        "#,
    )
    .execute(&mut *tx)
    .await?;

    if version_row.is_none() {
        sqlx::query(
            "INSERT INTO void_schema_metadata (id, version) VALUES (1, $1)",
        )
        .bind(SCHEMA_VERSION)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}
