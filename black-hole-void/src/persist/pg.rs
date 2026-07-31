//! PostgreSQL persistence backend for void objects.

use crate::persist::{ObjectRecord, PersistenceError, Result, VoidStore};
use sqlx::Row;
use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// Current schema version for migrations.
const SCHEMA_VERSION: i32 = 0;

#[derive(Debug, Clone)]
pub struct PgStore {
    pool: sqlx::PgPool,
}

impl PgStore {
    pub fn builder() -> PgStoreBuilder {
        PgStoreBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct PgStoreBuilder {
    connection_string: Option<String>,
    max_connections: u32,
}

impl PgStoreBuilder {
    pub fn connection_string(mut self, value: impl Into<String>) -> Self {
        self.connection_string = Some(value.into());
        self
    }

    pub fn max_connections(mut self, value: u32) -> Self {
        self.max_connections = value;
        self
    }

    pub async fn build(self) -> Result<PgStore> {
        let connection_string = self
            .connection_string
            .ok_or(PersistenceError::MissingPostgresConnectionString)?;
        let pool = PgPoolOptions::new()
            .max_connections(self.max_connections)
            .connect(&connection_string)
            .await
            .map_err(PersistenceError::PostgresConnect)?;
        Ok(PgStore { pool })
    }
}

#[async_trait]
impl VoidStore for PgStore {
    async fn migrate(&self) -> Result<()> {
        migrate_v0(&self.pool).await.map_err(PersistenceError::PostgresQuery)
    }

    async fn insert_object(
        &self,
        id: Uuid,
        bucket: String,
        key: String,
        size_bytes: i64,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO void_objects (id, bucket, key, size_bytes)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE SET size_bytes = EXCLUDED.size_bytes
            "#,
        )
        .bind(id)
        .bind(bucket)
        .bind(key)
        .bind(size_bytes)
        .execute(&self.pool)
        .await
        .map_err(PersistenceError::PostgresQuery)?;
        Ok(())
    }

    async fn get_object(&self, id: Uuid) -> Result<Option<ObjectRecord>> {
        let row = sqlx::query(
            r#"
            SELECT id, bucket, key, size_bytes
            FROM void_objects
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(PersistenceError::PostgresQuery)?;

        row.map(|row| Ok(ObjectRecord {
            id: row.get("id"),
            bucket: row.get("bucket"),
            key: row.get("key"),
            size_bytes: row.get("size_bytes"),
        })).transpose()
    }

    async fn delete_object(&self, id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM void_objects WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(PersistenceError::PostgresQuery)?;
        Ok(())
    }
}

/// Migration V0: create the schema metadata table and void_objects table.
async fn migrate_v0(pool: &sqlx::PgPool) -> sqlx::Result<()> {
    let mut tx = pool.begin().await?;

    // Schema metadata tracking (same pattern as jungle).
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

    // Main objects table: maps opaque UUIDs to S3 bucket+key references.
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

    // Index on bucket for listing objects per bucket.
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
