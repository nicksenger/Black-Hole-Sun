//! PostgreSQL persistence backend for void objects.

use crate::migrate;
use crate::persist::{ObjectRecord, PersistenceError, Result, VoidStore};
use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;

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
        migrate::migrate_postgres_v0(&self.pool)
            .await
            .map_err(PersistenceError::PostgresQuery)
    }

    async fn insert_object(
        &self,
        id: uuid::Uuid,
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

    async fn get_object(&self, id: uuid::Uuid) -> Result<Option<ObjectRecord>> {
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

        row.map(|row| {
            Ok(ObjectRecord {
                id: row.get("id"),
                bucket: row.get("bucket"),
                key: row.get("key"),
                size_bytes: row.get("size_bytes"),
            })
        })
        .transpose()
    }

    async fn delete_object(&self, id: uuid::Uuid) -> Result<()> {
        sqlx::query("DELETE FROM void_objects WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(PersistenceError::PostgresQuery)?;
        Ok(())
    }
}
