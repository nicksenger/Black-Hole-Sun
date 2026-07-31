//! Persistence layer for tracking object metadata.
//!
//! When the `postgres` feature is enabled, provides a PostgreSQL-backed store
//! that maps opaque UUIDs to S3 bucket+key references.

#[cfg(feature = "postgres")]
pub mod pg;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use uuid::Uuid;

pub type Error = PersistenceError;
pub type Result<T> = std::result::Result<T, Error>;

/// Metadata for a stored object.
#[derive(Debug, Clone)]
pub struct ObjectRecord {
    /// Opaque UUID assigned by the server.
    pub id: Uuid,
    /// S3 bucket name.
    pub bucket: String,
    /// S3 object key (matches the id).
    pub key: String,
    /// Object size in bytes.
    pub size_bytes: i64,
}

/// Trait for persistence backends that track void objects.
#[async_trait::async_trait]
pub trait VoidStore: dyn_clone::DynClone + Send + Sync {
    /// Run schema migrations (no-op for in-memory stores).
    async fn migrate(&self) -> Result<()>;

    /// Record a newly uploaded object.
    async fn insert_object(
        &self,
        id: Uuid,
        bucket: String,
        key: String,
        size_bytes: i64,
    ) -> Result<()>;

    /// Look up an object by its opaque ID.
    async fn get_object(&self, id: Uuid) -> Result<Option<ObjectRecord>>;

    /// Delete an object record.
    async fn delete_object(&self, id: Uuid) -> Result<()>;
}

dyn_clone::clone_trait_object!(VoidStore);

/// In-memory relational store using a HashMap.
#[derive(Clone)]
pub struct InMemoryStore {
    map: Arc<RwLock<HashMap<Uuid, ObjectRecord>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl VoidStore for InMemoryStore {
    async fn migrate(&self) -> Result<()> {
        // No migrations needed for in-memory store.
        Ok(())
    }

    async fn insert_object(
        &self,
        id: Uuid,
        bucket: String,
        key: String,
        size_bytes: i64,
    ) -> Result<()> {
        let record = ObjectRecord {
            id,
            bucket,
            key,
            size_bytes,
        };
        self.map.write().await.insert(id, record);
        Ok(())
    }

    async fn get_object(&self, id: Uuid) -> Result<Option<ObjectRecord>> {
        let record = self.map.read().await.get(&id).cloned();
        Ok(record)
    }

    async fn delete_object(&self, id: Uuid) -> Result<()> {
        self.map.write().await.remove(&id);
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("{0}")]
    Message(String),
    #[cfg(feature = "postgres")]
    #[error("postgres connection string is required")]
    MissingPostgresConnectionString,
    #[cfg(feature = "postgres")]
    #[error("postgres connection failed: {0}")]
    PostgresConnect(#[source] sqlx::Error),
    #[cfg(feature = "postgres")]
    #[error("postgres query failed: {0}")]
    PostgresQuery(#[source] sqlx::Error),
}
