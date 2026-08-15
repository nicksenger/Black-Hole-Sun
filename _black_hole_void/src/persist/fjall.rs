//! Fjall persistence backend for void object metadata.

use crate::persist::{ObjectRecord, PersistenceError, Result, VoidStore};
use async_trait::async_trait;
use std::path::Path;

const OBJECTS_KEYSPACE: &str = "void_objects";

#[derive(Clone)]
pub struct FjallStore {
    db: fjall::Database,
    objects: fjall::Keyspace,
}

impl FjallStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let db = fjall::Database::builder(path.as_ref())
            .open()
            .map_err(|e| {
                PersistenceError::Message(format!("failed to open fjall database: {e}"))
            })?;
        let objects = db
            .keyspace(OBJECTS_KEYSPACE, fjall::KeyspaceCreateOptions::default)
            .map_err(|e| {
                PersistenceError::Message(format!(
                    "failed to open fjall keyspace {OBJECTS_KEYSPACE}: {e}"
                ))
            })?;
        Ok(Self { db, objects })
    }

    fn persist_journal(&self) -> Result<()> {
        self.db
            .persist(fjall::PersistMode::Buffer)
            .map_err(|e| PersistenceError::Message(format!("failed to persist fjall journal: {e}")))
    }
}

#[async_trait]
impl VoidStore for FjallStore {
    async fn migrate(&self) -> Result<()> {
        // Fjall keyspace open/create acts as schema setup.
        Ok(())
    }

    async fn insert_object(
        &self,
        id: uuid::Uuid,
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
        let encoded = postcard::to_allocvec(&record).map_err(|e| {
            PersistenceError::Message(format!("failed to encode object record for fjall: {e}"))
        })?;

        self.objects.insert(id.as_bytes(), encoded).map_err(|e| {
            PersistenceError::Message(format!("failed to write fjall object record: {e}"))
        })?;
        self.persist_journal()
    }

    async fn get_object(&self, id: uuid::Uuid) -> Result<Option<ObjectRecord>> {
        let Some(value) = self.objects.get(id.as_bytes()).map_err(|e| {
            PersistenceError::Message(format!("failed to read fjall object record: {e}"))
        })?
        else {
            return Ok(None);
        };

        let record = postcard::from_bytes(value.as_ref()).map_err(|e| {
            PersistenceError::Message(format!("failed to decode fjall object record: {e}"))
        })?;
        Ok(Some(record))
    }

    async fn delete_object(&self, id: uuid::Uuid) -> Result<()> {
        self.objects.remove(id.as_bytes()).map_err(|e| {
            PersistenceError::Message(format!("failed to remove fjall object record: {e}"))
        })?;
        self.persist_journal()
    }
}
