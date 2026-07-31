//! Object storage abstraction for void objects.
//!
//! Provides a trait backed by either S3 or an in-memory HashMap.

use async_trait::async_trait;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum ObjectStoreError {
    #[error("object not found: {0}")]
    NotFound(String),
    #[error("s3 error: {0}")]
    S3(String),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, ObjectStoreError>;

/// Trait for storing and retrieving object data.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Store an object with the given key.
    async fn put(&self, key: String, data: Vec<u8>) -> Result<()>;

    /// Retrieve an object by key.
    async fn get(&self, key: &str) -> Result<Vec<u8>>;
}

/// S3-backed object store.
pub struct S3Store {
    client: aws_sdk_s3::Client,
    bucket: String,
}

impl S3Store {
    pub fn new(client: aws_sdk_s3::Client, bucket: impl Into<String>) -> Self {
        Self {
            client,
            bucket: bucket.into(),
        }
    }
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(data.into())
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let output = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;

        let body = output.body.collect().await.map_err(|e| {
            ObjectStoreError::Message(format!("failed to read s3 body: {e}"))
        })?;

        Ok(body.into_bytes().to_vec())
    }
}

/// In-memory object store using a HashMap.
pub struct InMemoryObjectStore {
    map: RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryObjectStore {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl ObjectStore for InMemoryObjectStore {
    async fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        self.map.write().await.insert(key, data);
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.map.read().await.get(key).cloned()
            .ok_or_else(|| ObjectStoreError::NotFound(key.to_string()))
    }
}
