//! Object storage abstraction for void objects.
//!
//! Provides a trait backed by S3, the local filesystem, or an in-memory HashMap.

use async_trait::async_trait;
use std::{
    collections::HashMap,
    fs, io,
    path::{Component, Path, PathBuf},
};
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
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;

        let body = output
            .body
            .collect()
            .await
            .map_err(|e| ObjectStoreError::Message(format!("failed to read s3 body: {e}")))?;

        Ok(body.into_bytes().to_vec())
    }
}

/// In-memory object store using a HashMap.
pub struct InMemoryObjectStore {
    map: RwLock<HashMap<String, Vec<u8>>>,
}

impl Default for InMemoryObjectStore {
    fn default() -> Self {
        Self::new()
    }
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
        self.map
            .read()
            .await
            .get(key)
            .cloned()
            .ok_or_else(|| ObjectStoreError::NotFound(key.to_string()))
    }
}

/// Filesystem-backed object store.
pub struct FilesystemObjectStore {
    root: PathBuf,
}

impl FilesystemObjectStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| {
            ObjectStoreError::Message(format!(
                "failed to create filesystem object root {}: {e}",
                root.display()
            ))
        })?;
        Ok(Self { root })
    }

    fn object_path(&self, key: &str) -> Result<PathBuf> {
        let key_path = Path::new(key);
        let mut components = key_path.components();
        let Some(component) = components.next() else {
            return Err(ObjectStoreError::Message(
                "object key cannot be empty".to_string(),
            ));
        };

        if components.next().is_some() || !matches!(component, Component::Normal(_)) {
            return Err(ObjectStoreError::Message(format!(
                "invalid object key for filesystem storage: {key}"
            )));
        }

        Ok(self.root.join(key))
    }
}

#[async_trait]
impl ObjectStore for FilesystemObjectStore {
    async fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        let path = self.object_path(&key)?;
        fs::write(&path, data).map_err(|e| {
            ObjectStoreError::Message(format!(
                "failed to write object {} to {}: {e}",
                key,
                path.display()
            ))
        })?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.object_path(key)?;
        fs::read(&path).map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => ObjectStoreError::NotFound(key.to_string()),
            _ => ObjectStoreError::Message(format!(
                "failed to read object {} from {}: {e}",
                key,
                path.display()
            )),
        })
    }
}
