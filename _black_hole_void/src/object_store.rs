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

    /// Retrieve a byte range of an object. The default implementation reads
    /// the whole object and slices; backends with native range support (S3)
    /// override this to stream.
    async fn get_range(&self, key: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        let data = self.get(key).await?;
        let start = offset as usize;
        if start >= data.len() {
            return Ok(Vec::new());
        }
        let end = std::cmp::min(data.len(), start.saturating_add(length as usize));
        Ok(data[start..end].to_vec())
    }

    /// Begin a multipart upload session for `key`. Returns an opaque
    /// backend-specific session ID.
    async fn begin_multipart(&self, key: &str) -> Result<String>;

    /// Upload one part (1-indexed) of a multipart session. Returns an opaque
    /// token identifying the stored part (an ETag for S3).
    async fn upload_part(
        &self,
        key: &str,
        session_id: &str,
        part_number: u32,
        data: Vec<u8>,
    ) -> Result<String>;

    /// Complete a multipart session, materializing the object at `key` from
    /// the stored parts in `part_number` order.
    async fn finish_multipart(
        &self,
        key: &str,
        session_id: &str,
        parts: &[(u32, String)],
    ) -> Result<()>;

    /// Abort a multipart session, discarding any stored parts.
    async fn abort_multipart(&self, key: &str, session_id: &str) -> Result<()>;
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

    async fn get_range(&self, key: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        // S3 Range header uses inclusive byte bounds.
        let end = offset.saturating_add(length.saturating_sub(1));
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .range(format!("bytes={offset}-{end}"))
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

    async fn begin_multipart(&self, key: &str) -> Result<String> {
        let output = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;

        output.upload_id.ok_or_else(|| {
            ObjectStoreError::Message(
                "s3 create_multipart_upload returned no upload id".to_string(),
            )
        })
    }

    async fn upload_part(
        &self,
        key: &str,
        session_id: &str,
        part_number: u32,
        data: Vec<u8>,
    ) -> Result<String> {
        let output = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(session_id)
            .part_number(i32::try_from(part_number).map_err(|_| {
                ObjectStoreError::Message(format!("invalid s3 part number: {part_number}"))
            })?)
            .body(data.into())
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;

        Ok(output.e_tag.unwrap_or_default())
    }

    async fn finish_multipart(
        &self,
        key: &str,
        session_id: &str,
        parts: &[(u32, String)],
    ) -> Result<()> {
        let mut completed = aws_sdk_s3::types::CompletedMultipartUpload::builder();
        for (part_number, etag) in parts {
            let part = aws_sdk_s3::types::CompletedPart::builder()
                .part_number(i32::try_from(*part_number).unwrap_or(i32::MAX))
                .e_tag(etag)
                .build();
            completed = completed.parts(part);
        }

        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(session_id)
            .multipart_upload(completed.build())
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;
        Ok(())
    }

    async fn abort_multipart(&self, key: &str, session_id: &str) -> Result<()> {
        self.client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(session_id)
            .send()
            .await
            .map_err(|e| ObjectStoreError::S3(e.to_string()))?;
        Ok(())
    }
}

/// In-memory object store using a HashMap.
pub struct InMemoryObjectStore {
    map: RwLock<HashMap<String, Vec<u8>>>,
    /// In-flight multipart sessions: session ID -> (part number -> data).
    multipart: RwLock<HashMap<String, HashMap<u32, Vec<u8>>>>,
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
            multipart: RwLock::new(HashMap::new()),
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

    async fn begin_multipart(&self, key: &str) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.multipart
            .write()
            .await
            .insert(session_id.clone(), HashMap::new());
        let _ = key; // Key is bound at finish time.
        Ok(session_id)
    }

    async fn upload_part(
        &self,
        _key: &str,
        session_id: &str,
        part_number: u32,
        data: Vec<u8>,
    ) -> Result<String> {
        let mut sessions = self.multipart.write().await;
        let parts = sessions
            .get_mut(session_id)
            .ok_or_else(|| ObjectStoreError::NotFound(format!("multipart session {session_id}")))?;
        parts.insert(part_number, data);
        Ok(format!("inmemory-part-{part_number}"))
    }

    async fn finish_multipart(
        &self,
        key: &str,
        session_id: &str,
        parts: &[(u32, String)],
    ) -> Result<()> {
        let mut sessions = self.multipart.write().await;
        let stored = sessions
            .remove(session_id)
            .ok_or_else(|| ObjectStoreError::NotFound(format!("multipart session {session_id}")))?;
        drop(sessions);

        let mut data = Vec::new();
        for (part_number, _) in parts {
            let part = stored
                .get(part_number)
                .ok_or_else(|| ObjectStoreError::Message(format!("missing part {part_number}")))?;
            data.extend_from_slice(part);
        }

        self.map.write().await.insert(key.to_string(), data);
        Ok(())
    }

    async fn abort_multipart(&self, _key: &str, session_id: &str) -> Result<()> {
        self.multipart.write().await.remove(session_id);
        Ok(())
    }
}

/// Filesystem-backed object store.
pub struct FilesystemObjectStore {
    root: PathBuf,
}

/// Directory under the store root holding in-flight multipart part files.
const MULTIPART_DIR: &str = ".inprogress";

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

impl FilesystemObjectStore {
    fn session_dir(&self, session_id: &str) -> Result<PathBuf> {
        // Session IDs are server-generated UUIDs, but validate anyway so a
        // corrupted ID cannot escape the multipart directory.
        if session_id.is_empty() || Path::new(session_id).components().count() != 1 {
            return Err(ObjectStoreError::Message(format!(
                "invalid multipart session id: {session_id}"
            )));
        }
        Ok(self.root.join(MULTIPART_DIR).join(session_id))
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

    async fn begin_multipart(&self, _key: &str) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let dir = self.session_dir(&session_id)?;
        fs::create_dir_all(&dir).map_err(|e| {
            ObjectStoreError::Message(format!(
                "failed to create multipart directory {}: {e}",
                dir.display()
            ))
        })?;
        Ok(session_id)
    }

    async fn upload_part(
        &self,
        _key: &str,
        session_id: &str,
        part_number: u32,
        data: Vec<u8>,
    ) -> Result<String> {
        let dir = self.session_dir(session_id)?;
        if !dir.exists() {
            return Err(ObjectStoreError::NotFound(format!(
                "multipart session {session_id}"
            )));
        }
        let part_path = dir.join(format!("{part_number:08x}.part"));
        fs::write(&part_path, data).map_err(|e| {
            ObjectStoreError::Message(format!(
                "failed to write multipart part {} to {}: {e}",
                part_number,
                part_path.display()
            ))
        })?;
        Ok(format!("fs-part-{part_number}"))
    }

    async fn finish_multipart(
        &self,
        key: &str,
        session_id: &str,
        parts: &[(u32, String)],
    ) -> Result<()> {
        let dir = self.session_dir(session_id)?;
        if !dir.exists() {
            return Err(ObjectStoreError::NotFound(format!(
                "multipart session {session_id}"
            )));
        }
        let path = self.object_path(key)?;

        // Assemble the final object from part files in part-number order.
        let mut out = fs::File::create(&path).map_err(|e| {
            ObjectStoreError::Message(format!(
                "failed to create object {} at {}: {e}",
                key,
                path.display()
            ))
        })?;
        for (part_number, _) in parts {
            let part_path = dir.join(format!("{part_number:08x}.part"));
            let mut part = fs::File::open(&part_path).map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => ObjectStoreError::Message(format!(
                    "missing multipart part file {} for session {session_id}",
                    part_path.display()
                )),
                _ => ObjectStoreError::Message(format!(
                    "failed to read multipart part {}: {e}",
                    part_path.display()
                )),
            })?;
            io::copy(&mut part, &mut out).map_err(|e| {
                ObjectStoreError::Message(format!(
                    "failed to assemble object {} from parts: {e}",
                    key
                ))
            })?;
        }

        if let Err(e) = fs::remove_dir_all(&dir) {
            return Err(ObjectStoreError::Message(format!(
                "failed to remove multipart directory {}: {e}",
                dir.display()
            )));
        }
        Ok(())
    }

    async fn abort_multipart(&self, _key: &str, session_id: &str) -> Result<()> {
        let dir = self.session_dir(session_id)?;
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(|e| {
                ObjectStoreError::Message(format!(
                    "failed to remove multipart directory {}: {e}",
                    dir.display()
                ))
            })?;
        }
        Ok(())
    }
}
