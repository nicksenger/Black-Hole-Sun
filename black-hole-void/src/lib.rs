use std::{fs, io, net::SocketAddr, path::PathBuf, sync::Arc};

use aws_sdk_s3::Client as S3Client;
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tracing::{error, info, warn};

pub mod persist;

const DEFAULT_LISTEN_ADDR: &str = "[::1]:4434";
const S3_MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MB

/// Wire request sent by the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum VoidIn {
    /// Upload data to object storage. Server responds with VoidOut::Uploaded(id).
    Upload { data: Vec<u8> },
    /// Download an object by its opaque ID. Server responds with VoidOut::Downloaded(data).
    Download { id: String },
}

/// Wire response sent by the server.
#[derive(Debug, Serialize, Deserialize)]
pub enum VoidOut {
    /// Confirms upload; contains the opaque ID.
    Uploaded { id: String },
    /// Returns downloaded data.
    Downloaded { data: Vec<u8> },
    /// Error message for any failure.
    Error { message: String },
}

#[cfg(feature = "postgres")]
pub struct ServerBuilder {
    keylog: bool,
    key: Option<PathBuf>,
    cert: Option<PathBuf>,
    stateless_retry: bool,
    listen: SocketAddr,
    s3_client: S3Client,
    s3_bucket: String,
    store: Box<dyn persist::VoidStore>,
}

#[cfg(feature = "postgres")]
impl ServerBuilder {
    pub fn new(
        s3_client: S3Client,
        s3_bucket: impl Into<String>,
        store: Box<dyn persist::VoidStore>,
    ) -> Self {
        Self {
            keylog: false,
            key: None,
            cert: None,
            stateless_retry: false,
            listen: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default listen address must be valid"),
            s3_client,
            s3_bucket: s3_bucket.into(),
            store,
        }
    }

    pub fn keylog(mut self, v: bool) -> Self {
        self.keylog = v;
        self
    }

    pub fn key(mut self, p: PathBuf) -> Self {
        self.key = Some(p);
        self
    }

    pub fn cert(mut self, p: PathBuf) -> Self {
        self.cert = Some(p);
        self
    }

    pub fn stateless_retry(mut self, v: bool) -> Self {
        self.stateless_retry = v;
        self
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen = addr;
        self
    }

    pub async fn run(self) -> Result<()> {
        // Run migrations before accepting connections.
        self.store.migrate().await.map_err(|e| {
            ServerError::Store(persist::PersistenceError::Message(format!(
                "migration failed: {e}"
            )))
        })?;

        let (cert_chain, key) = if self.key.is_some() && self.cert.is_some() {
            load_user_cert_chain_and_key(&self.key.unwrap(), &self.cert.unwrap())?
        } else {
            load_or_generate_self_signed_cert()?
        };

        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(cert_chain, key)
            .map_err(ServerError::RustlsCertConfig)?;

        if self.keylog {
            server_config.key_log = Arc::new(rustls::KeyLogFile::new());
        }

        let crypto = QuicServerConfig::try_from(server_config)
            .map_err(ServerError::QuicServerConfig)?;

        let endpoint_cfg = quinn::ServerConfig::with_crypto(Arc::new(crypto));

        let listener = std::net::UdpSocket::bind(self.listen)
            .map_err(ServerError::BindEndpoint)?;
        let runtime = quinn::TokioRuntime;
        let endpoint = quinn::Endpoint::new(
            Default::default(),
            Some(endpoint_cfg),
            listener,
            Arc::new(runtime),
        ).map_err(ServerError::BindEndpoint)?;

        let local_addr = endpoint.local_addr().map_err(ServerError::LocalAddr)?;
        info!(%local_addr, bucket = %self.s3_bucket, "listening");

        let context = Arc::new(VoidContext {
            s3_client: self.s3_client,
            s3_bucket: self.s3_bucket,
            store: self.store,
        });

        loop {
            let conn = tokio::select! {
                incoming = endpoint.accept() => match incoming {
                    Some(c) => c,
                    None => break,
                },
            };

            if self.stateless_retry && !conn.remote_address_validated() {
                info!("requiring connection to validate its address");
                let _ = conn.retry();
                continue;
            }

            info!(remote = %conn.remote_address(), "accepting connection");
            let ctx = Arc::clone(&context);
            tokio::spawn(handle_connection(conn, ctx));
        }

        Ok(())
    }
}

struct VoidContext {
    s3_client: S3Client,
    s3_bucket: String,
    #[cfg(feature = "postgres")]
    store: Box<dyn persist::VoidStore>,
}

async fn handle_connection(incoming: quinn::Incoming, context: Arc<VoidContext>) {
    let connection = match incoming.await {
        Ok(c) => c,
        Err(e) => {
            error!("connection failed: {e}");
            return;
        }
    };

    info!("established");

    loop {
        let stream = match connection.accept_bi().await {
            Err(quinn::ConnectionError::ApplicationClosed { .. }) => {
                info!("connection closed");
                return;
            }
            Err(e) => {
                error!("stream error: {e}");
                return;
            }
            Ok(s) => s,
        };

        let ctx = Arc::clone(&context);
        tokio::spawn(handle_stream(stream, ctx));
    }
}

async fn handle_stream(
    (mut send, recv): (quinn::SendStream, quinn::RecvStream),
    context: Arc<VoidContext>,
) {
    let request = match read_frame(recv).await {
        Ok(r) => r,
        Err(e) => {
            error!("failed to read request frame: {e}");
            return;
        }
    };

    let response = match request {
        VoidIn::Upload { data } => handle_upload(&context, data).await,
        VoidIn::Download { id } => handle_download(&context, id).await,
    };

    if let Err(e) = write_frame(&mut send, &response).await {
        error!("failed to write response frame: {e}");
    }
}

async fn handle_upload(context: &VoidContext, data: Vec<u8>) -> VoidOut {
    if data.len() > S3_MAX_FRAME_SIZE {
        return VoidOut::Error {
            message: format!(
                "upload size {} exceeds maximum {}",
                data.len(),
                S3_MAX_FRAME_SIZE
            ),
        };
    }

    let id = uuid::Uuid::new_v4();
    let key = id.to_string();
    let size_bytes = i64::try_from(data.len()).unwrap_or(i64::MAX);

    match context.s3_client.put_object()
        .bucket(&context.s3_bucket)
        .key(&key)
        .body(data.into())
        .send()
        .await
    {
        Ok(_) => {
            // Persist the object metadata to postgres.
            #[cfg(feature = "postgres")]
            if let Err(e) = context.store.insert_object(
                id,
                context.s3_bucket.clone(),
                key.clone(),
                size_bytes,
            ).await {
                warn!(error = %e, "failed to persist object metadata");
            }

            info!(%id, "uploaded");
            VoidOut::Uploaded { id: id.to_string() }
        }
        Err(e) => {
            error!(error = %e, "s3 put_object failed");
            VoidOut::Error {
                message: format!("upload failed: {e}"),
            }
        }
    }
}

async fn handle_download(context: &VoidContext, id_str: String) -> VoidOut {
    let id = match uuid::Uuid::parse_str(&id_str) {
        Ok(u) => u,
        Err(_) => return VoidOut::Error {
            message: format!("invalid object id: {id_str}"),
        },
    };

    // Look up the object in postgres to get bucket+key.
    #[cfg(feature = "postgres")]
    let record = match context.store.get_object(id).await {
        Ok(Some(r)) => r,
        Ok(None) => return VoidOut::Error {
            message: format!("object not found: {id}"),
        },
        Err(e) => {
            error!(error = %e, "failed to look up object");
            return VoidOut::Error {
                message: format!("lookup failed: {e}"),
            };
        }
    };

    #[cfg(feature = "postgres")]
    match context.s3_client.get_object()
        .bucket(&record.bucket)
        .key(&record.key)
        .send()
        .await
    {
        Ok(output) => {
            let body = output.body.collect().await;
            match body {
                Ok(aggregated) => {
                    let data = aggregated.into_bytes();
                    info!(%id, bytes = data.len(), "downloaded");
                    VoidOut::Downloaded {
                        data: data.to_vec(),
                    }
                }
                Err(e) => {
                    error!(error = %e, "failed to read s3 body");
                    VoidOut::Error {
                        message: format!("download failed: {e}"),
                    }
                }
            }
        }
        Err(e) => {
            warn!(%id, error = %e, "s3 get_object failed (object may not exist)");
            VoidOut::Error {
                message: format!("not found: {e}"),
            }
        }
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = id;
        VoidOut::Error {
            message: "postgres feature not enabled".to_string(),
        }
    }
}

/// Read a length-prefixed postcard frame from the stream.
async fn read_frame(mut recv: quinn::RecvStream) -> std::result::Result<VoidIn, ServerError> {
    let len = match recv.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(ServerError::UnexpectedEof);
        }
        Err(e) => {
            return Err(ServerError::ReadFrameLength(e));
        }
    };

    if len > S3_MAX_FRAME_SIZE {
        return Err(ServerError::FrameTooLarge(len));
    }

    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload).await
        .map_err(ServerError::ReadFramePayload)?;

    from_bytes(&payload).map_err(ServerError::DecodeRequest)
}

/// Write a length-prefixed postcard frame to the stream.
async fn write_frame(send: &mut quinn::SendStream, out: &VoidOut) -> std::result::Result<(), ServerError> {
    let payload = to_allocvec(out).map_err(ServerError::EncodeResponse)?;
    let len = u32::try_from(payload.len())
        .map_err(|_| ServerError::FrameTooLarge(payload.len()))?;

    send.write_all(&len.to_be_bytes()).await
        .map_err(ServerError::WriteFrame)?;
    send.write_all(&payload).await
        .map_err(ServerError::WriteFrame)?;
    Ok(())
}

fn load_user_cert_chain_and_key(
    key_path: &PathBuf,
    cert_path: &PathBuf,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let key = if key_path.extension().is_some_and(|x| x == "der") {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            fs::read(key_path).map_err(ServerError::ReadPrivateKeyFile)?,
        ))
    } else {
        PrivateKeyDer::from_pem_file(key_path).map_err(ServerError::ReadPrivateKeyPem)?
    };

    let cert_chain = if cert_path.extension().is_some_and(|x| x == "der") {
        vec![CertificateDer::from(
            fs::read(cert_path).map_err(ServerError::ReadCertChainFile)?,
        )]
    } else {
        CertificateDer::pem_file_iter(cert_path)
            .map_err(ServerError::ReadCertChainPem)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(ServerError::InvalidPemCert)?
    };

    Ok((cert_chain, key))
}

fn load_or_generate_self_signed_cert(
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let dirs = directories_next::ProjectDirs::from("org", "blackhole", "void").unwrap();
    let path = dirs.data_local_dir();
    let cert_path = path.join("cert.der");
    let key_path = path.join("key.der");

    let (cert, key) = match fs::read(&cert_path).and_then(|x| Ok((x, fs::read(&key_path)?))) {
        Ok((cert, key)) => (
            CertificateDer::from(cert),
            PrivateKeyDer::try_from(key).map_err(|e| ServerError::ParseDerKey(e.to_owned()))?,
        ),
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
            info!("generating self-signed certificate");
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
            let key = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
            let cert = cert.cert.into();
            fs::create_dir_all(path).map_err(ServerError::CreateCertDir)?;
            fs::write(&cert_path, &cert).map_err(ServerError::WriteCert)?;
            fs::write(&key_path, key.secret_pkcs8_der()).map_err(ServerError::WritePrivateKey)?;
            (cert, key.into())
        }
        Err(e) => {
            return Err(ServerError::ReadCertificate(e));
        }
    };

    Ok((vec![cert], key))
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("failed to read private key file: {0}")]
    ReadPrivateKeyFile(#[source] io::Error),
    #[error("failed to read PEM from private key file: {0}")]
    ReadPrivateKeyPem(#[source] rustls::pki_types::pem::Error),
    #[error("failed to read certificate chain file: {0}")]
    ReadCertChainFile(#[source] io::Error),
    #[error("failed to read PEM from certificate chain file: {0}")]
    ReadCertChainPem(#[source] rustls::pki_types::pem::Error),
    #[error("invalid PEM-encoded certificate: {0}")]
    InvalidPemCert(#[source] rustls::pki_types::pem::Error),
    #[error("failed to parse DER private key: {0}")]
    ParseDerKey(String),
    #[error("failed to create certificate directory: {0}")]
    CreateCertDir(#[source] io::Error),
    #[error("failed to write certificate: {0}")]
    WriteCert(#[source] io::Error),
    #[error("failed to write private key: {0}")]
    WritePrivateKey(#[source] io::Error),
    #[error("failed to read certificate: {0}")]
    ReadCertificate(io::Error),
    #[error("failed to configure rustls certificate: {0}")]
    RustlsCertConfig(#[source] rustls::Error),
    #[error("failed to build QUIC rustls config: {0}")]
    QuicServerConfig(#[source] quinn::crypto::rustls::NoInitialCipherSuite),
    #[error("failed to bind QUIC endpoint: {0}")]
    BindEndpoint(#[source] io::Error),
    #[error("failed to fetch local listen address: {0}")]
    LocalAddr(#[source] io::Error),
    #[error("unexpected EOF while reading frame length")]
    UnexpectedEof,
    #[error("failed to read frame length: {0}")]
    ReadFrameLength(io::Error),
    #[error("frame payload too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("failed to read frame payload: {0}")]
    ReadFramePayload(#[source] quinn::ReadExactError),
    #[error("failed to decode request: {0}")]
    DecodeRequest(postcard::Error),
    #[error("failed to encode response: {0}")]
    EncodeResponse(postcard::Error),
    #[error("failed to write frame: {0}")]
    WriteFrame(quinn::WriteError),
    #[cfg(feature = "postgres")]
    #[error("persistence error: {0}")]
    Store(#[source] persist::PersistenceError),
}

pub type Result<T> = std::result::Result<T, ServerError>;

pub fn init_tracing() -> std::result::Result<(), tracing::subscriber::SetGlobalDefaultError> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish(),
    )
}
