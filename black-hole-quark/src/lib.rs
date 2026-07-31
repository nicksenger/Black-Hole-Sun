use std::{fs, io, net::SocketAddr, path::PathBuf, sync::Arc};

use paramecia_engine::ModelEngine;
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, info, warn};

use black_hole_type::{ObjectId, QuzoIn, QuzoInferRequest, QuzoOut};

const DEFAULT_LISTEN_ADDR: &str = "[::1]:4433";
const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MB

// ---------------------------------------------------------------------------
// Void client — connects to black-hole-void over QUIC, sends/receives frames
// ---------------------------------------------------------------------------

/// Wire request sent to the void service.
#[derive(Debug, Serialize, Deserialize)]
enum VoidIn {
    Upload { data: Vec<u8> },
    Download { id: String },
}

/// Wire response from the void service.
#[derive(Debug, Serialize, Deserialize)]
enum VoidOut {
    Uploaded { id: String },
    Downloaded { data: Vec<u8> },
    Error { message: String },
}

/// No-op certificate verifier for connecting to void (self-signed certs in local dev).
#[derive(Debug)]
struct VoidCertVerifier;

impl rustls::client::danger::ServerCertVerifier for VoidCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}


/// A QUIC client connection to the void object store.
pub struct VoidClient {
    endpoint: quinn::Endpoint,
    remote_addr: SocketAddr,
}

impl VoidClient {
    /// Connect to the void service at `addr`.
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        // Void uses self-signed certs by default in local dev — skip verification.
        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(VoidCertVerifier))
            .with_no_client_auth();

        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
            .map_err(|e| ServerError::VoidCrypto(e.to_string()))?;

        let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

        // Bind to any local address/port.
        let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let mut endpoint = quinn::Endpoint::client(local_addr)
            .map_err(|e| ServerError::BindVoidClient(e))?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            endpoint,
            remote_addr: addr,
        })
    }

    /// Open a bidirectional stream, send a void request, and read the response.
    async fn call(&self, req: VoidIn) -> Result<VoidOut> {
        let server_name = self.remote_addr.ip().to_string();
        let conn = self
            .endpoint
            .connect(self.remote_addr, &server_name)
            .map_err(|e| ServerError::VoidConnect(e.to_string()))?
            .await
            .map_err(|e| ServerError::VoidConnect(e.to_string()))?;

        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| ServerError::VoidStream(e.to_string()))?;

        write_frame(&mut send, &req).await?;
        let resp = read_frame(&mut recv).await?;
        Ok(resp)
    }

    /// Download an object from void by its ID. Returns the raw bytes.
    pub async fn download(&self, id: &str) -> Result<Vec<u8>> {
        let resp = self.call(VoidIn::Download { id: id.to_string() }).await?;
        match resp {
            VoidOut::Downloaded { data } => Ok(data),
            VoidOut::Error { message } => Err(ServerError::VoidError(message)),
            _ => Err(ServerError::VoidError("unexpected void response for download".into())),
        }
    }

    /// Upload data to void. Returns the assigned object ID.
    pub async fn upload(&self, data: Vec<u8>) -> Result<ObjectId> {
        let resp = self.call(VoidIn::Upload { data }).await?;
        match resp {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(ServerError::VoidError(message)),
            _ => Err(ServerError::VoidError("unexpected void response for upload".into())),
        }
    }
}

/// Write a length-prefixed postcard frame to a QUIC send stream.
async fn write_frame(send: &mut quinn::SendStream, msg: &impl Serialize) -> Result<()> {
    let payload = to_allocvec(msg).map_err(ServerError::EncodeFrame)?;
    let len = u32::try_from(payload.len())
        .map_err(|_| ServerError::FrameTooLarge(payload.len()))?;

    send.write_all(&len.to_be_bytes()).await
        .map_err(ServerError::WriteFrame)?;
    send.write_all(&payload).await
        .map_err(ServerError::WriteFrame)?;
    Ok(())
}

/// Read a length-prefixed postcard frame from a QUIC recv stream.
async fn read_frame<T: for<'de> Deserialize<'de>>(recv: &mut quinn::RecvStream) -> Result<T> {
    let len = match recv.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(ServerError::UnexpectedEof);
        }
        Err(e) => return Err(ServerError::ReadFrameLength(e)),
    };

    if len > MAX_FRAME_SIZE {
        return Err(ServerError::FrameTooLarge(len));
    }

    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(ServerError::ReadFramePayload)?;

    from_bytes(&payload).map_err(ServerError::DecodeFrame)
}

// ---------------------------------------------------------------------------
// QuZO state — tracks perturb/up-loss between client requests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuzoState {
    /// Initial or post-update: next expected step is PerturbUp.
    Idle,
    /// After PerturbUp: waiting for Infer (up).
    PostPerturbUp,
    /// After up-inference: waiting for PerturbDown with loss_up.
    AwaitingPerturbDown,
    /// After PerturbDown: waiting for Infer (down).
    PostPerturbDown,
    /// After down-inference: waiting for Optimize with loss_down.
    AwaitingOptimize,
}

struct QuzoSession {
    state: QuzoState,
}

impl QuzoSession {
    fn new() -> Self {
        Self {
            state: QuzoState::Idle,
        }
    }
}

// ---------------------------------------------------------------------------
// Server context — shared across connections
// ---------------------------------------------------------------------------

struct QuarkContext {
    engine: Arc<ModelEngine>,
    void_client: Option<Arc<VoidClient>>,
    quzo: tokio::sync::Mutex<QuzoSession>,
}

// ---------------------------------------------------------------------------
// Server builder
// ---------------------------------------------------------------------------

pub struct ServerBuilder {
    keylog: bool,
    key: Option<PathBuf>,
    cert: Option<PathBuf>,
    stateless_retry: bool,
    listen: SocketAddr,
    model_path: PathBuf,
    void_addr: Option<SocketAddr>,
}

impl ServerBuilder {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            keylog: false,
            key: None,
            cert: None,
            stateless_retry: false,
            listen: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default listen address must be valid"),
            model_path: model_path.into(),
            void_addr: None,
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

    pub fn void_addr(mut self, addr: SocketAddr) -> Self {
        self.void_addr = Some(addr);
        self
    }

    pub async fn run(self) -> Result<()> {
        // Load the model engine.
        let model_path_str = self.model_path.to_string_lossy().to_string();
        info!(model_path = %model_path_str, "loading model");
        let engine = paramecia_engine::ModelEngineBuilder::new(&self.model_path)
            .build()
            .map_err(|e| ServerError::ModelError(e.to_string()))?;
        info!("model loaded");

        // Optionally connect to void.
        let void_client = if let Some(addr) = self.void_addr {
            info!(%addr, "connecting to void");
            let client = VoidClient::connect(addr).await?;
            Some(Arc::new(client))
        } else {
            warn!("no void address configured — inference will fail without object store");
            None
        };

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
        info!(%local_addr, "listening");

        let context = Arc::new(QuarkContext {
            engine: Arc::new(engine),
            void_client,
            quzo: tokio::sync::Mutex::new(QuzoSession::new()),
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

impl Default for ServerBuilder {
    fn default() -> Self {
        panic!("use ServerBuilder::new(model_path) instead");
    }
}

// ---------------------------------------------------------------------------
// Connection / stream handlers
// ---------------------------------------------------------------------------

async fn handle_connection(incoming: quinn::Incoming, context: Arc<QuarkContext>) {
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
    (mut send, mut recv): (quinn::SendStream, quinn::RecvStream),
    context: Arc<QuarkContext>,
) {
    let req: QuzoIn = match read_frame(&mut recv).await {
        Ok(r) => r,
        Err(e) => {
            let _ = write_frame(&mut send, &QuzoOut::Error { message: e.to_string() }).await;
            return;
        }
    };

    debug!(?req, "handling request");

    let out = match handle_request(req, &context).await {
        Ok(o) => o,
        Err(e) => QuzoOut::Error { message: e.to_string() },
    };

    if write_frame(&mut send, &out).await.is_err() {
        warn!("failed to write response");
    }
}

async fn handle_request(
    req: QuzoIn,
    ctx: &QuarkContext,
) -> Result<QuzoOut> {
    match req {
        QuzoIn::PerturbUp => handle_perturb_up(ctx).await,
        QuzoIn::Infer { input_id } => handle_infer(input_id, ctx).await,
        QuzoIn::PerturbDown => handle_perturb_down(ctx).await,
        QuzoIn::Optimize { loss_up, loss_down } => handle_optimize(loss_up, loss_down, ctx).await,
    }
}

// ---------------------------------------------------------------------------
// QuZO step handlers
// ---------------------------------------------------------------------------

async fn handle_perturb_up(ctx: &QuarkContext) -> Result<QuzoOut> {
    let mut session = ctx.quzo.lock().await;
    if session.state != QuzoState::Idle {
        return Err(ServerError::InvalidQuzoState(
            format!("expected Idle, got {:?}", session.state),
        ));
    }

    ctx.engine.perturb_up().await.map_err(|e| {
        ServerError::ModelError(e.to_string())
    })?;

    session.state = QuzoState::PostPerturbUp;
    Ok(QuzoOut::Ack)
}

async fn handle_infer(input_id: ObjectId, ctx: &QuarkContext) -> Result<QuzoOut> {
    let state = {
        let session = ctx.quzo.lock().await;
        if session.state != QuzoState::PostPerturbUp && session.state != QuzoState::PostPerturbDown {
            return Err(ServerError::InvalidQuzoState(
                format!("inference requires PostPerturbUp or PostPerturbDown state, got {:?}", session.state),
            ));
        }
        session.state
    };

    // Resolve void client.
    let void = ctx.void_client.as_ref().ok_or_else(|| {
        ServerError::VoidNotConfigured
    })?;

    // Download input object from void.
    let input_bytes = void.download(&input_id).await?;

    // Decode the inference request (QuzoInferRequest -> ModelInput list).
    let infer_req: QuzoInferRequest =
        from_bytes(&input_bytes).map_err(ServerError::DecodeFrame)?;

    let inputs: Vec<paramecia_engine::ModelInput> = infer_req
        .inputs
        .into_iter()
        .map(|inp| match inp {
            black_hole_type::QuzoInferInput::Text(t) => {
                paramecia_engine::ModelInput::Text(t)
            }
            black_hole_type::QuzoInferInput::Tokens(ids) => {
                paramecia_engine::ModelInput::Tokens(ids)
            }
            black_hole_type::QuzoInferInput::Soft(entries) => {
                paramecia_engine::ModelInput::Soft(
                    entries.into_iter().map(|e| paramecia_engine::LogitEntry {
                        token_id: e.token_id,
                        log_prob: e.log_prob,
                    }).collect(),
                )
            }
        })
        .collect();

    // Run inference: fill context then predict completion.
    let predicted_tokens = run_inference(&ctx.engine, &inputs).await?;

    // Decode output text via tokenizer.
    let tokenizer = ctx.engine.tokenizer();
    let output_text = tokenizer.decode(&predicted_tokens, false)
        .map_err(|e| ServerError::ModelError(e.to_string()))?;

    // Upload output to void.
    let output_bytes = to_allocvec(&output_text).map_err(ServerError::EncodeFrame)?;
    let output_id = void.upload(output_bytes).await?;

    // Advance state.
    {
        let mut session = ctx.quzo.lock().await;
        session.state = if state == QuzoState::PostPerturbUp {
            QuzoState::AwaitingPerturbDown
        } else {
            QuzoState::AwaitingOptimize
        };
    }

    Ok(QuzoOut::Inferred { output_id })
}

async fn handle_perturb_down(ctx: &QuarkContext) -> Result<QuzoOut> {
    let mut session = ctx.quzo.lock().await;
    if session.state != QuzoState::AwaitingPerturbDown {
        return Err(ServerError::InvalidQuzoState(
            format!("expected AwaitingPerturbDown, got {:?}", session.state),
        ));
    }

    ctx.engine.perturb_down().await.map_err(|e| {
        ServerError::ModelError(e.to_string())
    })?;

    session.state = QuzoState::PostPerturbDown;
    Ok(QuzoOut::Ack)
}

async fn handle_optimize(loss_up: f32, loss_down: f32, ctx: &QuarkContext) -> Result<QuzoOut> {
    let mut session = ctx.quzo.lock().await;
    if session.state != QuzoState::AwaitingOptimize {
        return Err(ServerError::InvalidQuzoState(
            format!("expected AwaitingOptimize, got {:?}", session.state),
        ));
    }

    ctx.engine.update(loss_up, loss_down).await.map_err(|e| {
        ServerError::ModelError(e.to_string())
    })?;

    session.state = QuzoState::Idle;
    Ok(QuzoOut::Ack)
}

// ---------------------------------------------------------------------------
// Inference helper
// ---------------------------------------------------------------------------

async fn run_inference(
    engine: &ModelEngine,
    inputs: &[paramecia_engine::ModelInput],
) -> Result<Vec<u32>> {
    // Fill context with the provided inputs. Returns a progress receiver.
    let _progress_rx = engine.fill_context_inputs(inputs).await
        .map_err(|e| ServerError::ModelError(e.to_string()))?;

    // Start streaming completion — returns (result_rx, cancel_tx).
    let (mut result_rx, _cancel_tx) = engine.predict_completion().await
        .map_err(|e| ServerError::ModelError(e.to_string()))?;

    let mut tokens = Vec::new();
    while let Some(result) = result_rx.recv().await {
        match result {
            Ok(predicted) => tokens.push(predicted.token_id),
            Err(e) => {
                // Non-fatal errors (e.g. max length) are fine.
                warn!(error = %e, "prediction ended with error");
                break;
            }
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Certificate helpers
// ---------------------------------------------------------------------------

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
    let dirs = directories_next::ProjectDirs::from("org", "blackhole", "quark").unwrap();
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

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

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
    #[error("model error: {0}")]
    ModelError(String),
    #[error("invalid QuZO state machine transition: {0}")]
    InvalidQuzoState(String),
    #[error("void service not configured")]
    VoidNotConfigured,
    #[error("failed to bind void client endpoint: {0}")]
    BindVoidClient(#[source] io::Error),
    #[error("failed to connect to void: {0}")]
    VoidConnect(String),
    #[error("failed to open void stream: {0}")]
    VoidStream(String),
    #[error("void crypto config error: {0}")]
    VoidCrypto(String),
    #[error("void error: {0}")]
    VoidError(String),
    #[error("unexpected EOF while reading frame length")]
    UnexpectedEof,
    #[error("failed to read frame length: {0}")]
    ReadFrameLength(#[source] io::Error),
    #[error("frame payload too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("failed to read frame payload: {0}")]
    ReadFramePayload(#[source] quinn::ReadExactError),
    #[error("failed to decode frame: {0}")]
    DecodeFrame(postcard::Error),
    #[error("failed to encode frame: {0}")]
    EncodeFrame(postcard::Error),
    #[error("failed to write frame: {0}")]
    WriteFrame(quinn::WriteError),
}

pub type Result<T> = std::result::Result<T, ServerError>;

pub fn init_tracing() -> std::result::Result<(), tracing::subscriber::SetGlobalDefaultError> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish(),
    )
}
