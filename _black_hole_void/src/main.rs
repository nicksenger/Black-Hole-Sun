use std::{net::SocketAddr, path::PathBuf, sync::Once};

#[cfg(feature = "postgres")]
use aws_sdk_s3::Client as S3Client;
use clap::Parser;

#[derive(Debug, Clone, Copy)]
enum StoreMode {
    Memory,
    Filesystem,
    #[cfg(feature = "postgres")]
    S3Postgres,
}

#[derive(Parser, Debug, Clone)]
#[clap(name = "black-hole-void")]
struct Opt {
    /// File to log TLS keys to for debugging
    #[clap(long = "keylog")]
    keylog: bool,
    /// TLS private key in PEM or DER format
    #[clap(short = 'k', long = "key", requires = "cert")]
    key: Option<PathBuf>,
    /// TLS certificate in PEM or DER format
    #[clap(short = 'c', long = "cert", requires = "key")]
    cert: Option<PathBuf>,
    /// Enable stateless retries
    #[clap(long = "stateless-retry")]
    stateless_retry: bool,
    /// Address to listen on
    #[clap(long = "listen", default_value = "[::1]:4434")]
    listen: SocketAddr,
    /// Use TCP transport instead of QUIC
    #[clap(long = "tcp")]
    tcp: bool,
    /// Use in-memory storage for both objects and metadata (no S3 or PostgreSQL needed)
    #[clap(long = "memory-store", conflicts_with = "filesystem_store")]
    memory_store: bool,
    /// Use filesystem object storage and fjall metadata persistence.
    #[clap(long = "filesystem-store", conflicts_with = "memory_store")]
    filesystem_store: bool,
    /// S3 bucket name
    #[clap(long = "bucket")]
    bucket: Option<String>,
    /// S3 endpoint URL (for non-AWS S3-compatible storage)
    #[clap(long = "endpoint")]
    endpoint: Option<String>,
    /// AWS region
    #[clap(long = "region", default_value = "us-east-1")]
    region: String,
    /// PostgreSQL connection string
    #[cfg(feature = "postgres")]
    #[clap(long = "postgres-connection-string")]
    postgres_connection_string: Option<String>,
}

#[cfg(feature = "postgres")]
async fn build_s3_client(opt: &Opt) -> S3Client {
    let mut builder = aws_sdk_s3::config::Builder::new()
        .region(aws_sdk_s3::config::Region::new(opt.region.clone()));

    if let Some(endpoint) = &opt.endpoint {
        builder = builder.endpoint_url(endpoint.clone());
    }

    S3Client::from_conf(builder.build())
}

fn install_rustls_crypto_provider() {
    static RUSTLS_PROVIDER: Once = Once::new();
    RUSTLS_PROVIDER.call_once(|| {
        // rustls may be built with multiple crypto backends enabled via transitive
        // features; explicitly selecting ring avoids runtime provider ambiguity.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn default_filesystem_store_paths() -> (PathBuf, PathBuf) {
    let home = directories_next::BaseDirs::new()
        .map(|dirs| dirs.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let root = home.join(".black-hole-sun").join("void");
    (root.join("objects"), root.join("relations"))
}

fn apply_common_server_options(
    mut builder: black_hole_void::ServerBuilder,
    opt: &Opt,
) -> black_hole_void::ServerBuilder {
    builder = builder
        .keylog(opt.keylog)
        .stateless_retry(opt.stateless_retry)
        .listen(opt.listen);
    if opt.tcp {
        builder = builder.tcp();
    }

    if let Some(key) = opt.key.clone() {
        builder = builder.key(key);
    }
    if let Some(cert) = opt.cert.clone() {
        builder = builder.cert(cert);
    }
    builder
}

fn main() {
    install_rustls_crypto_provider();

    if black_hole_void::init_tracing().is_err() {
        eprintln!("ERROR: failed to initialize tracing subscriber");
        std::process::exit(1);
    }

    let opt = Opt::parse();
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let mode = if opt.memory_store {
        StoreMode::Memory
    } else if opt.filesystem_store {
        StoreMode::Filesystem
    } else {
        #[cfg(feature = "postgres")]
        {
            StoreMode::S3Postgres
        }
        #[cfg(not(feature = "postgres"))]
        {
            StoreMode::Memory
        }
    };

    let code = match mode {
        StoreMode::Memory => {
            let object_store: Box<dyn black_hole_void::object_store::ObjectStore> =
                Box::new(black_hole_void::object_store::InMemoryObjectStore::new());
            let store: Box<dyn black_hole_void::persist::VoidStore> =
                Box::new(black_hole_void::persist::InMemoryStore::new());

            let builder = apply_common_server_options(
                black_hole_void::ServerBuilder::new(object_store, store).object_namespace("memory"),
                &opt,
            );
            if let Err(e) = rt.block_on(builder.run()) {
                eprintln!("ERROR: {e}");
                1
            } else {
                0
            }
        }
        StoreMode::Filesystem => {
            let (objects_path, relations_path) = default_filesystem_store_paths();
            let object_store: Box<dyn black_hole_void::object_store::ObjectStore> =
                match black_hole_void::object_store::FilesystemObjectStore::new(&objects_path) {
                    Ok(store) => Box::new(store),
                    Err(e) => {
                        eprintln!("ERROR: failed to initialize filesystem object store: {e}");
                        std::process::exit(1);
                    }
                };
            let store: Box<dyn black_hole_void::persist::VoidStore> =
                match black_hole_void::persist::fjall::FjallStore::new(&relations_path) {
                    Ok(store) => Box::new(store),
                    Err(e) => {
                        eprintln!("ERROR: failed to initialize fjall relation store: {e}");
                        std::process::exit(1);
                    }
                };

            let builder = apply_common_server_options(
                black_hole_void::ServerBuilder::new(object_store, store)
                    .object_namespace("filesystem"),
                &opt,
            );
            if let Err(e) = rt.block_on(builder.run()) {
                eprintln!("ERROR: {e}");
                1
            } else {
                0
            }
        }
        #[cfg(feature = "postgres")]
        StoreMode::S3Postgres => {
            let bucket = opt.bucket.clone().unwrap_or_else(|| {
                eprintln!(
                    "ERROR: --bucket is required unless --memory-store or --filesystem-store is used"
                );
                std::process::exit(1);
            });

            let s3_client = rt.block_on(build_s3_client(&opt));

            let store: Box<dyn black_hole_void::persist::VoidStore> = {
                let connection_string =
                    opt.postgres_connection_string.clone().unwrap_or_else(|| {
                        eprintln!("ERROR: --postgres-connection-string is required");
                        std::process::exit(1);
                    });
                let store = rt
                    .block_on(
                        black_hole_void::persist::pg::PgStore::builder()
                            .connection_string(connection_string)
                            .build(),
                    )
                    .expect("failed to build postgres store");
                Box::new(store)
            };

            let object_store: Box<dyn black_hole_void::object_store::ObjectStore> = Box::new(
                black_hole_void::object_store::S3Store::new(s3_client, &bucket),
            );

            let builder = apply_common_server_options(
                black_hole_void::ServerBuilder::new(object_store, store)
                    .object_namespace(bucket.clone()),
                &opt,
            );
            if let Err(e) = rt.block_on(builder.run()) {
                eprintln!("ERROR: {e}");
                1
            } else {
                0
            }
        }
    };

    std::process::exit(code);
}
