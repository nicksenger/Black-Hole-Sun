use std::{net::SocketAddr, path::PathBuf};

use aws_sdk_s3::Client as S3Client;
use clap::Parser;

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
    /// S3 bucket name
    #[clap(long = "bucket")]
    bucket: String,
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

async fn build_s3_client(opt: &Opt) -> S3Client {
    let mut builder = aws_sdk_s3::config::Builder::new()
        .region(aws_sdk_s3::config::Region::new(opt.region.clone()));

    if let Some(endpoint) = &opt.endpoint {
        builder = builder.endpoint_url(endpoint.clone());
    }

    S3Client::from_conf(builder.build())
}

fn main() {
    if black_hole_void::init_tracing().is_err() {
        eprintln!("ERROR: failed to initialize tracing subscriber");
        std::process::exit(1);
    }

    let opt = Opt::parse();
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let s3_client = rt.block_on(build_s3_client(&opt));

    #[cfg(feature = "postgres")]
    let store: Box<dyn black_hole_void::persist::VoidStore> = {
        let connection_string = opt.postgres_connection_string.unwrap_or_else(|| {
            eprintln!("ERROR: --postgres-connection-string is required");
            std::process::exit(1);
        });
        let store = rt.block_on(
            black_hole_void::persist::pg::PgStore::builder()
                .connection_string(connection_string)
                .build(),
        ).expect("failed to build postgres store");
        Box::new(store)
    };

    let mut builder = black_hole_void::ServerBuilder::new(s3_client, &opt.bucket, store)
        .keylog(opt.keylog)
        .stateless_retry(opt.stateless_retry)
        .listen(opt.listen);

    if let Some(key) = opt.key {
        builder = builder.key(key);
    }
    if let Some(cert) = opt.cert {
        builder = builder.cert(cert);
    }

    let code = if let Err(e) = rt.block_on(builder.run()) {
        eprintln!("ERROR: {e}");
        1
    } else {
        0
    };
    std::process::exit(code);
}
