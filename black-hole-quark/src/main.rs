use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[clap(name = "black-hole-quark")]
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
    #[clap(long = "listen", default_value = "[::1]:4433")]
    listen: SocketAddr,
    /// Path to the GGUF model file
    #[clap(long = "model")]
    model: PathBuf,
    /// Address of the void object store (e.g. [::1]:4434)
    #[clap(long = "void-addr")]
    void_addr: Option<SocketAddr>,
}

impl From<Opt> for black_hole_quark::ServerBuilder {
    fn from(opt: Opt) -> Self {
        let mut builder = black_hole_quark::ServerBuilder::new(&opt.model)
            .keylog(opt.keylog)
            .stateless_retry(opt.stateless_retry)
            .listen(opt.listen);

        if let Some(key) = opt.key {
            builder = builder.key(key);
        }
        if let Some(cert) = opt.cert {
            builder = builder.cert(cert);
        }
        if let Some(addr) = opt.void_addr {
            builder = builder.void_addr(addr);
        }

        builder
    }
}

fn main() {
    if black_hole_quark::init_tracing().is_err() {
        eprintln!("ERROR: failed to initialize tracing subscriber");
        std::process::exit(1);
    }

    let builder = black_hole_quark::ServerBuilder::from(Opt::parse());
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let code = if let Err(e) = rt.block_on(builder.run()) {
        eprintln!("ERROR: {e}");
        1
    } else {
        0
    };
    std::process::exit(code);
}
