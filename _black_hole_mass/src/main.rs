use std::{net::SocketAddr, path::PathBuf, sync::Once, time::Duration};

use black_hole_spec::MassPerturbationMode;
use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[clap(name = "black-hole-mass")]
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
    /// Disable weight mutations (perturb and optimize become no-ops)
    #[clap(long = "frozen")]
    frozen: bool,
    /// Address to listen on
    #[clap(long = "listen", default_value = "[::1]:4433")]
    listen: SocketAddr,
    /// Use TCP transport instead of QUIC
    #[clap(long = "tcp")]
    tcp: bool,
    /// Path to the GGUF model file
    #[clap(long = "model")]
    model: PathBuf,
    /// Address of the void object store (e.g. [::1]:4434). Required for inference/checkpoint, including tunnel workers.
    #[clap(long = "void-addr")]
    void_addr: Option<SocketAddr>,
    /// Default top-k sampler value for model instances
    #[clap(long = "top-k")]
    top_k: Option<usize>,
    /// Default temperature for model instances (0 => greedy)
    #[clap(long = "temperature")]
    temperature: Option<f64>,
    /// Optional top-p sampler value for model instances
    #[clap(long = "top-p")]
    top_p: Option<f64>,
    /// Default repeat penalty for model instances
    #[clap(long = "repeat-penalty")]
    repeat_penalty: Option<f32>,
    /// Default presence penalty for model instances
    #[clap(long = "presence-penalty")]
    presence_penalty: Option<f32>,
    /// Default max generation length when request limit is omitted
    #[clap(long = "default-inference-limit")]
    default_inference_limit: Option<u32>,
    /// Default QuZO learning rate for model instances
    #[clap(long = "training-lr")]
    training_lr: Option<f64>,
    /// Default QuZO epsilon for model instances
    #[clap(long = "training-epsilon")]
    training_epsilon: Option<f64>,
    /// Default QuZO perturbation direction mode for model instances
    #[clap(long = "training-perturbation-mode", value_parser = ["weight", "low-rank"])]
    training_perturbation_mode: Option<String>,
    /// Number of factors used by low-rank perturbations (used when the mode is low-rank)
    #[clap(long = "training-perturbation-rank", default_value_t = 1)]
    training_perturbation_rank: usize,
    /// Optional root mass endpoint to register with as a tunnel worker
    #[clap(long = "tunnel")]
    tunnel: Option<SocketAddr>,
    /// Max seconds to keep retrying tunnel registration before exiting (default: no deadline)
    #[clap(long = "tunnel-connect-deadline-seconds")]
    tunnel_connect_deadline_seconds: Option<u64>,
    /// Optional max concurrent model instances handled by this mass (defaults to 1)
    #[clap(long = "max-instances")]
    max_instances: Option<usize>,
}

impl From<Opt> for black_hole_mass::ServerBuilder {
    fn from(opt: Opt) -> Self {
        let mut builder = black_hole_mass::ServerBuilder::new(&opt.model)
            .keylog(opt.keylog)
            .stateless_retry(opt.stateless_retry)
            .listen(opt.listen);
        if opt.tcp {
            builder = builder.tcp();
        }
        if opt.frozen {
            builder = builder.frozen();
        }

        if let Some(key) = opt.key {
            builder = builder.key(key);
        }
        if let Some(cert) = opt.cert {
            builder = builder.cert(cert);
        }
        if let Some(addr) = opt.void_addr {
            builder = builder.void_addr(addr);
        }
        if let Some(top_k) = opt.top_k {
            builder = builder.top_k(top_k);
        }
        if let Some(temperature) = opt.temperature {
            builder = builder.temperature(temperature);
        }
        if let Some(top_p) = opt.top_p {
            builder = builder.top_p(Some(top_p));
        }
        if let Some(repeat_penalty) = opt.repeat_penalty {
            builder = builder.repeat_penalty(repeat_penalty);
        }
        if let Some(presence_penalty) = opt.presence_penalty {
            builder = builder.presence_penalty(presence_penalty);
        }
        if let Some(limit) = opt.default_inference_limit {
            builder = builder.default_inference_limit(limit);
        }
        if let Some(lr) = opt.training_lr {
            builder = builder.training_lr(lr);
        }
        if let Some(epsilon) = opt.training_epsilon {
            builder = builder.training_epsilon(epsilon);
        }
        if let Some(mode) = opt.training_perturbation_mode {
            let mode = match mode.as_str() {
                "weight" => MassPerturbationMode::Weight,
                "low-rank" => MassPerturbationMode::LowRank(opt.training_perturbation_rank),
                _ => unreachable!("clap validates perturbation mode values"),
            };
            builder = builder.training_perturbation_mode(mode);
        }
        if let Some(tunnel) = opt.tunnel {
            builder = builder.tunnel(tunnel);
        }
        if let Some(seconds) = opt.tunnel_connect_deadline_seconds {
            builder = builder.tunnel_connect_deadline(Duration::from_secs(seconds));
        }
        if let Some(max_instances) = opt.max_instances {
            builder = builder.max_instances(max_instances);
        }

        builder
    }
}

fn install_rustls_crypto_provider() {
    static RUSTLS_PROVIDER: Once = Once::new();
    RUSTLS_PROVIDER.call_once(|| {
        // rustls may be built with multiple crypto backends enabled via transitive
        // features; explicitly selecting ring avoids runtime provider ambiguity.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn main() {
    install_rustls_crypto_provider();

    if black_hole_mass::init_tracing().is_err() {
        eprintln!("ERROR: failed to initialize tracing subscriber");
        std::process::exit(1);
    }

    let opt = Opt::parse();
    if opt.training_perturbation_mode.as_deref() == Some("low-rank")
        && opt.training_perturbation_rank == 0
    {
        eprintln!("ERROR: --training-perturbation-rank must be greater than 0 in low-rank mode");
        std::process::exit(1);
    }
    let builder = black_hole_mass::ServerBuilder::from(opt);
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let code = if let Err(e) = rt.block_on(builder.run()) {
        eprintln!("ERROR: {e}");
        1
    } else {
        0
    };
    std::process::exit(code);
}
