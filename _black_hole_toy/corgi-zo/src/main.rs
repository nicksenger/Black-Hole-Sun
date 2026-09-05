use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use black_hole_sun::{MassClient, VoidClient};
use candle::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use corgi_fwd::model::{
    build_head, build_stage1, build_stage2, build_stage3, build_stage4, build_stem,
};
use corgi_zo::contracts::CorgiZo;
use corgi_zo::jungle::{CorgiJungle, capabilities};
use corgi_zo::operations::{
    HeadModel, HeadOperation, ModelOperation, Stage1Operation, Stage2Operation, Stage3Operation,
    Stage4Operation, StemOperation,
};
use jungle_sdk::FusedClient;
use toy_common::dataset::{DATASET_SAMPLES, configure_hf_cache, model_path};
use toy_common::runtime::{RunCheck, ServerSpecs, run_until};

#[derive(Debug, Parser)]
#[command(about = "Run two-sided zeroth-order optimization on ResNet-18")]
struct Args {
    /// Number of ZO steps to run before exiting.
    #[arg(long, default_value_t = 10)]
    steps: usize,

    /// Save model checkpoints every N completed ZO steps; disabled by default.
    #[arg(long, default_value_t = 0)]
    checkpoint_steps: usize,

    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Writable Hugging Face cache directory for model and dataset files.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn mutable_stage<C, F>(
    path: &Path,
    device: &Device,
    build: F,
) -> Result<ModelOperation<C, candle_nn::Func<'static>>, Box<dyn std::error::Error>>
where
    C: black_hole_sun::TensorContract<Metadata = toy_common::dataset::SampleMetadata>,
    F: FnOnce(VarBuilder<'_>) -> candle::Result<candle_nn::Func<'static>>,
{
    let mut varmap = VarMap::new();
    let model = build(VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    varmap.load(path)?;
    Ok(ModelOperation::new(model, &varmap, device.clone()))
}

fn mutable_head(
    path: &Path,
    device: &Device,
) -> Result<ModelOperation<corgi_fwd::contracts::HeadOp, HeadModel>, Box<dyn std::error::Error>> {
    let source = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let original = build_head(source)?;
    let mut varmap = VarMap::new();
    let model = candle_nn::linear(512, 2, VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    varmap.set_one("weight", original.weight())?;
    varmap.set_one("bias", original.bias().expect("corgi head has a bias"))?;
    Ok(ModelOperation::new(
        HeadModel(model),
        &varmap,
        device.clone(),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if args.steps == 0 {
        return Ok(());
    }
    let checkpoint_dir = if args.checkpoint_steps == 0 {
        None
    } else {
        Some(tempfile::tempdir()?)
    };
    black_hole_sun::configure_checkpointing(
        args.checkpoint_steps,
        checkpoint_dir.as_ref().map(|dir| dir.path().to_path_buf()),
    );
    if let Some(dir) = &checkpoint_dir {
        eprintln!("saving checkpoints to {}", dir.path().display());
    }
    let cache = configure_hf_cache(args.cache_dir, "corgi-zo")?;
    eprintln!("using Hugging Face cache {}", cache.display());
    let path = model_path(args.model)?;
    let device = Device::Cpu;

    let servers = ServerSpecs::new()
        .operation(StemOperation(mutable_stage(&path, &device, build_stem)?))
        .operation(Stage1Operation(mutable_stage(&path, &device, build_stage1)?))
        .operation(Stage2Operation(mutable_stage(&path, &device, build_stage2)?))
        .operation(Stage3Operation(mutable_stage(&path, &device, build_stage3)?))
        .operation(Stage4Operation(mutable_stage(&path, &device, build_stage4)?))
        .operation(HeadOperation(mutable_head(&path, &device)?))
        .start()
        .await?;

    let client = FusedClient::builder().build().await?;
    let jungle = CorgiJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(servers.void_addr),
        stem: MassClient::new_tcp_typed(servers.mass_addrs[0]).requiring(capabilities()),
        stage1: MassClient::new_tcp_typed(servers.mass_addrs[1]).requiring(capabilities()),
        stage2: MassClient::new_tcp_typed(servers.mass_addrs[2]).requiring(capabilities()),
        stage3: MassClient::new_tcp_typed(servers.mass_addrs[3]).requiring(capabilities()),
        stage4: MassClient::new_tcp_typed(servers.mass_addrs[4]).requiring(capabilities()),
        head: MassClient::new_tcp_typed(servers.mass_addrs[5]).requiring(capabilities()),
    };

    let result = run_until::<CorgiJungle, CorgiZo, _, _>(&jungle, &client, &(), 8, |_| {
        async {
            if corgi_zo::OPTIMIZED_STEPS.load(Ordering::Acquire) >= args.steps {
                RunCheck::Done
            } else {
                RunCheck::Continue
            }
        }
    })
    .await;

    servers.shutdown();

    println!(
        "corgi-zo completed {} step(s) (source dataset contains {DATASET_SAMPLES})",
        args.steps
    );
    result
        .map(|_| ())
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
}
