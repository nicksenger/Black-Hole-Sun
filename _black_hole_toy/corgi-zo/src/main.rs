use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use black_hole_sun::{MassClient, VoidClient};
use candle::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use corgi_fwd::model::{
    build_head, build_stage1, build_stage2, build_stage3, build_stage4, build_stem,
};
use corgi_zo::spec::CorgiZo;
use corgi_zo::flow::{capabilities, CorgiJungle};
use corgi_zo::op::{
    HeadModel, HeadOperation, ModelOperation, Stage1Operation, Stage2Operation, Stage3Operation,
    Stage4Operation, StemOperation,
};
use jungle_sdk::FusedClient;
use toy_common::dataset::{configure_hf_cache, model_path, DATASET_SAMPLES};
use toy_common::runtime::{run_until, RunCheck, ServerSpecs};

#[derive(Debug, Parser)]
#[command(about = "Run two-sided zeroth-order optimization on ResNet-18")]
struct Args {
    /// Number of ZO steps to run before exiting.
    #[arg(long, default_value_t = 10)]
    steps: usize,

    /// Save model checkpoints every N completed ZO steps; disabled by default.
    #[arg(long, default_value_t = 0)]
    checkpoint_steps: usize,

    /// Save unified model checkpoints by default, or one checkpoint per stage.
    #[arg(long, value_enum, default_value_t = CheckpointMode::Unified)]
    checkpoint_mode: CheckpointMode,

    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Writable Hugging Face cache directory for model and dataset files.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CheckpointMode {
    Unified,
    Sharded,
}

const OPERATION_COUNT: usize = 6;

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
) -> Result<ModelOperation<corgi_fwd::spec::HeadOp, HeadModel>, Box<dyn std::error::Error>> {
    let source = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let original = build_head(source)?;
    let mut varmap = VarMap::new();
    let model = candle_nn::linear(
        512,
        2,
        VarBuilder::from_varmap(&varmap, DType::F32, device).pp("fc"),
    )?;
    varmap.set_one("fc.weight", original.weight())?;
    varmap.set_one("fc.bias", original.bias().expect("corgi head has a bias"))?;
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
    let checkpoint_path = checkpoint_dir.as_ref().map(|dir| dir.path().to_path_buf());
    corgi_zo::spec::configure_unified_checkpointing(
        if matches!(args.checkpoint_mode, CheckpointMode::Unified) {
            args.checkpoint_steps
        } else {
            0
        },
        matches!(args.checkpoint_mode, CheckpointMode::Unified)
            .then(|| checkpoint_dir.as_ref().map(|dir| dir.path().to_path_buf()))
            .flatten(),
    );
    let cache = configure_hf_cache(args.cache_dir, "corgi-zo")?;
    eprintln!("using Hugging Face cache {}", cache.display());
    let path = model_path(args.model)?;
    let device = Device::Cpu;

    let servers = ServerSpecs::new()
        .operation(StemOperation(mutable_stage(&path, &device, build_stem)?))
        .operation(Stage1Operation(mutable_stage(
            &path,
            &device,
            build_stage1,
        )?))
        .operation(Stage2Operation(mutable_stage(
            &path,
            &device,
            build_stage2,
        )?))
        .operation(Stage3Operation(mutable_stage(
            &path,
            &device,
            build_stage3,
        )?))
        .operation(Stage4Operation(mutable_stage(
            &path,
            &device,
            build_stage4,
        )?))
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
        completed_optimization_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    let result = run_until::<CorgiJungle, CorgiZo, _, _>(&jungle, &client, &(), 8, |_| {
        let checkpoint_path = checkpoint_path.clone();
        async move {
            let optimized_steps = corgi_zo::OPTIMIZED_STEPS.load(Ordering::Acquire);
            if optimized_steps < args.steps {
                return RunCheck::Continue;
            }
            let expected_checkpoints = checkpoint_path
                .as_ref()
                .map(|_| {
                    let steps = args.steps / args.checkpoint_steps;
                    match args.checkpoint_mode {
                        CheckpointMode::Unified => steps,
                        CheckpointMode::Sharded => steps * OPERATION_COUNT,
                    }
                })
                .unwrap_or(0);
            let checkpoints_ready = checkpoint_path.as_ref().is_none_or(|path| {
                fs::read_dir(path)
                    .map(|entries| {
                        entries
                            .flatten()
                            .filter(|entry| {
                                entry.file_name().to_str().is_some_and(|name| {
                                    let Some(id) = name
                                        .strip_prefix("step-")
                                        .and_then(|name| name.strip_suffix(".checkpoint"))
                                    else {
                                        return false;
                                    };
                                    match args.checkpoint_mode {
                                        CheckpointMode::Unified => !id.contains('-'),
                                        CheckpointMode::Sharded => id.contains('-'),
                                    }
                                })
                            })
                            .count()
                            >= expected_checkpoints
                    })
                    .unwrap_or(false)
            });
            if checkpoints_ready {
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
