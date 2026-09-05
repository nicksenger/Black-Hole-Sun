use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;

use black_hole_sun::{MassClient, VoidClient};
use candle::{DType, Device};
use candle_nn::{Module, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use corgi_bwd::contracts::{CorgiBackward, COMPLETED_STEPS, MICRO_BATCHES};
use corgi_bwd::jungle::{required_capabilities, CorgiJungle};
use corgi_bwd::operations::{
    HeadOperation, OptimizerConfig, Stage1Operation, Stage2Operation, Stage3Operation,
    Stage4Operation, StemOperation, TrainOperation,
};
use corgi_fwd::model::{
    build_stage1, build_stage2, build_stage3, build_stage4, build_trainable_stem, pool_stage4,
};
use jungle_sdk::{FusedClient, JourneyStatus, JungleClient};
use toy_common::dataset::{configure_hf_cache, model_path, BATCH_SIZE, DATASET_SAMPLES};
use toy_common::runtime::{run_until, RunCheck, ServerSpecs};

#[derive(Debug, Parser)]
#[command(about = "Train a pipeline-parallel ResNet-18 corgi identifier")]
struct Args {
    /// Number of optimizer steps (each consumes eight four-image micro-batches).
    #[arg(long, default_value_t = 1)]
    steps: usize,
    /// Save model checkpoints every N completed optimizer steps; disabled by default.
    #[arg(long, default_value_t = 0)]
    checkpoint_steps: usize,
    /// Save unified model checkpoints by default, or one checkpoint per stage.
    #[arg(long, value_enum, default_value_t = CheckpointMode::Unified)]
    checkpoint_mode: CheckpointMode,
    /// Learning rate used independently by every stage.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<std::path::PathBuf>,
    /// Writable Hugging Face cache directory.
    #[arg(long)]
    cache_dir: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CheckpointMode {
    Unified,
    Sharded,
}

const OPERATION_COUNT: usize = 6;

fn build_trainable<M>(
    path: &Path,
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, VarMap)> {
    let mut variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    variables.load(path)?;
    Ok((model, variables))
}

fn build_random<M>(
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, VarMap)> {
    let variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    Ok((model, variables))
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
    corgi_bwd::contracts::configure_unified_checkpointing(
        if matches!(args.checkpoint_mode, CheckpointMode::Unified) {
            args.checkpoint_steps
        } else {
            0
        },
        matches!(args.checkpoint_mode, CheckpointMode::Unified)
            .then(|| checkpoint_dir.as_ref().map(|dir| dir.path().to_path_buf()))
            .flatten(),
    );
    configure_hf_cache(args.cache_dir, "corgi-bwd")?;
    let path = model_path(args.model)?;
    let device = Device::Cpu;

    let optimizer_config = black_hole_sun::OperationConfig {
        encoding: black_hole_sun::EncodingId::POSTCARD_V1,
        data: postcard::to_allocvec(&OptimizerConfig {
            learning_rate: args.learning_rate,
        })?,
    };

    let (stem_model, stem_vars) = build_trainable(&path, &device, build_trainable_stem)?;
    let (stage1_model, stage1_vars) = build_trainable(&path, &device, build_stage1)?;
    let (stage2_model, stage2_vars) = build_trainable(&path, &device, build_stage2)?;
    let (stage3_model, stage3_vars) = build_trainable(&path, &device, build_stage3)?;
    let (stage4_model, stage4_vars) = build_trainable(&path, &device, build_stage4)?;
    let (head_model, head_vars) = build_random(&device, |vb| {
        let linear = candle_nn::linear(512, 2, vb.pp("fc"))?;
        Ok(candle_nn::Func::new(move |xs| {
            linear.forward(&pool_stage4(xs)?)
        }))
    })?;

    let servers = ServerSpecs::new()
        .operation(StemOperation(TrainOperation::new(
            stem_model,
            device.clone(),
            stem_vars,
            args.learning_rate,
        )?))
        .operation(Stage1Operation(TrainOperation::new(
            stage1_model,
            device.clone(),
            stage1_vars,
            args.learning_rate,
        )?))
        .operation(Stage2Operation(TrainOperation::new(
            stage2_model,
            device.clone(),
            stage2_vars,
            args.learning_rate,
        )?))
        .operation(Stage3Operation(TrainOperation::new(
            stage3_model,
            device.clone(),
            stage3_vars,
            args.learning_rate,
        )?))
        .operation(Stage4Operation(TrainOperation::new(
            stage4_model,
            device.clone(),
            stage4_vars,
            args.learning_rate,
        )?))
        .operation(HeadOperation(TrainOperation::new(
            head_model,
            device.clone(),
            head_vars,
            args.learning_rate,
        )?))
        .start()
        .await?;

    let client = FusedClient::builder().build().await?;
    let jungle = CorgiJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(servers.void_addr),
        stem: MassClient::new_tcp_typed(servers.mass_addrs[0]).requiring(required_capabilities()),
        stage1: MassClient::new_tcp_typed(servers.mass_addrs[1]).requiring(required_capabilities()),
        stage2: MassClient::new_tcp_typed(servers.mass_addrs[2]).requiring(required_capabilities()),
        stage3: MassClient::new_tcp_typed(servers.mass_addrs[3]).requiring(required_capabilities()),
        stage4: MassClient::new_tcp_typed(servers.mass_addrs[4]).requiring(required_capabilities()),
        head: MassClient::new_tcp_typed(servers.mass_addrs[5]).requiring(required_capabilities()),
        optimizer_config,
    };

    let result = run_until::<CorgiJungle, CorgiBackward<MICRO_BATCHES>, _, _>(
        &jungle,
        &client,
        &(),
        8,
        |handle| {
            let journey_id = handle.journey_id;
            let client = &client;
            let checkpoint_path = checkpoint_path.clone();
            async move {
                let completed_steps = COMPLETED_STEPS.load(Ordering::Acquire);
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
                if completed_steps >= args.steps && checkpoints_ready {
                    return RunCheck::Done;
                }
                match client.journey_details(journey_id).await {
                    Ok(JourneyStatus::Dead | JourneyStatus::Stopped | JourneyStatus::Completed) => {
                        RunCheck::Failed(
                            "corgi-bwd pipeline stopped before completing the requested steps"
                                .to_owned(),
                        )
                    }
                    Ok(_) => RunCheck::Continue,
                    Err(error) => RunCheck::Failed(error.to_string()),
                }
            }
        },
    )
    .await;

    servers.shutdown();

    println!(
        "corgi-bwd completed {} optimizer step(s), {} micro-batches of {} images each (source dataset contains {DATASET_SAMPLES})",
        args.steps, MICRO_BATCHES, BATCH_SIZE
    );
    result
        .map(|_| ())
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
}
