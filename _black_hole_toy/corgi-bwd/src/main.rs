use std::path::Path;
use std::sync::atomic::Ordering;

use black_hole_sun::{MassClient, VoidClient};
use candle::{DType, Device, Var};
use candle_nn::{Module, VarBuilder, VarMap};
use clap::Parser;
use corgi_bwd::contracts::{CorgiBackward, COMPLETED_EPOCHS, MICRO_BATCHES};
use corgi_bwd::jungle::{CorgiJungle, required_capabilities};
use corgi_bwd::operations::{
    HeadOperation, OptimizerConfig, Stage1Operation, Stage2Operation, Stage3Operation,
    Stage4Operation, StemOperation, TrainOperation,
};
use corgi_fwd::model::{
    build_stage1, build_stage2, build_stage3, build_stage4, build_trainable_stem, pool_stage4,
};
use jungle_sdk::{FusedClient, JourneyStatus, JungleClient};
use toy_common::dataset::{DATASET_SAMPLES, configure_hf_cache, model_path};
use toy_common::runtime::{RunCheck, ServerSpecs, run_until};

#[derive(Debug, Parser)]
#[command(about = "Train a pipeline-parallel ResNet-18 corgi identifier")]
struct Args {
    /// Number of optimizer steps (each consumes eight image micro-batches).
    #[arg(long, default_value_t = 1)]
    epochs: usize,
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

fn build_trainable<M>(
    path: &Path,
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, Vec<Var>)> {
    let mut variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    variables.load(path)?;
    Ok((model, variables.all_vars()))
}

fn build_random<M>(
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, Vec<Var>)> {
    let variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    Ok((model, variables.all_vars()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if args.epochs == 0 {
        return Ok(());
    }
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
        let linear = candle_nn::linear(512, 2, vb.pp("binary_head"))?;
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
            async move {
                if COMPLETED_EPOCHS.load(Ordering::Acquire) >= args.epochs {
                    return RunCheck::Done;
                }
                match client.journey_details(journey_id).await {
                    Ok(JourneyStatus::Dead | JourneyStatus::Stopped | JourneyStatus::Completed) =>
                        RunCheck::Failed(
                            "corgi-bwd pipeline stopped before completing the requested epochs"
                                .to_owned(),
                        ),
                    Ok(_) => RunCheck::Continue,
                    Err(error) => RunCheck::Failed(error.to_string()),
                }
            }
        },
    )
    .await;

    servers.shutdown();

    println!(
        "corgi-bwd completed {} optimizer step(s), {} micro-batches each (dataset contains {DATASET_SAMPLES})",
        args.epochs, MICRO_BATCHES
    );
    result
        .map(|_| ())
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
}
