use std::path::PathBuf;
use std::sync::atomic::Ordering;

use black_hole_beam::BeamBuilder;
use black_hole_sun::{MassClient, VoidClient};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use corgi_fwd::flow::{capabilities, CorgiJungle};
use corgi_fwd::model::{
    build_head, build_stage1, build_stage2, build_stage3, build_stage4, build_stem,
};
use corgi_fwd::op::{
    HeadOperation, ModelOperation, Stage1Operation, Stage2Operation, Stage3Operation,
    Stage4Operation, StemOperation,
};
use corgi_fwd::spec::{CorgiForward, LOGGED_OUTPUTS};
use corgi_fwd::spec::{HeadCell, Stage1Cell, Stage2Cell, Stage3Cell, Stage4Cell, StemCell};
use jungle_sdk::FusedClient;
use toy_common::dataset::{configure_hf_cache, model_path, DATASET_SAMPLES};
use toy_common::runtime::{launch, run_until, RunCheck, RunningServers, ServerSpecs};

#[derive(Debug, Parser)]
#[command(about = "Run ResNet-18 over Stanford Dogs images")]
struct Args {
    /// Number of dataset images to process before exiting.
    #[arg(long, default_value_t = 10)]
    n_samples: usize,

    /// Open a live Black Hole Beam viewer of the pipeline instead of running
    /// until `n-samples` images are processed; the journey keeps running
    /// until the window is closed.
    #[arg(long)]
    beam: bool,

    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Writable Hugging Face cache directory for model and dataset files.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn builder(
    path: &PathBuf,
    device: &Device,
) -> Result<VarBuilder<'static>, Box<dyn std::error::Error>> {
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? })
}

/// Start the void + mass servers, the fused jungle client, and the corgi
/// jungle wired over them.
async fn setup(
    args: &Args,
) -> Result<(RunningServers, FusedClient, CorgiJungle), Box<dyn std::error::Error>> {
    let cache_dir = configure_hf_cache(args.cache_dir.clone(), "corgi-fwd")?;
    eprintln!("using Hugging Face cache {}", cache_dir.display());
    let path = model_path(args.model.clone())?;
    let device = Device::Cpu;

    let servers = ServerSpecs::new()
        .operation(StemOperation(ModelOperation::new(
            build_stem(builder(&path, &device)?)?,
            device.clone(),
        )))
        .operation(Stage1Operation(ModelOperation::new(
            build_stage1(builder(&path, &device)?)?,
            device.clone(),
        )))
        .operation(Stage2Operation(ModelOperation::new(
            build_stage2(builder(&path, &device)?)?,
            device.clone(),
        )))
        .operation(Stage3Operation(ModelOperation::new(
            build_stage3(builder(&path, &device)?)?,
            device.clone(),
        )))
        .operation(Stage4Operation(ModelOperation::new(
            build_stage4(builder(&path, &device)?)?,
            device.clone(),
        )))
        .operation(HeadOperation(ModelOperation::new(
            build_head(builder(&path, &device)?)?,
            device.clone(),
        )))
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

    Ok((servers, client, jungle))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if !args.beam && args.n_samples == 0 {
        return Ok(());
    }

    // The runtime hosts the servers, client, and worker pool. In beam mode it
    // keeps running in the background while the Black Hole Beam event loop
    // blocks the main thread; dropping it tears everything down.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (servers, client, jungle) = runtime.block_on(setup(&args))?;

    if args.beam {
        // Keep a worker available for the parent and every one of the six
        // operation cells; one more runner per additional cell.
        let (parent, workers) =
            launch::<CorgiJungle, CorgiForward>(&runtime, &jungle, &client, &(), 8)?;
        println!(
            "Spawned journey {}. Opening Black Hole Beam.",
            parent.journey_id
        );

        let beam = BeamBuilder::new()
            .title("Black Hole Sun: corgi-fwd")
            .register_subpanel_animal::<StemCell>()
            .register_subpanel_animal::<Stage1Cell>()
            .register_subpanel_animal::<Stage2Cell>()
            .register_subpanel_animal::<Stage3Cell>()
            .register_subpanel_animal::<Stage4Cell>()
            .register_subpanel_animal::<HeadCell>()
            .microdot_layout();
        beam.view_live::<CorgiForward>(client, parent.journey_id)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;

        for worker in workers {
            worker.abort();
        }
    } else {
        // Keep a worker available for the parent and every one of the six
        // operation cells; one more runner per additional cell.
        let result = runtime.block_on(run_until::<CorgiJungle, CorgiForward, _, _>(
            &jungle,
            &client,
            &(),
            8,
            |_| async {
                if LOGGED_OUTPUTS.load(Ordering::Acquire) >= args.n_samples {
                    RunCheck::Done
                } else {
                    RunCheck::Continue
                }
            },
        ));

        let processed = LOGGED_OUTPUTS.load(Ordering::Acquire);
        println!(
            "corgi-fwd processed {} sample(s) (source dataset contains {DATASET_SAMPLES})",
            processed
        );
        result.map(|_| ())?;
    }

    servers.shutdown();
    Ok(())
}
