use std::sync::atomic::Ordering;
use std::time::Duration;

use black_hole_beam::BeamBuilder;
use black_hole_sun::{MassClient, VoidClient};
use clap::Parser;
use jungle_sdk::FusedClient;
use tensor_fwd::flow::{forward_reset_capabilities, TensorJungle};
use tensor_fwd::op::{MatmulOperation, ReluOperation, ScaleOperation};
use tensor_fwd::spec::{MatmulCell, MatmulForward, ReluCell, ScaleCell, LOGGED_OUTPUTS};
use toy_common::runtime::{launch, run_until, RunCheck, RunningServers, ServerSpecs};

const TARGET_PASSES: usize = 4;

#[derive(Debug, Parser)]
#[command(about = "Run the small forward-only tensor pipeline")]
struct Args {
    /// Open a live Black Hole Beam viewer of the pipeline instead of running
    /// until four output passes complete; the journey keeps running until
    /// the window is closed.
    #[arg(long)]
    beam: bool,
}

/// Start the void + mass servers, the fused jungle client, and the tensor
/// jungle wired over them.
async fn setup() -> Result<(RunningServers, FusedClient, TensorJungle), Box<dyn std::error::Error>>
{
    let servers = ServerSpecs::new()
        .operation(MatmulOperation)
        .operation(ScaleOperation)
        .operation(ReluOperation)
        .start()
        .await?;

    let client = FusedClient::builder().build().await?;
    let jungle = TensorJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(servers.void_addr),
        matmul: MassClient::new_tcp_typed(servers.mass_addrs[0])
            .requiring(forward_reset_capabilities()),
        scale: MassClient::new_tcp_typed(servers.mass_addrs[1])
            .requiring(forward_reset_capabilities()),
        relu: MassClient::new_tcp_typed(servers.mass_addrs[2])
            .requiring(forward_reset_capabilities()),
    };

    Ok((servers, client, jungle))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();

    // The runtime hosts the servers, client, and worker pool. In beam mode it
    // keeps running in the background while the Black Hole Beam event loop
    // blocks the main thread; dropping it tears everything down.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (servers, client, jungle) = runtime.block_on(setup())?;

    if args.beam {
        // Keep a worker available for the parent and every one of the three
        // operation cells.
        let (parent, workers) =
            launch::<TensorJungle, MatmulForward>(&runtime, &jungle, &client, &(), 4)?;
        println!(
            "Spawned journey {}. Opening Black Hole Beam.",
            parent.journey_id
        );

        let beam = BeamBuilder::new()
            .title("Black Hole Sun: tensor-fwd")
            .register_subpanel_animal::<MatmulCell>()
            .register_subpanel_animal::<ScaleCell>()
            .register_subpanel_animal::<ReluCell>()
            .microdot_layout();
        beam.view_live::<MatmulForward>(client, parent.journey_id)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;

        for worker in workers {
            worker.abort();
        }
    } else {
        let result = runtime.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(20),
                run_until::<TensorJungle, MatmulForward, _, _>(
                    &jungle,
                    &client,
                    &(),
                    4,
                    |_| async {
                        if LOGGED_OUTPUTS.load(Ordering::Acquire) >= TARGET_PASSES {
                            RunCheck::Done
                        } else {
                            RunCheck::Continue
                        }
                    },
                ),
            )
            .await
        });

        result
            .map_err(|_| "timed out waiting for tensor-fwd output")?
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        println!(
            "tensor-fwd completed {} output pass(es)",
            LOGGED_OUTPUTS.load(Ordering::Acquire)
        );
    }

    servers.shutdown();
    Ok(())
}
