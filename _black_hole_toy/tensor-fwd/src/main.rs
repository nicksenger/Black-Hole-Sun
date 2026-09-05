use std::sync::atomic::Ordering;
use std::time::Duration;

use black_hole_sun::{MassClient, VoidClient};
use jungle_sdk::FusedClient;
use tensor_fwd::spec::{MatmulForward, LOGGED_OUTPUTS};
use tensor_fwd::flow::{TensorJungle, forward_reset_capabilities};
use tensor_fwd::op::{MatmulOperation, ReluOperation, ScaleOperation};
use toy_common::runtime::{RunCheck, ServerSpecs, run_until};

const TARGET_PASSES: usize = 4;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;

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
        matmul: MassClient::new_tcp_typed(servers.mass_addrs[0]).requiring(
            forward_reset_capabilities(),
        ),
        scale: MassClient::new_tcp_typed(servers.mass_addrs[1]).requiring(
            forward_reset_capabilities(),
        ),
        relu: MassClient::new_tcp_typed(servers.mass_addrs[2]).requiring(
            forward_reset_capabilities(),
        ),
    };

    let result = tokio::time::timeout(Duration::from_secs(20), async {
        run_until::<TensorJungle, MatmulForward, _, _>(&jungle, &client, &(), 4, |_| async {
            if LOGGED_OUTPUTS.load(Ordering::Acquire) >= TARGET_PASSES {
                RunCheck::Done
            } else {
                RunCheck::Continue
            }
        })
        .await
    })
    .await;

    servers.shutdown();

    result
        .map_err(|_| "timed out waiting for tensor-fwd output")?
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    println!(
        "tensor-fwd completed {} output pass(es)",
        LOGGED_OUTPUTS.load(Ordering::Acquire)
    );
    Ok(())
}
