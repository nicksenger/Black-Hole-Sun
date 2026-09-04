//! Local in-process runtime for toy examples: void + mass servers, a worker
//! pool, and run-until-completion polling.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use black_hole_sun::{
    MassServerBuilder, OperationImplementation, VoidServerBuilder, object_store, persist,
};
use jungle_sdk::core::{JungleWorker, SupportedAnimalGenerations};
use jungle_sdk::prelude::*;
use jungle_sdk::typosaurus::collections::sp::{FlattenNodes, SPFlatten};
use jungle_sdk::{FusedClient, JourneyHandle};

const LOOPBACK: &str = "127.0.0.1:0";
const POLL_INTERVAL: Duration = Duration::from_millis(50);

enum ServerTask {
    Void(tokio::task::JoinHandle<Result<(), black_hole_sun::black_hole_void::ServerError>>),
    Mass(tokio::task::JoinHandle<Result<(), black_hole_sun::black_hole_mass::ServerError>>),
}

impl ServerTask {
    fn abort(self) {
        match self {
            Self::Void(task) => task.abort(),
            Self::Mass(task) => task.abort(),
        }
    }
}

/// In-process void server plus one mass server per registered operation.
pub struct RunningServers {
    /// Address of the in-memory void server.
    pub void_addr: SocketAddr,
    /// Addresses of the mass servers, in registration order.
    pub mass_addrs: Vec<SocketAddr>,
    tasks: Vec<ServerTask>,
}

impl RunningServers {
    /// Abort every server task.
    pub fn shutdown(self) {
        for task in self.tasks {
            task.abort();
        }
    }
}

/// Builder that registers operation implementations and starts local servers.
#[derive(Default)]
pub struct ServerSpecs {
    operations: Vec<Arc<dyn OperationImplementation>>,
}

impl ServerSpecs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one tensor operation for its own mass server.
    pub fn operation(mut self, operation: impl OperationImplementation + 'static) -> Self {
        self.operations.push(Arc::new(operation));
        self
    }

    /// Start an in-memory void server and one mass server per registered
    /// operation. The model path is unused by tensor operations; the builder
    /// still requires one.
    pub async fn start(self) -> Result<RunningServers, String> {
        let (void_addr, void_task) = VoidServerBuilder::new(
            Box::new(object_store::InMemoryObjectStore::new()),
            Box::new(persist::InMemoryStore::new()),
        )
        .tcp()
        .listen(LOOPBACK.parse().expect("valid loopback address"))
        .serve()
        .await
        .map_err(|error| error.to_string())?;

        let mut tasks = vec![ServerTask::Void(void_task)];
        let mut mass_addrs = Vec::with_capacity(self.operations.len());
        for operation in self.operations {
            let (addr, task) = MassServerBuilder::new("unused")
                .tcp()
                .listen(LOOPBACK.parse().expect("valid loopback address"))
                .void_addr(void_addr)
                .operation_shared(operation)
                .serve()
                .await
                .map_err(|error| error.to_string())?;
            mass_addrs.push(addr);
            tasks.push(ServerTask::Mass(task));
        }

        Ok(RunningServers {
            void_addr,
            mass_addrs,
            tasks,
        })
    }
}

/// Outcome of one poll of [`run_until`]'s completion check.
#[derive(Debug, Clone)]
pub enum RunCheck {
    /// Keep running.
    Continue,
    /// The run has finished successfully.
    Done,
    /// The run has failed with a message.
    Failed(String),
}

/// Spawn the top-level animal, run a pool of workers until `check` reports
/// completion, then abort the workers and return the parent journey handle.
///
/// Every live journey needs its own runner task, so `workers` should be at
/// least as large as the number of animals that can be alive at once (the
/// top-level animal plus every cell it spawns). A worker error fails the run
/// immediately; otherwise the check is polled every 50 ms.
pub async fn run_until<J, A, F, Fut>(
    jungle: &J,
    client: &FusedClient,
    seed: &A::Seed,
    workers: usize,
    mut check: F,
) -> Result<JourneyHandle, String>
where
    J: Ecosystem + Clone + Send + Sync + 'static,
    J::Animals: Animals,
    <J::Animals as Animals>::List: FlattenNodes,
    SPFlatten<<J::Animals as Animals>::List>: StripAnimalHeaders,
    AnimalSet<J::Animals>: SupportedAnimalGenerations<J>,
    A: SpawnableAnimal,
    A::Seed: Sync + Send,
    F: FnMut(&JourneyHandle) -> Fut,
    Fut: Future<Output = RunCheck>,
{
    let parent = client.spawn::<A>(seed).await.map_err(|error| error.to_string())?;

    let worker_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let jungle = jungle.clone();
        let client = client.clone();
        let worker_error = Arc::clone(&worker_error);
        handles.push(tokio::spawn(async move {
            let worker = JungleWorker::new(jungle, client);
            if let Err(error) = worker.spawn().await {
                if let Ok(mut slot) = worker_error.lock() {
                    slot.get_or_insert(error.to_string());
                }
            }
        }));
    }

    let result = loop {
        match check(&parent).await {
            RunCheck::Done => break Ok(parent),
            RunCheck::Failed(reason) => break Err(reason),
            RunCheck::Continue => {}
        }
        if let Some(error) = worker_error.lock().ok().and_then(|slot| slot.clone()) {
            break Err(error);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };

    for handle in handles {
        handle.abort();
        let _ = handle.await;
    }
    result
}
