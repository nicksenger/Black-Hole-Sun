//! Local in-process runtime for toy examples: void + mass servers, a worker
//! pool, and run-until-completion polling.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use black_hole_sun::{
    object_store, persist, MassServerBuilder, OperationImplementation, VoidServerBuilder,
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
    /// Address of the filesystem-backed void server.
    pub void_addr: SocketAddr,
    /// Addresses of the mass servers, in registration order.
    pub mass_addrs: Vec<SocketAddr>,
    tasks: Vec<ServerTask>,
    /// Keeps the per-run void storage isolated and removes it on shutdown.
    _void_store_dir: tempfile::TempDir,
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
pub struct ServerSpecs {
    operations: Vec<Arc<dyn OperationImplementation>>,
    max_instances: usize,
}

impl Default for ServerSpecs {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            max_instances: 1,
        }
    }
}

impl ServerSpecs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allow each local mass server to host multiple concurrent model
    /// instances, as required by data-parallel examples.
    pub fn max_instances(mut self, max_instances: usize) -> Self {
        self.max_instances = max_instances.max(1);
        self
    }

    /// Register one tensor operation for its own mass server.
    pub fn operation(mut self, operation: impl OperationImplementation + 'static) -> Self {
        self.operations.push(Arc::new(operation));
        self
    }

    /// Start a filesystem-backed void server in a temporary directory and one
    /// mass server per registered operation. The model path is unused by
    /// tensor operations; the builder still requires one.
    pub async fn start(self) -> Result<RunningServers, String> {
        let void_store_dir = tempfile::tempdir()
            .map_err(|error| format!("failed to create temporary void directory: {error}"))?;
        let objects_path = void_store_dir.path().join("objects");
        let relations_path = void_store_dir.path().join("relations");
        let object_store = object_store::FilesystemObjectStore::new(&objects_path)
            .map_err(|error| format!("failed to create filesystem void store: {error}"))?;
        let store = persist::fjall::FjallStore::new(&relations_path)
            .map_err(|error| format!("failed to create filesystem void metadata store: {error}"))?;

        let (void_addr, void_task) =
            VoidServerBuilder::new(Box::new(object_store), Box::new(store))
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
                .max_instances(self.max_instances)
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
            _void_store_dir: void_store_dir,
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

/// Spawn the top-level animal and a pool of workers, returning the parent
/// journey handle and the worker tasks without waiting for completion.
///
/// Every live journey needs its own runner task, so `workers` should be at
/// least as large as the number of animals that can be alive at once (the
/// top-level animal plus every cell it spawns.
///
/// The returned `Arc` slot records the first worker error, if any, so a
/// caller that polls for completion can fail fast on a dead worker.
async fn spawn_with_pool<J, A>(
    jungle: &J,
    client: &FusedClient,
    seed: &A::Seed,
    workers: usize,
) -> Result<
    (
        JourneyHandle,
        Vec<tokio::task::JoinHandle<()>>,
        Arc<Mutex<Option<String>>>,
    ),
    String,
>
where
    J: Ecosystem + Clone + Send + Sync + 'static,
    J::Animals: Animals,
    <J::Animals as Animals>::List: FlattenNodes,
    SPFlatten<<J::Animals as Animals>::List>: StripAnimalHeaders,
    AnimalSet<J::Animals>: SupportedAnimalGenerations<J>,
    A: SpawnableAnimal,
    A::Seed: Sync + Send,
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

    Ok((parent, handles, worker_error))
}

/// Spawn the top-level animal and a pool of workers on `runtime` without
/// waiting for completion.
///
/// Unlike [`run_until`], this returns immediately with the parent journey
/// handle and the worker tasks. The workers keep progressing on `runtime`
/// until they are aborted, so a caller can hold the current thread doing
/// something else (for example running a GUI event loop on the main thread)
/// while the journey stays alive, then tear everything down afterwards.
pub fn launch<J, A>(
    runtime: &tokio::runtime::Runtime,
    jungle: &J,
    client: &FusedClient,
    seed: &A::Seed,
    workers: usize,
) -> Result<(JourneyHandle, Vec<tokio::task::JoinHandle<()>>), String>
where
    J: Ecosystem + Clone + Send + Sync + 'static,
    J::Animals: Animals,
    <J::Animals as Animals>::List: FlattenNodes,
    SPFlatten<<J::Animals as Animals>::List>: StripAnimalHeaders,
    AnimalSet<J::Animals>: SupportedAnimalGenerations<J>,
    A: SpawnableAnimal,
    A::Seed: Sync + Send,
{
    let (parent, handles, _worker_error) =
        runtime.block_on(spawn_with_pool::<J, A>(jungle, client, seed, workers))?;
    Ok((parent, handles))
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
    let (parent, handles, worker_error) =
        spawn_with_pool::<J, A>(jungle, client, seed, workers).await?;

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
