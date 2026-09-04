//! Deployment actions — spawning animals, generating seeds, and finalizing
//! the neutral topology for a program's state.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use black_hole_spec::{QwenDarkInference, TensorContract};
use black_hole_type::ObjectId;
use jungle_sdk::prelude::*;
use typosaurus::num::Unsigned;
use uuid::Uuid;

use super::effect::{GenFusionSeedEffect, GenUuidEffect};
use crate::topology::{DeclaredEdges, NodeIdsFromList, SunTopologyState};

// ---------------------------------------------------------------------------
// Spawn — descriptor-specific animal spawning and graph registration
// ---------------------------------------------------------------------------

pub(crate) fn register_vertex<Program: super::SunProgram>(
    state: &mut Program::State,
    vertex_id: u32,
    node_label: String,
    ports: &[(u32, ObjectId)],
    contract: black_hole_type::ContractDescriptor,
    declared_edges: Vec<crate::topology::DeclaredEdge>,
    journey_id: Uuid,
    warp_journey_id: Option<Uuid>,
) {
    let mut topology = state.topology().lock().unwrap();

    topology.journey_ids.entry(vertex_id).or_insert(journey_id);
    if let Some(warp_journey_id) = warp_journey_id {
        topology.warp_journey_ids.insert(vertex_id, warp_journey_id);
    }
    topology.node_labels.entry(vertex_id).or_insert(node_label);
    topology
        .node_operational_states
        .entry(vertex_id)
        .or_insert(crate::topology::SunOperationalState::Queued);
    topology.node_state_sequences.entry(vertex_id).or_default();
    topology
        .vertex_ports
        .entry(vertex_id)
        .or_insert_with(|| ports.iter().map(|(port_id, _)| *port_id).collect());
    topology.node_contracts.entry(vertex_id).or_insert(contract);
    topology
        .declared_outputs
        .entry(vertex_id)
        .or_insert_with(|| declared_edges.iter().map(|edge| edge.port_id).collect());
    topology
        .declared_edges
        .entry(vertex_id)
        .or_insert(declared_edges);

    for &(port_id, _) in ports {
        if topology.port_vertices.contains_key(&port_id) {
            topology.duplicate_ports.insert(port_id);
            continue;
        }
        topology.port_vertices.insert(port_id, vertex_id);
    }
    drop(topology);
    Program::register_inboxes(state, ports);
}

fn short_type_name<T: ?Sized>() -> String {
    short_type_label(core::any::type_name::<T>())
}

fn short_type_label(label: &str) -> String {
    let mut shortened = String::with_capacity(label.len());
    let mut token = String::new();

    for ch in label.chars() {
        let token_char = ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '\'');
        if token_char {
            token.push(ch);
            continue;
        }

        push_shortened_type_token(&mut shortened, &mut token);
        shortened.push(ch);
    }

    push_shortened_type_token(&mut shortened, &mut token);
    shortened.trim().to_string()
}

fn push_shortened_type_token(shortened: &mut String, token: &mut String) {
    if token.is_empty() {
        return;
    }

    let short = token.rsplit("::").next().unwrap_or(token.as_str());
    shortened.push_str(short);
    token.clear();
}

/// Spawns and registers a [`Unary`](crate::topology::Unary) descriptor.
pub struct SpawnUnary<P, A, E, Program, Op = QwenDarkInference>(
    PhantomData<fn() -> (P, A, E, Program, Op)>,
);

#[jungle::action]
impl<P, A, E, Program, Op> Action for SpawnUnary<P, A, E, Program, Op>
where
    P: Unsigned,
    Program: super::SunProgram,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed>
        + crate::topology::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = Program::UnarySeed;
    type Output = ();
    type Carry = Program::UnarySeed;

    fn emit(
        _state: &Program::State,
        input: Self::Input,
    ) -> (Program::UnarySeed, Program::UnarySeed) {
        (input.clone(), input)
    }

    fn absorb(
        state: &mut Program::State,
        output: EffectCompletion<Self::Effect>,
        seed: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let port_id = P::U32;
        register_vertex::<Program>(
            state,
            port_id,
            short_type_name::<A>(),
            &[(port_id, Program::unary_inbox(&seed))],
            Op::descriptor(),
            E::declared_edges(),
            journey_id,
            None,
        );

        Ok(())
    }
}

/// Backwards-compatible name for the unary spawn action.
pub type Spawn<P, A, E, Program, Op = QwenDarkInference> = SpawnUnary<P, A, E, Program, Op>;

/// Spawns and registers a [`Binary`](crate::topology::Binary) descriptor.
pub struct SpawnBinary<P1, P2, A, E, Program, Op = QwenDarkInference>(
    PhantomData<fn() -> (P1, P2, A, E, Program, Op)>,
);

#[jungle::action]
impl<P1, P2, A, E, Program, Op> Action for SpawnBinary<P1, P2, A, E, Program, Op>
where
    P1: Unsigned,
    P2: Unsigned,
    Program: super::SunProgram,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed>
        + crate::topology::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = Program::BinarySeed;
    type Output = ();
    type Carry = Program::BinarySeed;

    fn emit(
        _state: &Program::State,
        seed: Self::Input,
    ) -> (Program::BinarySeed, Program::BinarySeed) {
        (seed.clone(), seed)
    }

    fn absorb(
        state: &mut Program::State,
        output: EffectCompletion<Self::Effect>,
        seed: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let p1 = P1::U32;
        let p2 = P2::U32;
        let [p1_inbox, p2_inbox] = Program::binary_inboxes(&seed);
        register_vertex::<Program>(
            state,
            p1,
            short_type_name::<A>(),
            &[(p1, p1_inbox), (p2, p2_inbox)],
            Op::descriptor(),
            E::declared_edges(),
            journey_id,
            None,
        );

        Ok(())
    }
}

/// Spawns and registers a [`Warp`](crate::topology::Warp) descriptor's boundary node.
///
/// This runs in two steps:
/// 1. Spawn the nested warp animal and keep its journey id.
/// 2. Spawn the boundary animal with [`super::BoundaryInit`], then register
///    the boundary journey as the parent graph vertex for scheduling.
pub struct SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op = QwenDarkInference>(
    PhantomData<fn() -> (P, WarpAnimalT, BoundaryAnimalT, E, Program, Op)>,
);

#[jungle::action]
impl<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op> Action
    for SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>
where
    P: Unsigned,
    Program: super::SunProgram,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed>
        + crate::topology::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<WarpAnimalT>;
    type Input = Program::UnarySeed;
    type Output = (Program::UnarySeed, Uuid);
    type Carry = Program::UnarySeed;

    fn emit(_state: &Program::State, input: Self::Input) -> ((), Program::UnarySeed) {
        ((), input)
    }

    fn absorb(
        _state: &mut Program::State,
        output: EffectCompletion<Self::Effect>,
        carry: Program::UnarySeed,
    ) -> Result<Self::Output, Failure> {
        let warp_journey_id =
            output.map_err(|e| Failure::Message(format!("warp spawn failed: {e}")))?;
        Ok((carry, warp_journey_id))
    }
}

pub struct SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op = QwenDarkInference>(
    PhantomData<fn() -> (P, WarpAnimalT, BoundaryAnimalT, E, Program, Op)>,
);

#[jungle::action]
impl<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op> Action
    for SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>
where
    P: Unsigned,
    Program: super::SunProgram,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed>
        + crate::topology::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<BoundaryAnimalT>;
    type Input = (Program::UnarySeed, Uuid);
    type Output = ();
    type Carry = (Program::UnarySeed, Uuid);

    fn emit(
        _state: &Program::State,
        input: Self::Input,
    ) -> (Program::WarpSeed, (Program::UnarySeed, Uuid)) {
        let (init, warp_journey_id) = input;
        (
            Program::warp_seed(Program::unary_inbox(&init), warp_journey_id),
            (init, warp_journey_id),
        )
    }

    fn absorb(
        state: &mut Program::State,
        output: EffectCompletion<Self::Effect>,
        carry: (Program::UnarySeed, Uuid),
    ) -> Result<Self::Output, Failure> {
        let boundary_journey_id =
            output.map_err(|e| Failure::Message(format!("boundary spawn failed: {e}")))?;
        let (init, warp_journey_id) = carry;
        let port_id = P::U32;
        register_vertex::<Program>(
            state,
            port_id,
            format!(
                "Warp<{}, {}>",
                short_type_name::<WarpAnimalT>(),
                short_type_name::<BoundaryAnimalT>()
            ),
            &[(port_id, Program::unary_inbox(&init))],
            Op::descriptor(),
            E::declared_edges(),
            boundary_journey_id,
            Some(warp_journey_id),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GenUuid
// ---------------------------------------------------------------------------

/// Generates the initial inbox used to seed one spawned cell journey.
///
/// The selected [`SunProgram`](super::SunProgram) owns the accumulation
/// setting so deployment no longer needs a universal const generic.
pub struct GenUuid<P: super::SunProgram>(PhantomData<fn() -> P>);

#[jungle::action]
impl<P: super::SunProgram> Action for GenUuid<P> {
    type Effect = GenUuidEffect;
    type Input = ();
    type Output = P::UnarySeed;

    fn emit(_state: &P::State, _input: Self::Input) {}
    fn absorb(
        _state: &mut P::State,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let recv_id =
            output.map_err(|_e| Failure::Message("failed to generate a uuid...".to_string()))?;
        Ok(P::unary_seed(recv_id))
    }
}

/// Generates the two independent initial inboxes for a binary vertex.
pub struct GenFusionSeed<P: super::SunProgram>(PhantomData<fn() -> P>);

#[jungle::action]
impl<P: super::SunProgram> Action for GenFusionSeed<P> {
    type Effect = GenFusionSeedEffect;
    type Input = ();
    type Output = P::BinarySeed;

    fn emit(_state: &P::State, _input: Self::Input) {}

    fn absorb(
        _state: &mut P::State,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let seed =
            output.map_err(|_| Failure::Message("failed to generate fusion seed".to_string()))?;
        Ok(P::binary_seed([seed.p1_recv_id, seed.p2_recv_id]))
    }
}

fn resolve_neutral_topology(topology: &mut crate::topology::SunTopology) -> Result<u32, Failure> {
    topology.finalized = false;
    if !topology.duplicate_ports.is_empty() {
        let mut ports = topology.duplicate_ports.iter().copied().collect::<Vec<_>>();
        ports.sort_unstable();
        return Err(Failure::Message(format!(
            "duplicate input port ownership: {ports:?}"
        )));
    }
    let vertices = topology.journey_ids.keys().copied().collect::<HashSet<_>>();
    if vertices.is_empty() {
        return Err(Failure::Message(
            "sun graph must contain at least one vertex".to_string(),
        ));
    }
    let mut producer_counts = topology
        .port_vertices
        .keys()
        .copied()
        .map(|port| (port, 0usize))
        .collect::<HashMap<_, _>>();
    let mut outgoing = vertices
        .iter()
        .copied()
        .map(|id| (id, Vec::new()))
        .collect::<HashMap<u32, Vec<crate::topology::PortTarget>>>();
    let mut incoming = vertices
        .iter()
        .copied()
        .map(|id| (id, Vec::new()))
        .collect::<HashMap<u32, Vec<u32>>>();

    for (&source, output_ports) in &topology.declared_outputs {
        let source_contract = topology.node_contracts.get(&source).ok_or_else(|| {
            Failure::Message(format!(
                "vertex {source} did not register an operation contract"
            ))
        })?;
        let declared = topology
            .declared_edges
            .get(&source)
            .cloned()
            .unwrap_or_default();
        for &port in output_ports {
            let target = *topology.port_vertices.get(&port).ok_or_else(|| {
                Failure::Message(format!(
                    "output from vertex {source} targets missing port {port}"
                ))
            })?;
            let edge = declared
                .iter()
                .find(|edge| edge.port_id == port)
                .ok_or_else(|| {
                    Failure::Message(format!(
                        "output from vertex {source} to port {port} has no contract descriptor"
                    ))
                })?;
            let destination_contract = topology.node_contracts.get(&target).ok_or_else(|| {
                Failure::Message(format!(
                    "destination vertex {target} did not register an operation contract"
                ))
            })?;
            if &edge.source_contract != source_contract {
                return Err(Failure::Message(format!(
                    "source contract mismatch for edge {source} -> port {port}"
                )));
            }
            if &edge.destination_contract != destination_contract {
                return Err(Failure::Message(format!(
                    "destination contract mismatch for edge {source} -> port {port}"
                )));
            }
            if source_contract.outputs != destination_contract.inputs {
                return Err(Failure::Message(format!(
                    "artifact bundle mismatch for edge {source} -> port {port}"
                )));
            }
            *producer_counts.get_mut(&port).expect("registered port") += 1;
            outgoing
                .entry(source)
                .or_default()
                .push(crate::topology::PortTarget {
                    port_id: port,
                    vertex_id: target,
                });
            incoming.entry(target).or_default().push(source);
        }
    }
    for (&port, &count) in &producer_counts {
        if count > 1 {
            return Err(Failure::Message(format!(
                "input port {port} has {count} producers; expected at most one"
            )));
        }
    }
    for (&vertex, ports) in &topology.vertex_ports {
        let counts = ports
            .iter()
            .map(|port| producer_counts.get(port).copied().unwrap_or(0))
            .collect::<Vec<_>>();
        if !counts.iter().all(|count| *count == 0) && !counts.iter().all(|count| *count == 1) {
            return Err(Failure::Message(format!(
                "vertex {vertex} has incorrect producer counts for ports {ports:?}: {counts:?}"
            )));
        }
    }
    let mut degrees = incoming
        .iter()
        .map(|(&id, sources)| (id, sources.len()))
        .collect::<HashMap<_, _>>();
    let mut roots = degrees
        .iter()
        .filter_map(|(&id, &degree)| (degree == 0).then_some(id))
        .collect::<Vec<_>>();
    roots.sort_unstable();
    let mut queue = VecDeque::from(roots);
    let mut visited = 0;
    while let Some(id) = queue.pop_front() {
        visited += 1;
        for target in outgoing.get(&id).into_iter().flatten() {
            let degree = degrees.get_mut(&target.vertex_id).expect("resolved vertex");
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(target.vertex_id);
            }
        }
    }
    if visited != vertices.len() {
        return Err(Failure::Message("sun graph contains a cycle".to_string()));
    }
    let mut sinks = vertices
        .iter()
        .copied()
        .filter(|id| outgoing.get(id).is_none_or(Vec::is_empty))
        .collect::<Vec<_>>();
    sinks.sort_unstable();
    if sinks.len() != 1 {
        return Err(Failure::Message(format!(
            "sun graph must contain exactly one sink; found {sinks:?}"
        )));
    }
    topology.incoming = incoming;
    topology.outgoing = outgoing;
    topology.finalized = true;
    Ok(sinks[0])
}

/// Finalizes a graph for a forward program without allocating any P1/P2/PO
/// strategy mailboxes.
pub struct FinalizeForwardGraph<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for FinalizeForwardGraph<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &crate::forward::ForwardSunState<S>, _input: ()) {}

    fn absorb(
        state: &mut crate::forward::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("finalize forward graph failed".to_string()))?;
        let sink = resolve_neutral_topology(&mut state.topology.lock().unwrap())?;
        state.runtime.pending.clear();
        state.runtime.ready.clear();
        state.runtime.next_inputs.clear();
        state.runtime.outputs.clear();
        state.runtime.sink_id = Some(sink);
        Ok(())
    }
}

/// Finalizes a topology for a program with no forward or QuZO runtime.
pub struct FinalizeNeutralGraph<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for FinalizeNeutralGraph<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &crate::forward::NeutralSunState<S>, _input: ()) {}

    fn absorb(
        state: &mut crate::forward::NeutralSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("finalize neutral graph failed".to_string()))?;
        resolve_neutral_topology(&mut state.topology.lock().unwrap())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jungle_sdk::Id;
    use typenum::{U0, U1, U2};
    use typosaurus::collections::list::Empty;

    use crate::forward::action::{
        CompleteForwardPass, PrepareForwardPass, ProcessForwardNode, SendForwardRoots,
    };
    use crate::programs::two_sided_zo::action::{BroadcastPotentiation, FinalizeGraph};
    use crate::topology::sorted_node_ids;

    struct GenericType<T>(std::marker::PhantomData<T>);

    struct TestSunAnimalWithPayload;

    impl Animal for TestSunAnimalWithPayload {
        type Id = Id<U0>;
        type Generation = U0;
        type State = crate::programs::two_sided_zo::SunState<(String, String)>;
        type Seed = ();
        type Flow = ();
    }

    struct TestForwardAnimal;

    impl Animal for TestForwardAnimal {
        type Id = ();
        type Generation = ();
        type State = crate::forward::ForwardSunState;
        type Seed = ();
        type Flow = ();
    }

    struct TestUnaryChildAnimal;

    impl Animal for TestUnaryChildAnimal {
        type Id = Id<U1>;
        type Generation = U0;
        type State = crate::CellState;
        type Seed = crate::nodes::cell::action::Init;
        type Flow = crate::Primordium;
    }

    struct TestFusionChildAnimal;

    impl Animal for TestFusionChildAnimal {
        type Id = Id<U2>;
        type Generation = U0;
        type State = crate::FusionState;
        type Seed = crate::FusionSeed;
        type Flow = crate::Fusion<crate::Primordium>;
    }

    struct TestWarpChildAnimal;

    impl Animal for TestWarpChildAnimal {
        type Id = Id<U1>;
        type Generation = U0;
        type State = crate::programs::two_sided_zo::SunState;
        type Seed = ();
        type Flow = ();
    }

    impl Observe for TestWarpChildAnimal {
        type Appearance = crate::Ray;

        fn observe(_state: &Self::State) -> Self::Appearance {
            crate::Ray { frozen: false }
        }
    }

    struct TestWarpBoundaryAnimal;

    impl Animal for TestWarpBoundaryAnimal {
        type Id = Id<U2>;
        type Generation = U0;
        type State = crate::BoundaryState<<TestWarpChildAnimal as Observe>::Appearance>;
        type Seed = crate::topology::BoundaryInit;
        type Flow = ();
    }

    #[test]
    fn sun_actions_bind_with_custom_state_payload() {
        type Payload = (String, String);

        let state = crate::programs::two_sided_zo::SunState::<Payload>::default();

        type Program = crate::programs::two_sided_zo::DeploymentProgram<Payload, 1>;
        type GenUuidBound = <GenUuid<Program> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenUuidBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type GenFusionSeedBound =
            <GenFusionSeed<Program> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenFusionSeedBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type FinalizeBound = <FinalizeGraph<Payload, 1> as Action>::Bind<TestSunAnimalWithPayload>;
        <FinalizeBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type BroadcastBound =
            <BroadcastPotentiation<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <BroadcastBound as BoundAction<TestSunAnimalWithPayload>>::emit(
            &state,
            black_hole_type::Potentiation {
                loss_up: 0.1,
                loss_down: 0.2,
                seed: 7,
            },
        );

        type SpawnUnaryBound =
            <SpawnUnary<U1, TestUnaryChildAnimal, Empty, Program> as Action>::Bind<
                TestSunAnimalWithPayload,
            >;
        let seed = crate::nodes::cell::action::Init {
            recv_id: Uuid::new_v4(),
            grad_steps: 1,
        };
        let effect_seed =
            <SpawnUnaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, seed);
        assert_eq!(effect_seed, seed);

        type SpawnBinaryBound =
            <SpawnBinary<U1, U2, TestFusionChildAnimal, Empty, Program> as Action>::Bind<
                TestSunAnimalWithPayload,
            >;
        let seed = crate::FusionSeed {
            p1_recv_id: Uuid::new_v4(),
            p2_recv_id: Uuid::new_v4(),
            grad_steps: 1,
        };
        let effect_seed =
            <SpawnBinaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, seed);
        assert_eq!(effect_seed.p1_recv_id, seed.p1_recv_id);
        assert_eq!(effect_seed.p2_recv_id, seed.p2_recv_id);
    }

    #[test]
    fn warp_actions_seed_boundary_with_spawned_warp_journey() {
        type Payload = (String, String);
        let mut state = crate::programs::two_sided_zo::SunState::<Payload>::default();
        let init = crate::nodes::cell::action::Init {
            recv_id: Uuid::new_v4(),
            grad_steps: 3,
        };

        type SpawnWarpAnimalBound = <SpawnWarpAnimal<
            U1,
            TestWarpChildAnimal,
            TestWarpBoundaryAnimal,
            Empty,
            crate::programs::two_sided_zo::DeploymentProgram<Payload, 3>,
        > as Action>::Bind<TestSunAnimalWithPayload>;
        let spawn_input =
            <SpawnWarpAnimalBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, init);
        assert_eq!(spawn_input, ());

        let warp_journey_id = Uuid::new_v4();
        let (boundary_init, boundary_carry) = <SpawnWarpAnimalBound as BoundAction<
            TestSunAnimalWithPayload,
        >>::absorb_with_carry(
            &mut state, Ok(warp_journey_id), init
        )
        .unwrap();
        assert_eq!(boundary_init, init);
        assert_eq!(boundary_carry, warp_journey_id);

        type SpawnWarpBoundaryBound = <SpawnWarpBoundary<
            U1,
            TestWarpChildAnimal,
            TestWarpBoundaryAnimal,
            Empty,
            crate::programs::two_sided_zo::DeploymentProgram<Payload, 3>,
        > as Action>::Bind<TestSunAnimalWithPayload>;
        let boundary_seed = <SpawnWarpBoundaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(
            &state,
            (boundary_init, boundary_carry),
        );
        assert_eq!(boundary_seed.recv_id, init.recv_id);
        assert_eq!(boundary_seed.grad_steps, init.grad_steps);
        assert_eq!(boundary_seed.warp_journey_id, warp_journey_id);

        let boundary_journey_id = Uuid::new_v4();
        <SpawnWarpBoundaryBound as BoundAction<TestSunAnimalWithPayload>>::absorb_with_carry(
            &mut state,
            Ok(boundary_journey_id),
            (boundary_init, boundary_carry),
        )
        .unwrap();

        let topology = state.topology.lock().unwrap();
        assert_eq!(
            topology.journey_ids.get(&U1::U32),
            Some(&boundary_journey_id)
        );
        assert_eq!(
            topology.warp_journey_ids.get(&U1::U32),
            Some(&warp_journey_id)
        );
        assert_eq!(topology.port_vertices.get(&U1::U32), Some(&U1::U32));
    }

    #[test]
    fn short_type_name_preserves_generic_arguments() {
        type Nested = GenericType<Result<String, Vec<u8>>>;
        assert_eq!(
            short_type_name::<Nested>(),
            "GenericType<Result<String, Vec<u8>>>"
        );
        assert_eq!(
            short_type_label("my::module::Animal<crate::Inner<leaf::Type>, alloc::vec::Vec<u8>>"),
            "Animal<Inner<Type>, Vec<u8>>"
        );
    }

    #[test]
    fn neutral_forward_pass_routes_typed_artifacts_and_rotates_inboxes() {
        type Program = crate::programs::forward_only::ForwardOnly<(), QwenDarkInference>;
        let mut state = crate::forward::ForwardSunState::default();
        let add =
            |state: &mut crate::forward::ForwardSunState, vertex_id, port_id, outputs: &[u32]| {
                let ports = [(port_id, Uuid::new_v4())];
                register_vertex::<Program>(
                    state,
                    vertex_id,
                    format!("Node{vertex_id}"),
                    &ports,
                    QwenDarkInference::descriptor(),
                    outputs
                        .iter()
                        .map(|&port_id| crate::topology::DeclaredEdge {
                            port_id,
                            source_contract: QwenDarkInference::descriptor(),
                            destination_contract: QwenDarkInference::descriptor(),
                        })
                        .collect(),
                    Uuid::new_v4(),
                    None,
                );
            };
        add(&mut state, 0, 0, &[1]);
        add(&mut state, 1, 1, &[]);
        type Finalize = <FinalizeForwardGraph<()> as Action>::Bind<TestForwardAnimal>;
        <Finalize as BoundAction<TestForwardAnimal>>::absorb(&mut state, Ok(())).unwrap();

        let delivery = black_hole_type::ArtifactDelivery::<()> {
            emission_id: black_hole_type::EmissionId::new(Uuid::new_v4()),
            recv: Uuid::new_v4(),
            send: Uuid::new_v4(),
        };

        type Prepare = <PrepareForwardPass<(), ()> as Action>::Bind<TestForwardAnimal>;
        <Prepare as BoundAction<TestForwardAnimal>>::emit(&state, delivery);
        let prepared = <Prepare as BoundAction<TestForwardAnimal>>::absorb_with_carry(
            &mut state,
            Ok(()),
            delivery,
        )
        .unwrap();
        assert_eq!(sorted_node_ids(&state.runtime.ready), vec![0]);

        type Send = <SendForwardRoots<()> as Action>::Bind<TestForwardAnimal>;
        let root_input = <Send as BoundAction<TestForwardAnimal>>::emit(&state, prepared);
        assert_eq!(root_input.targets.len(), 1);
        assert_eq!(root_input.targets[0].node_id, 0);
        let routed = <Send as BoundAction<TestForwardAnimal>>::absorb_with_carry(
            &mut state,
            Ok(vec![0]),
            prepared,
        )
        .unwrap();

        type Process = <ProcessForwardNode<()> as Action>::Bind<TestForwardAnimal>;
        let routed = <Process as BoundAction<TestForwardAnimal>>::absorb(
            &mut state,
            Ok(crate::forward::effect::SchedulerDelivery {
                node_id: 0,
                delivery: routed,
                sent_node_ids: vec![1],
            }),
        )
        .unwrap();
        assert_eq!(sorted_node_ids(&state.runtime.ready), vec![1]);

        let completed = <Process as BoundAction<TestForwardAnimal>>::absorb(
            &mut state,
            Ok(crate::forward::effect::SchedulerDelivery {
                node_id: 1,
                delivery: routed,
                sent_node_ids: vec![],
            }),
        )
        .unwrap();
        assert!(state.runtime.pending.is_empty());

        let next_inboxes = state.runtime.next_inputs.clone();
        type Complete = <CompleteForwardPass<(), ()> as Action>::Bind<TestForwardAnimal>;
        <Complete as BoundAction<TestForwardAnimal>>::emit(&state, completed);
        <Complete as BoundAction<TestForwardAnimal>>::absorb_with_carry(
            &mut state,
            Ok(()),
            completed,
        )
        .unwrap();

        let topology = state.topology.lock().unwrap();
        assert_eq!(state.runtime.inputs, next_inboxes);
        assert_eq!(
            topology.node_operational_states.get(&0),
            Some(&crate::topology::SunOperationalState::Succeeded)
        );
        assert_eq!(
            topology.node_operational_states.get(&1),
            Some(&crate::topology::SunOperationalState::Succeeded)
        );
        assert_eq!(
            topology.node_phase_annotations.get(&1).map(String::as_str),
            Some("forward")
        );
    }
}
