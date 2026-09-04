//! Forward-only serving program.

use std::marker::PhantomData;

use black_hole_spec::TensorContract;
use black_hole_type::ObjectId;
use uuid::Uuid;

use crate::compile::SunProgram;
use crate::forward::{ForwardSunState, ServeFlow, ServeFlowWithPolicy};
use crate::nodes::fusion::action::FusionSeed;
use crate::topology::BoundaryInit;

/// Forward-only Sun program for a homogeneous operation topology.
///
/// Nodes used with this program can run [`crate::ForwardOperationCell`] and
/// therefore require only `MassOps<Op>`; no perturb or optimize capability is
/// part of the driver.
pub struct ForwardOnly<Source, InputOp: TensorContract, OutputOp: TensorContract = InputOp, S = ()>(
    PhantomData<Source>,
    PhantomData<InputOp>,
    PhantomData<OutputOp>,
    PhantomData<fn() -> S>,
);

/// Forward-only Sun program with a policy flow applied to each completed
/// sink artifact.
pub struct ForwardOnlyWithPolicy<
    Source,
    InputOp: TensorContract,
    OutputOp: TensorContract = InputOp,
    Policy = (),
    S = (),
>(
    PhantomData<Source>,
    PhantomData<InputOp>,
    PhantomData<OutputOp>,
    PhantomData<Policy>,
    PhantomData<fn() -> S>,
);

impl<Source, InputOp, OutputOp, Policy, S> SunProgram
    for ForwardOnlyWithPolicy<Source, InputOp, OutputOp, Policy, S>
where
    InputOp: TensorContract,
    OutputOp: TensorContract,
    InputOp::Input: Send + 'static,
    OutputOp::Output: Send + 'static,
{
    type State = ForwardSunState<S>;
    type Driver = ServeFlowWithPolicy<Source, InputOp::Input, OutputOp::Output, S, Policy>;
    type UnarySeed = crate::nodes::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::nodes::cell::action::Init {
            recv_id: inbox,
            grad_steps: 1,
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: 1,
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: 1,
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state.runtime.inputs.extend(ports.iter().copied());
    }
}

impl<Source, InputOp, OutputOp, S> SunProgram for ForwardOnly<Source, InputOp, OutputOp, S>
where
    InputOp: TensorContract,
    OutputOp: TensorContract,
    InputOp::Input: Send + 'static,
    OutputOp::Output: Send + 'static,
{
    type State = ForwardSunState<S>;
    type Driver = ServeFlow<Source, InputOp::Input, OutputOp::Output, S>;
    type UnarySeed = crate::nodes::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::nodes::cell::action::Init {
            recv_id: inbox,
            grad_steps: 1,
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: 1,
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: 1,
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state.runtime.inputs.extend(ports.iter().copied());
    }
}
