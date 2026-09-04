//! Checkpoint-then-evaluate program: compile a neutral topology, run an
//! arbitrary checkpoint flow, then an arbitrary evaluation flow.

use std::marker::PhantomData;

use black_hole_type::ObjectId;
use jungle_sdk::prelude::*;
use uuid::Uuid;

use crate::compile::SunProgram;
use crate::forward::NeutralSunState;
use crate::nodes::fusion::action::FusionSeed;
use crate::topology::BoundaryInit;

/// A small non-forward, non-QuZO schedule proving that a program can compile
/// a topology and then run arbitrary checkpoint and evaluation flows without
/// inheriting propagation state.
#[derive(Flow)]
pub struct CheckpointEvaluateFlow<Checkpoint, Evaluation, S>(
    Step<crate::compile::action::FinalizeNeutralGraph<S>>,
    Checkpoint,
    Evaluation,
);

pub struct CheckpointEvaluate<Checkpoint, Evaluation, S = ()>(
    PhantomData<Checkpoint>,
    PhantomData<Evaluation>,
    PhantomData<fn() -> S>,
);

impl<Checkpoint, Evaluation, S> SunProgram for CheckpointEvaluate<Checkpoint, Evaluation, S> {
    type State = NeutralSunState<S>;
    type Driver = CheckpointEvaluateFlow<Checkpoint, Evaluation, S>;
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
    fn register_inboxes(_state: &mut Self::State, _ports: &[(u32, ObjectId)]) {}
}