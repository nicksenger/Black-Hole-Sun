use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
use uuid::Uuid;

// 1. Spawn all children getting uuids
// OUTER LOOP
// // FOR STAGES (prop1, prop2, potentiation)
// // 2. Build topological ordering
// // // WHILE TOPO NOT EMPTY
// // // 3. Pop topo vec into current
// // // // WHILE CURRENT NOT EMPTY
// // // // 4. wait for FIRST rx of the set
// // // // 5. remove from current and rotate rx
// // // // 6. construct and send to outgoing-tx with rx ObjectIds as send
// // // // 7. rotate tx

pub struct Tag<N, T, E>(PhantomData<N>, PhantomData<T>, PhantomData<E>);

pub struct SunState {
    incoming: HashMap<Uuid, Vec<Uuid>>,
    outgoing: HashMap<Uuid, Vec<Uuid>>,
    tx: HashMap<Uuid, ObjectId>,
    rx: HashMap<Uuid, ObjectId>,
    topo: Vec<HashSet<Uuid>>,
    current: HashSet<Uuid>,
}

pub struct Sun<Cells>(PhantomData<Cells>);

#[derive(Flow)]
pub struct Ray<T, U>(Step<Spawn<T>>, U);
