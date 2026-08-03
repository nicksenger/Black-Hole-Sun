use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

pub struct SunState(Arc<Mutex<SunInner>>);
pub struct SunInner {
    outgoing: HashMap<Uuid, Vec<Uuid>>,
    tx: HashMap<Uuid, ObjectId>,
    rx: HashMap<Uuid, ObjectId>,
}
