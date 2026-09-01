use std::future::Future;

use black_hole_sun::AtomError;

use super::*;

#[jungle::effect(id = 84)]
impl<J> Effect<J> for DelayedLeftEffect {
    type In = ();
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(EmissionId::new(Uuid::from_u128(LEFT_EMISSION)))
        }
    }
}

#[jungle::effect(id = 85)]
impl<J: FusionProbeOps> Effect<J> for RecordFusionInputsEffect {
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (transform_id, (p1, p2)): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        jungle.record_fusion_inputs(transform_id, p1.id(), p2.id());
        std::future::ready(Ok(EmissionId::new(Uuid::from_u128(FUSED_EMISSION))))
    }
}
