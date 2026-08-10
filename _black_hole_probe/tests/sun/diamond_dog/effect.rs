use std::future::Future;

use black_hole_flux::AtomError;

use super::*;

impl<J> EffectSchema<J> for DelayedLeftEffect {
    type Id = u64;
    type In = ();
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for DelayedLeftEffect {
    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(EmissionId(Uuid::from_u128(LEFT_EMISSION)))
        }
    }
}

impl<J> EffectSchema<J> for RecordFusionInputsEffect {
    type Id = u64;
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for RecordFusionInputsEffect
where
    J: FusionProbeOps,
{
    fn effect(
        jungle: &J,
        (transform_id, (p1, p2)): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        jungle.record_fusion_inputs(transform_id, p1.0, p2.0);
        std::future::ready(Ok(EmissionId(Uuid::from_u128(FUSED_EMISSION))))
    }
}
