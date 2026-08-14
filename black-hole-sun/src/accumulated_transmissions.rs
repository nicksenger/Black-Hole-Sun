use black_hole_flux::ops::{InferenceOutputOps, VoidInferOps};
use black_hole_flux::{AtomError, InferenceOutput, Transmission};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Monoid behavior for metadata merging when accumulating transmissions.
pub trait Monoid {
    fn identity() -> Self;
    fn merge(self, other: Self) -> Self;
}

impl<T> Monoid for Vec<T> {
    fn identity() -> Self {
        Self::default()
    }

    fn merge(mut self, other: Self) -> Self {
        self.extend(other);
        self
    }
}

impl Monoid for () {
    fn identity() -> Self {
        Self::default()
    }

    fn merge(self, _other: Self) -> Self {
        self
    }
}

impl Monoid for InferenceOutput {
    fn identity() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.results.extend(other.results);
        self
    }
}

/// Owned collection of paired up/down propagation transmissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccumulatedTransmissions(pub Vec<(Transmission, Transmission)>);

impl AccumulatedTransmissions {
    pub fn from_array<const N: usize>(pairs: [(Transmission, Transmission); N]) -> Self {
        Self(pairs.into())
    }

    /// Downloads all propagation pairs, then merges outputs and metadata.
    pub async fn download_merged<J, M>(
        self,
        jungle: &J,
    ) -> Result<((InferenceOutput, M), (InferenceOutput, M)), AtomError>
    where
        J: VoidInferOps,
        M: Monoid + Serialize + DeserializeOwned + Send,
    {
        let mut left_output = InferenceOutput::identity();
        let mut right_output = InferenceOutput::identity();
        let mut left_metadata = M::identity();
        let mut right_metadata = M::identity();

        for (left, right) in self.0 {
            let (resolved_left, left_meta) =
                InferenceOutput::from_transmission_with_metadata::<J, M>(jungle, &left).await?;
            let (resolved_right, right_meta) =
                InferenceOutput::from_transmission_with_metadata::<J, M>(jungle, &right).await?;

            left_output = left_output.merge(resolved_left);
            right_output = right_output.merge(resolved_right);
            left_metadata = left_metadata.merge(left_meta);
            right_metadata = right_metadata.merge(right_meta);
        }

        Ok(((left_output, left_metadata), (right_output, right_metadata)))
    }
}
