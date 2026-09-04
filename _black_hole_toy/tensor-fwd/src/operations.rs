//! Server-side tensor operations for the three pipeline cells.

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::{
    decode_input, encode_output, OperationCapability, OperationConfig, OperationImplementation,
    TensorContract,
};

use crate::contracts::{Matmul, Relu, Scale};

pub struct MatmulOperation;

#[async_trait]
impl OperationImplementation for MatmulOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Matmul>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let decoded = decode_input::<Matmul>(&input).map_err(|error| error.to_string())?;
        let values = decoded.first_tensor()?.to_f32()?;
        let weights = [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let mut output = Vec::with_capacity(8);
        for row in 0..2 {
            for column in 0..4 {
                output.push(
                    (0..3)
                        .map(|k| values[row * 3 + k] * weights[k][column])
                        .sum(),
                );
            }
        }
        encode_output::<Matmul>(&[Matmul::output_f32(&[2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}

pub struct ScaleOperation;

#[async_trait]
impl OperationImplementation for ScaleOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Scale>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let decoded = decode_input::<Scale>(&input).map_err(|error| error.to_string())?;
        let values = decoded.first_tensor()?.to_f32()?;
        let output = values.iter().map(|value| value * 0.5).collect::<Vec<_>>();
        encode_output::<Scale>(&[Scale::output_f32(&[2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}

pub struct ReluOperation;

#[async_trait]
impl OperationImplementation for ReluOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Relu>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let decoded = decode_input::<Relu>(&input).map_err(|error| error.to_string())?;
        let values = decoded.first_tensor()?.to_f32()?;
        let output = values.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
        encode_output::<Relu>(&[Relu::output_f32(&[2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}
