//! A forward-only ResNet-18 pipeline for classifying Stanford Dogs images.
//!
//! The model is split at the natural ResNet boundaries:
//!
//! ```text
//! dataset generator -> stem -> stage1 -> stage2 -> stage3 -> stage4 -> binary head -> policy
//! ```

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::{
    glowstick::{Shape2, Shape4},
    SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::forward::ForwardSunState;
use black_hole_sun::topology::{Edge, TypedEdges, Unary};
use black_hole_sun::{
    ArtifactDelivery, ArtifactRef, CellInit, ContractId, DtypeConstraint, Emission,
    ForwardOnlyWithPolicy, ForwardOperationPrimordium, ObjectId, ObjectRef, OperationNode,
    RawTensor, TensorDtype, VoidOps,
};
use candle::{IndexOp, Tensor, D};
use candle_datasets::hub::from_hub;
use candle_nn::{batch_norm, Func, Linear, VarBuilder};
use hf_hub::HFClientSync;
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use parquet::record::{Field, Row};
use tracing::info;
use typenum::consts::{U0, U1, U128, U14, U2, U224, U256, U28, U3, U4, U5, U512, U56, U6, U64, U7};

pub const IMAGE_SIZE: usize = 224;
pub const DATASET_ID: &str = "maurice-fp/stanford-dogs";
pub const DATASET_SAMPLES: usize = 20_580;
const PEMBROKE_LABEL: u32 = 111;
const CARDIGAN_LABEL: u32 = 112;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SampleMetadata {
    pub dataset_label: u32,
}

pub struct ImagePort;
impl TensorPortSpec for ImagePort {
    type Shape = Shape4<U1, U3, U224, U224>;
    const NAME: &'static str = "image";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct StemPort;
impl TensorPortSpec for StemPort {
    type Shape = Shape4<U1, U64, U56, U56>;
    const NAME: &'static str = "stem";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage1Port;
impl TensorPortSpec for Stage1Port {
    type Shape = Shape4<U1, U64, U56, U56>;
    const NAME: &'static str = "stage1";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage2Port;
impl TensorPortSpec for Stage2Port {
    type Shape = Shape4<U1, U128, U28, U28>;
    const NAME: &'static str = "stage2";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage3Port;
impl TensorPortSpec for Stage3Port {
    type Shape = Shape4<U1, U256, U14, U14>;
    const NAME: &'static str = "stage3";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage4Port;
impl TensorPortSpec for Stage4Port {
    type Shape = Shape4<U1, U512, U7, U7>;
    const NAME: &'static str = "stage4";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct LogitsPort;
impl TensorPortSpec for LogitsPort {
    type Shape = Shape2<U1, U2>;
    const NAME: &'static str = "logits";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub type Image = SingleTensorSpec<ImagePort>;
pub type Stem = SingleTensorSpec<StemPort>;
pub type Stage1 = SingleTensorSpec<Stage1Port>;
pub type Stage2 = SingleTensorSpec<Stage2Port>;
pub type Stage3 = SingleTensorSpec<Stage3Port>;
pub type Stage4 = SingleTensorSpec<Stage4Port>;
pub type Logits = SingleTensorSpec<LogitsPort>;

macro_rules! contract {
    ($name:ident, $input:ty, $output:ty, $id:expr) => {
        pub struct $name;
        impl TensorContract for $name {
            type Input = $input;
            type Output = $output;
            type Metadata = SampleMetadata;
            const ID: ContractId = ContractId::from_u128($id);
            const VERSION: u32 = 1;
        }
    };
}

contract!(StemOp, Image, Stem, 0x636f7267692d7374656d2d3030303031);
contract!(Stage1Op, Stem, Stage1, 0x636f7267692d73746167653130303031);
contract!(Stage2Op, Stage1, Stage2, 0x636f7267692d73746167653230303031);
contract!(Stage3Op, Stage2, Stage3, 0x636f7267692d73746167653330303031);
contract!(Stage4Op, Stage3, Stage4, 0x636f7267692d73746167653430303031);
contract!(HeadOp, Stage4, Logits, 0x636f7267692d686561642d3030303031);

pub struct StemCell;
impl Animal for StemCell {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<StemOp>;
}
impl Observable for StemCell {
    type Observation = NoopObservation;
}
impl Perturbable for StemCell {
    type Perturbation = NoopPerturbation;
}
impl OperationNode<StemOp> for StemCell {}

macro_rules! operation_cell {
    ($cell:ident, $id:ty, $op:ty) => {
        pub struct $cell;
        impl Animal for $cell {
            type Id = Id<$id>;
            type Generation = U0;
            type State = CellState;
            type Seed = CellInit;
            type Flow = ForwardOperationPrimordium<$op>;
        }
        impl Observable for $cell {
            type Observation = NoopObservation;
        }
        impl Perturbable for $cell {
            type Perturbation = NoopPerturbation;
        }
        impl OperationNode<$op> for $cell {}
    };
}
operation_cell!(Stage1Cell, U1, Stage1Op);
operation_cell!(Stage2Cell, U2, Stage2Op);
operation_cell!(Stage3Cell, U3, Stage3Op);
operation_cell!(Stage4Cell, U4, Stage4Op);
operation_cell!(HeadCell, U5, HeadOp);

pub type CorgiGraph = list![
    Unary<U0, StemCell, TypedEdges<list![Edge<U1, Stage1Op>]>, StemOp>,
    Unary<U1, Stage1Cell, TypedEdges<list![Edge<U2, Stage2Op>]>, Stage1Op>,
    Unary<U2, Stage2Cell, TypedEdges<list![Edge<U3, Stage3Op>]>, Stage2Op>,
    Unary<U3, Stage3Cell, TypedEdges<list![Edge<U4, Stage4Op>]>, Stage3Op>,
    Unary<U4, Stage4Cell, TypedEdges<list![Edge<U5, HeadOp>]>, Stage4Op>,
    Unary<U5, HeadCell, TypedEdges<list![]>, HeadOp>
];

#[derive(Flow)]
pub struct Generator(Step<GenerateImage>);

pub struct GenerateImage;
#[jungle::action]
impl Action for GenerateImage {
    type Effect = GenerateImageEffect;
    type Input = ();
    type Output = ArtifactDelivery<Image>;

    fn emit(_state: &ForwardSunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("image generator failed: {error}")))
    }
}

pub struct GenerateImageEffect;
#[jungle::effect(id = 201)]
impl<J: VoidOps> Effect<J> for GenerateImageEffect {
    type In = ();
    type Out = ArtifactDelivery<Image>;
    type Err = String;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let sample = next_sample()?;
            let tensor = image_tensor(&sample.image)?;
            let input = black_hole_sun::encode_input::<StemOp>(
                &[RawTensor {
                    name: ImagePort::NAME.to_owned(),
                    dtype: TensorDtype::F32,
                    shape: vec![1, 3, IMAGE_SIZE, IMAGE_SIZE],
                    data: tensor.into_iter().flat_map(f32::to_le_bytes).collect(),
                }],
                &SampleMetadata {
                    dataset_label: sample.label,
                },
            )
            .map_err(|error| error.to_string())?;
            let tensor_id = jungle.upload_to_void(input).await?;
            let emission = Emission::<SampleMetadata, Image> {
                metadata: SampleMetadata {
                    dataset_label: sample.label,
                },
                output_id: ArtifactRef::committed(ObjectRef::new(tensor_id)),
            };
            let emission_id = jungle
                .upload_to_void(postcard::to_allocvec(&emission).map_err(|e| e.to_string())?)
                .await?;
            // Give the forward sun time to observe the emission before the
            // generator's journey is suspended. This is also how the
            // matmul-fwd example avoids racing the worker subscription.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            Ok(ArtifactDelivery {
                emission_id: ObjectRef::new(emission_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            })
        }
    }
}

#[derive(Debug)]
struct DatasetSample {
    image: Vec<u8>,
    label: u32,
}

struct DatasetCursor {
    readers: Vec<parquet::record::reader::RowIter<'static>>,
    reader_index: usize,
}

impl DatasetCursor {
    fn new() -> Result<Self, String> {
        let api = HFClientSync::new().map_err(|e| format!("Hugging Face API: {e}"))?;
        let readers = from_hub(&api, DATASET_ID.to_owned())
            .map_err(|e| format!("download Stanford Dogs parquet: {e}"))?
            .into_iter()
            .map(IntoIterator::into_iter)
            .collect();
        Ok(Self {
            readers,
            reader_index: 0,
        })
    }

    fn next(&mut self) -> Result<Option<DatasetSample>, String> {
        loop {
            let Some(reader) = self.readers.get_mut(self.reader_index) else {
                return Ok(None);
            };
            match reader.next() {
                Some(Ok(row)) => return parse_row(row).map(Some),
                Some(Err(error)) => return Err(format!("read Stanford Dogs row: {error}")),
                None => self.reader_index += 1,
            }
        }
    }
}

static DATASET: OnceLock<Result<Mutex<DatasetCursor>, String>> = OnceLock::new();

fn next_sample() -> Result<DatasetSample, String> {
    let cursor = DATASET
        .get_or_init(|| DatasetCursor::new().map(Mutex::new))
        .as_ref()
        .map_err(Clone::clone)?;
    cursor
        .lock()
        .map_err(|_| "Stanford Dogs cursor poisoned".to_owned())?
        .next()?
        .ok_or_else(|| "Stanford Dogs dataset exhausted".to_owned())
}

fn parse_row(row: Row) -> Result<DatasetSample, String> {
    let mut image = None;
    let mut label = None;
    for (_name, field) in row.get_column_iter() {
        match field {
            Field::Group(group) => {
                for (name, nested) in group.get_column_iter() {
                    if name == "bytes" {
                        if let Field::Bytes(bytes) = nested {
                            image = Some(bytes.data().to_vec());
                        }
                    }
                }
            }
            Field::Long(value) => label = Some(*value as u32),
            _ => {}
        }
    }
    Ok(DatasetSample {
        image: image.ok_or_else(|| "dataset row has no image bytes".to_owned())?,
        label: label.ok_or_else(|| "dataset row has no label".to_owned())?,
    })
}

fn image_tensor(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("decode dataset image: {error}"))?
        .resize_to_fill(
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let mut output = vec![0.0; 3 * IMAGE_SIZE * IMAGE_SIZE];
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for (index, pixel) in image.pixels().enumerate() {
        let y = index / IMAGE_SIZE;
        let x = index % IMAGE_SIZE;
        for channel in 0..3 {
            output[channel * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
        }
    }
    Ok(output)
}

#[derive(Flow)]
pub struct LogPolicy(Step<LogPrediction>);

pub struct LogPrediction;
#[async_trait]
pub trait RawArtifactOps: VoidOps {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String>;
}

#[jungle::action]
impl Action for LogPrediction {
    type Effect = LogPredictionEffect;
    type Input = ArtifactDelivery<Logits>;
    type Output = ();

    fn emit(_state: &ForwardSunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("classification policy failed: {error}")))
    }
}

pub struct LogPredictionEffect;
pub static LOGGED_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 202)]
impl<J: RawArtifactOps> Effect<J> for LogPredictionEffect {
    type In = ArtifactDelivery<Logits>;
    type Out = ();
    type Err = String;

    fn effect(
        jungle: &J,
        delivery: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let emission_bytes = jungle.download_raw(delivery.emission_id.id()).await?;
            let emission: Emission<SampleMetadata, Logits> =
                postcard::from_bytes(&emission_bytes).map_err(|e| e.to_string())?;
            let output_bytes = jungle.receive_raw_artifact(&emission.output_id).await?;
            let output = black_hole_sun::decode_output::<HeadOp>(&output_bytes)
                .map_err(|e| e.to_string())?;
            let logits = output.tensors[0]
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect::<Vec<_>>();
            let prediction = if logits[0] >= logits[1] {
                "corgi"
            } else {
                "not a corgi"
            };
            let expected = matches!(
                emission.metadata.dataset_label,
                PEMBROKE_LABEL | CARDIGAN_LABEL
            );
            info!(prediction, expected, "corgi-fwd classification");
            LOGGED_OUTPUTS.fetch_add(1, Ordering::Release);
            Ok(())
        }
    }
}

// ResNet-18 component builders. These follow Candle's resnet implementation,
// but expose its stem, four stages, and final pooling as independent modules.
fn conv2d(
    c_in: usize,
    c_out: usize,
    ksize: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder,
) -> candle::Result<candle_nn::Conv2d> {
    candle_nn::conv2d_no_bias(
        c_in,
        c_out,
        ksize,
        candle_nn::Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        },
        vb,
    )
}

fn downsample(
    c_in: usize,
    c_out: usize,
    stride: usize,
    vb: VarBuilder,
) -> candle::Result<Func<'static>> {
    if stride != 1 || c_in != c_out {
        let conv = conv2d(c_in, c_out, 1, 0, stride, vb.pp(0))?;
        let bn = batch_norm(c_out, 1e-5, vb.pp(1))?;
        Ok(Func::new(move |xs| xs.apply(&conv)?.apply_t(&bn, false)))
    } else {
        Ok(Func::new(|xs| Ok(xs.clone())))
    }
}

fn basic_block(
    c_in: usize,
    c_out: usize,
    stride: usize,
    vb: VarBuilder,
) -> candle::Result<Func<'static>> {
    let conv1 = conv2d(c_in, c_out, 3, 1, stride, vb.pp("conv1"))?;
    let bn1 = batch_norm(c_out, 1e-5, vb.pp("bn1"))?;
    let conv2 = conv2d(c_out, c_out, 3, 1, 1, vb.pp("conv2"))?;
    let bn2 = batch_norm(c_out, 1e-5, vb.pp("bn2"))?;
    let downsample = downsample(c_in, c_out, stride, vb.pp("downsample"))?;
    Ok(Func::new(move |xs| {
        let ys = xs
            .apply(&conv1)?
            .apply_t(&bn1, false)?
            .relu()?
            .apply(&conv2)?
            .apply_t(&bn2, false)?;
        (xs.apply(&downsample)? + ys)?.relu()
    }))
}

fn basic_layer(
    c_in: usize,
    c_out: usize,
    stride: usize,
    count: usize,
    vb: VarBuilder,
) -> candle::Result<Func<'static>> {
    let mut layers = Vec::with_capacity(count);
    for index in 0..count {
        layers.push(basic_block(
            if index == 0 { c_in } else { c_out },
            c_out,
            if index == 0 { stride } else { 1 },
            vb.pp(index),
        )?);
    }
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for layer in &layers {
            xs = xs.apply(layer)?;
        }
        Ok(xs)
    }))
}

pub fn build_stem(vb: VarBuilder) -> candle::Result<Func<'static>> {
    let conv = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
    let bn = batch_norm(64, 1e-5, vb.pp("bn1"))?;
    Ok(Func::new(move |xs| {
        xs.apply(&conv)?
            .apply_t(&bn, false)?
            .relu()?
            .pad_with_same(D::Minus1, 1, 1)?
            .pad_with_same(D::Minus2, 1, 1)?
            .max_pool2d_with_stride(3, 2)
    }))
}

pub fn build_stage1(vb: VarBuilder) -> candle::Result<Func<'static>> {
    basic_layer(64, 64, 1, 2, vb.pp("layer1"))
}
pub fn build_stage2(vb: VarBuilder) -> candle::Result<Func<'static>> {
    basic_layer(64, 128, 2, 2, vb.pp("layer2"))
}
pub fn build_stage3(vb: VarBuilder) -> candle::Result<Func<'static>> {
    basic_layer(128, 256, 2, 2, vb.pp("layer3"))
}
pub fn build_stage4(vb: VarBuilder) -> candle::Result<Func<'static>> {
    basic_layer(256, 512, 2, 2, vb.pp("layer4"))
}

pub fn build_head(vb: VarBuilder) -> candle::Result<Linear> {
    // Candle's published ResNet checkpoint has ImageNet's 1,000-way head.
    // Turn the two corgi breeds (Pembroke and Cardigan) into the positive
    // class and average all remaining ImageNet rows into the negative class.
    let weights = vb.get((1000, 512), "fc.weight")?;
    let biases = vb.get(1000, "fc.bias")?;
    let positive_weight =
        ((weights.i(PEMBROKE_LABEL as usize)? + weights.i(CARDIGAN_LABEL as usize)?)? / 2.0)?;
    let positive_bias =
        ((biases.i(PEMBROKE_LABEL as usize)? + biases.i(CARDIGAN_LABEL as usize)?)? / 2.0)?;
    let negative_weight = ((weights.sum(0)?
        - weights.i(PEMBROKE_LABEL as usize)?
        - weights.i(CARDIGAN_LABEL as usize)?)?
        / 998.0)?;
    let negative_bias = ((biases.sum(0)?
        - biases.i(PEMBROKE_LABEL as usize)?
        - biases.i(CARDIGAN_LABEL as usize)?)?
        / 998.0)?;
    Ok(Linear::new(
        Tensor::stack(&[&positive_weight, &negative_weight], 0)?,
        Some(Tensor::stack(&[&positive_bias, &negative_bias], 0)?),
    ))
}

pub fn pool_stage4(xs: &Tensor) -> candle::Result<Tensor> {
    xs.mean(D::Minus1)?.mean(D::Minus1)
}

pub type CorgiSun =
    <CorgiGraph as BlackHole>::Sun<ForwardOnlyWithPolicy<Generator, StemOp, HeadOp, LogPolicy>>;

pub struct CorgiForward;
impl Animal for CorgiForward {
    type Id = Id<U6>;
    type Generation = U0;
    type State = ForwardSunState;
    type Seed = ();
    type Flow = CorgiSun;
}
impl Observable for CorgiForward {
    type Observation = NoopObservation;
}
impl Perturbable for CorgiForward {
    type Perturbation = NoopPerturbation;
}
