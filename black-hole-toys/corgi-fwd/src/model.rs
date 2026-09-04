//! ResNet-18 component builders. These follow Candle's resnet implementation,
//! but expose its stem, four stages, and final pooling as independent modules.

use candle::{IndexOp, Tensor, D};
use candle_nn::{batch_norm, Func, Linear, VarBuilder};

/// Stanford Dogs class for the Pembroke Welsh Corgi.
pub const PEMBROKE_LABEL: u32 = 111;
/// Stanford Dogs class for the Cardigan Welsh Corgi.
pub const CARDIGAN_LABEL: u32 = 112;

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

/// Builds the ResNet stem with the same overlapping 3x3 max pool expressed
/// using primitive differentiable ops. Candle's native max-pool backward only
/// supports equal kernel and stride sizes, while ResNet-18 uses stride 2.
pub fn build_trainable_stem(vb: VarBuilder) -> candle::Result<Func<'static>> {
    let conv = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
    let bn = batch_norm(64, 1e-5, vb.pp("bn1"))?;
    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv)?
            .apply_t(&bn, false)?
            .relu()?
            .pad_with_same(D::Minus1, 1, 1)?
            .pad_with_same(D::Minus2, 1, 1)?;
        let indexes = (0..56).map(|index| index * 2).collect::<Vec<u32>>();
        let indexes = Tensor::from_vec(indexes, (56,), xs.device())?;
        let mut windows = Vec::with_capacity(9);
        for y_offset in 0..3 {
            let y_indexes = (&indexes + y_offset as f64)?;
            let rows = xs.index_select(&y_indexes, 2)?;
            for x_offset in 0..3 {
                let x_indexes = (&indexes + x_offset as f64)?;
                windows.push(rows.index_select(&x_indexes, 3)?);
            }
        }
        Tensor::stack(&windows, 0)?.max(0)
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
