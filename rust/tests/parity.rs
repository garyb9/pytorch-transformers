use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use pytorch_transformers_rs::{src_fill_mask, tgt_fill_mask, ModelConfig, Transformer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Inputs {
    src: Vec<Vec<u32>>,
    tgt: Vec<Vec<u32>>,
    pad_id: u32,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity_small")
}

#[test]
fn forward_matches_python_logits() -> Result<()> {
    let dir = fixture_dir();
    let device = Device::Cpu;
    let config = ModelConfig::from_json_file(&dir.join("model.json"))?;
    let weights = dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = Transformer::load(&config, vb, &device)?;

    let raw = std::fs::read_to_string(dir.join("inputs.json"))?;
    let inputs: Inputs = serde_json::from_str(&raw)?;
    let batch = inputs.src.len();
    let src_len = inputs.src[0].len();
    let tgt_len = inputs.tgt[0].len();
    let src_data: Vec<u32> = inputs.src.iter().flatten().copied().collect();
    let tgt_data: Vec<u32> = inputs.tgt.iter().flatten().copied().collect();
    let src = Tensor::from_vec(src_data, (batch, src_len), &device)?;
    let tgt = Tensor::from_vec(tgt_data, (batch, tgt_len), &device)?;
    let src_mask = src_fill_mask(&src, inputs.pad_id)?;
    let tgt_mask = tgt_fill_mask(&tgt, inputs.pad_id)?;

    let logits = model.forward(&src, &tgt, Some(&src_mask), Some(&tgt_mask), false)?;

    let expected: HashMap<String, Tensor> =
        candle_core::safetensors::load(dir.join("expected.safetensors"), &device)?;
    let expected = &expected["logits"];
    assert_eq!(logits.dims(), expected.dims());

    let diff = (&logits - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "max abs diff vs python: {diff}");
    Ok(())
}
