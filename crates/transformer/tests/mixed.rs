use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use pytorch_transformers_rs::{
    beam_search, greedy_decode, BeamParams, DecodeParams, LangPair, ModelConfig, Transformer,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Inputs {
    src: Vec<Vec<u32>>,
    tgt: Vec<Vec<u32>>,
    pad_id: u32,
    source_ids: Vec<u32>,
    max_len: usize,
    bos_id: u32,
    eos_id: u32,
    src_lang_id: Option<u32>,
    tgt_lang_id: Option<u32>,
    expected_greedy: Vec<u32>,
    expected_beam: Vec<u32>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity_mixed")
}

fn load_model(device: &Device) -> Result<(Transformer, Inputs)> {
    let dir = fixture_dir();
    let config = ModelConfig::from_json_file(&dir.join("model.json"))?;
    assert!(config.lang_embedding);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, device)?
    };
    let model = Transformer::load(&config, vb, device)?;
    let raw = std::fs::read_to_string(dir.join("inputs.json"))?;
    Ok((model, serde_json::from_str(&raw)?))
}

fn decode_params(inputs: &Inputs) -> DecodeParams {
    DecodeParams {
        bos: inputs.bos_id,
        eos: inputs.eos_id,
        max_len: inputs.max_len,
        src_lang: inputs.src_lang_id,
        tgt_lang: inputs.tgt_lang_id,
    }
}

#[test]
fn mixed_forward_matches_python_logits() -> Result<()> {
    let device = Device::Cpu;
    let (model, inputs) = load_model(&device)?;
    let batch = inputs.src.len();
    let src_len = inputs.src[0].len();
    let tgt_len = inputs.tgt[0].len();
    let src_data: Vec<u32> = inputs.src.iter().flatten().copied().collect();
    let tgt_data: Vec<u32> = inputs.tgt.iter().flatten().copied().collect();
    let src = Tensor::from_vec(src_data, (batch, src_len), &device)?;
    let tgt = Tensor::from_vec(tgt_data, (batch, tgt_len), &device)?;
    let src_mask = pytorch_transformers_rs::src_fill_mask(&src, inputs.pad_id)?;
    let tgt_mask = pytorch_transformers_rs::tgt_fill_mask(&tgt, inputs.pad_id)?;
    let langs = LangPair {
        src: inputs.src_lang_id,
        tgt: inputs.tgt_lang_id,
    };

    let logits = model.forward(&src, &tgt, Some(&src_mask), Some(&tgt_mask), langs, false)?;
    let expected: HashMap<String, Tensor> =
        candle_core::safetensors::load(fixture_dir().join("expected.safetensors"), &device)?;
    let expected = &expected["logits"];
    assert_eq!(logits.dims(), expected.dims());
    let diff = (&logits - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "max abs diff vs python: {diff}");
    Ok(())
}

#[test]
fn mixed_greedy_and_beam_match_python() -> Result<()> {
    let device = Device::Cpu;
    let (model, inputs) = load_model(&device)?;
    let params = decode_params(&inputs);

    let greedy = greedy_decode(&model, &inputs.source_ids, &params, &device)?;
    assert_eq!(greedy, inputs.expected_greedy);

    let beam = beam_search(
        &model,
        &inputs.source_ids,
        &BeamParams {
            decode: decode_params(&inputs),
            beam_size: 3,
            length_penalty: 0.6,
        },
        &device,
    )?;
    assert_eq!(beam, inputs.expected_beam);
    Ok(())
}
