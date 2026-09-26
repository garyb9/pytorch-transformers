use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use pytorch_transformers_rs::{greedy_decode, ModelConfig, TokenizerWrapper, Transformer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Inputs {
    source_ids: Vec<u32>,
    max_len: usize,
    bos_id: u32,
    eos_id: u32,
    expected_greedy: Vec<u32>,
}

#[derive(Deserialize)]
struct TokenizerCases {
    texts: Vec<String>,
    ids: Vec<Vec<u32>>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity_small")
}

fn load_model(device: &Device) -> Result<Transformer> {
    let dir = fixture_dir();
    let config = ModelConfig::from_json_file(&dir.join("model.json"))?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, device)?
    };
    Transformer::load(&config, vb, device)
}

#[test]
fn greedy_matches_python() -> Result<()> {
    let device = Device::Cpu;
    let model = load_model(&device)?;
    let raw = std::fs::read_to_string(fixture_dir().join("inputs.json"))?;
    let inputs: Inputs = serde_json::from_str(&raw)?;

    let output = greedy_decode(
        &model,
        &inputs.source_ids,
        inputs.bos_id,
        inputs.eos_id,
        inputs.max_len,
        &device,
    )?;
    assert_eq!(output, inputs.expected_greedy);
    Ok(())
}

#[test]
fn tokenizer_matches_python() -> Result<()> {
    let tokenizer = TokenizerWrapper::from_file(&fixture_dir().join("tokenizer.json"))?;
    let raw = std::fs::read_to_string(fixture_dir().join("tokenizer_cases.json"))?;
    let cases: TokenizerCases = serde_json::from_str(&raw)?;
    for index in 0..cases.texts.len() {
        let ids = tokenizer.encode(&cases.texts[index])?;
        assert_eq!(
            ids, cases.ids[index],
            "mismatch for {:?}",
            cases.texts[index]
        );
    }
    Ok(())
}
