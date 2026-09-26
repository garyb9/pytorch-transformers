use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Deserialize;

use crate::model::{src_fill_mask, tgt_fill_mask};

#[derive(Deserialize)]
struct Manifest {
    vocab_size: usize,
    shards: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct Record {
    src: Vec<u32>,
    tgt: Vec<u32>,
}

pub struct Example {
    pub src: Vec<u32>,
    pub tgt: Vec<u32>,
}

pub struct TranslationDataset {
    pub examples: Vec<Example>,
    pub seq_len: usize,
}

impl TranslationDataset {
    pub fn from_dir(data_dir: &Path, split: &str, seq_len: usize, direction: &str) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_str(&std::fs::read_to_string(data_dir.join("manifest.json"))?)?;
        if manifest.vocab_size == 0 {
            anyhow::bail!("manifest is missing vocab_size");
        }
        let names = manifest.shards.get(split).cloned().unwrap_or_default();
        let mut examples = Vec::new();
        for name in names {
            let text = std::fs::read_to_string(data_dir.join(&name))?;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: Record = serde_json::from_str(line)?;
                let (src, tgt) = if direction == "ja-en" {
                    (record.tgt, record.src)
                } else {
                    (record.src, record.tgt)
                };
                examples.push(Example { src, tgt });
            }
        }
        Ok(Self { examples, seq_len })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

pub struct Batch {
    pub encoder_input: Tensor,
    pub decoder_input: Tensor,
    pub label: Tensor,
    pub encoder_mask: Tensor,
    pub decoder_mask: Tensor,
}

pub fn make_batch(
    dataset: &TranslationDataset,
    indices: &[usize],
    pad: u32,
    bos: u32,
    eos: u32,
    device: &Device,
) -> Result<Batch> {
    let batch = indices.len();
    let seq_len = dataset.seq_len;
    let body = seq_len - 2;
    let mut encoder = vec![pad; batch * seq_len];
    let mut decoder = vec![pad; batch * seq_len];
    let mut label = vec![pad; batch * seq_len];

    for (row, &index) in indices.iter().enumerate() {
        let example = &dataset.examples[index];
        let src_body = &example.src[..example.src.len().min(body)];
        let tgt_body = &example.tgt[..example.tgt.len().min(body)];

        let mut encoder_row = vec![bos];
        encoder_row.extend_from_slice(src_body);
        encoder_row.push(eos);

        let mut decoder_row = vec![bos];
        decoder_row.extend_from_slice(tgt_body);

        let mut label_row = tgt_body.to_vec();
        label_row.push(eos);

        let offset = row * seq_len;
        encoder[offset..offset + encoder_row.len()].copy_from_slice(&encoder_row);
        decoder[offset..offset + decoder_row.len()].copy_from_slice(&decoder_row);
        label[offset..offset + label_row.len()].copy_from_slice(&label_row);
    }

    let encoder_input = Tensor::from_vec(encoder, (batch, seq_len), device)?;
    let decoder_input = Tensor::from_vec(decoder, (batch, seq_len), device)?;
    let label = Tensor::from_vec(label, (batch, seq_len), device)?;
    let encoder_mask = src_fill_mask(&encoder_input, pad)?;
    let decoder_mask = tgt_fill_mask(&decoder_input, pad)?;

    Ok(Batch {
        encoder_input,
        decoder_input,
        label,
        encoder_mask,
        decoder_mask,
    })
}
