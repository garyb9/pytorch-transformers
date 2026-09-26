use anyhow::Result;
use candle_core::{Device, Tensor, D};

use crate::model::{src_fill_mask, tgt_fill_mask, Transformer};

pub fn greedy_decode(
    model: &Transformer,
    source_ids: &[u32],
    bos_id: u32,
    eos_id: u32,
    max_len: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let source_limit = model.max_src_len()?.saturating_sub(2);
    let source_ids = &source_ids[..source_ids.len().min(source_limit)];
    let max_len = max_len.min(model.max_tgt_len()?);
    let mut source_tokens = vec![bos_id];
    source_tokens.extend_from_slice(source_ids);
    source_tokens.push(eos_id);
    let src = Tensor::from_vec(source_tokens, (1, source_ids.len() + 2), device)?;
    let encoder_output = model.encode(&src, None)?;

    let mut tokens = vec![bos_id];
    for _ in 0..max_len {
        let length = tokens.len();
        let tgt = Tensor::from_vec(tokens.clone(), (1, length), device)?;
        let tgt_mask = tgt_fill_mask(&tgt, u32::MAX)?;
        let output = model.decode(&encoder_output, None, &tgt, Some(&tgt_mask))?;
        let last = output.narrow(1, length - 1, 1)?;
        let logits = model.project(&last)?.flatten_all()?;
        let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        tokens.push(next);
        if next == eos_id {
            break;
        }
    }
    Ok(tokens[1..].to_vec())
}

pub fn source_mask(ids: &Tensor, pad_id: u32) -> Result<Tensor> {
    src_fill_mask(ids, pad_id)
}
