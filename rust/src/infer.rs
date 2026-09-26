use std::cmp::Ordering;

use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::ops;

use crate::model::{tgt_fill_mask, Transformer};

fn source_tensor(
    model: &Transformer,
    source_ids: &[u32],
    bos_id: u32,
    eos_id: u32,
    device: &Device,
) -> Result<(Tensor, usize)> {
    let limit = model.max_src_len()?.saturating_sub(2);
    let source_ids = &source_ids[..source_ids.len().min(limit)];
    let mut tokens = vec![bos_id];
    tokens.extend_from_slice(source_ids);
    tokens.push(eos_id);
    let length = tokens.len();
    Ok((Tensor::from_vec(tokens, (1, length), device)?, length))
}

pub fn greedy_decode(
    model: &Transformer,
    source_ids: &[u32],
    bos_id: u32,
    eos_id: u32,
    max_len: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let (src, _) = source_tensor(model, source_ids, bos_id, eos_id, device)?;
    let encoder_output = model.encode(&src, None, false)?;
    let max_len = max_len.min(model.max_tgt_len()?);

    let mut tokens = vec![bos_id];
    for _ in 0..max_len {
        let length = tokens.len();
        let tgt = Tensor::from_vec(tokens.clone(), (1, length), device)?;
        let tgt_mask = tgt_fill_mask(&tgt, u32::MAX)?;
        let output = model.decode(&encoder_output, None, &tgt, Some(&tgt_mask), false)?;
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

fn normalised_score(tokens: &[u32], log_prob: f64, penalty: f64) -> f64 {
    log_prob / (tokens.len() as f64).powf(penalty)
}

pub struct BeamParams {
    pub max_len: usize,
    pub beam_size: usize,
    pub length_penalty: f64,
}

pub fn beam_search(
    model: &Transformer,
    source_ids: &[u32],
    bos_id: u32,
    eos_id: u32,
    params: &BeamParams,
    device: &Device,
) -> Result<Vec<u32>> {
    let max_len = params.max_len.min(model.max_tgt_len()?);
    let beam_size = params.beam_size.max(1);
    let length_penalty = params.length_penalty;
    let (src, _) = source_tensor(model, source_ids, bos_id, eos_id, device)?;
    let encoder_output = model.encode(&src, None, false)?;

    let mut beams: Vec<(Vec<u32>, f64, bool)> = vec![(vec![bos_id], 0.0, false)];
    let mut completed: Vec<(Vec<u32>, f64)> = Vec::new();

    for _ in 0..max_len {
        if beams.iter().all(|(_, _, finished)| *finished) {
            break;
        }
        let mut candidates: Vec<(Vec<u32>, f64, bool)> = Vec::new();
        for (tokens, log_prob, finished) in &beams {
            if *finished {
                candidates.push((tokens.clone(), *log_prob, true));
                continue;
            }
            let length = tokens.len();
            let tgt = Tensor::from_vec(tokens.clone(), (1, length), device)?;
            let tgt_mask = tgt_fill_mask(&tgt, u32::MAX)?;
            let output = model.decode(&encoder_output, None, &tgt, Some(&tgt_mask), false)?;
            let last = output.narrow(1, length - 1, 1)?;
            let logits = model.project(&last)?.flatten_all()?;
            let log_probs = ops::log_softmax(&logits, D::Minus1)?.to_vec1::<f32>()?;
            let mut indexed: Vec<(u32, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            for (token_id, value) in indexed.into_iter().take(beam_size) {
                let mut new_tokens = tokens.clone();
                new_tokens.push(token_id);
                let new_log_prob = log_prob + value as f64;
                if token_id == eos_id {
                    completed.push((new_tokens, new_log_prob));
                } else {
                    candidates.push((new_tokens, new_log_prob, false));
                }
            }
        }

        candidates.sort_by(|a, b| {
            normalised_score(&b.0, b.1, length_penalty)
                .partial_cmp(&normalised_score(&a.0, a.1, length_penalty))
                .unwrap_or(Ordering::Equal)
        });
        candidates.truncate(beam_size);
        beams = candidates;
    }

    let pool = if completed.is_empty() {
        beams
            .into_iter()
            .map(|(tokens, log_prob, _)| (tokens, log_prob))
            .collect()
    } else {
        completed
    };
    let best = pool
        .into_iter()
        .max_by(|a, b| {
            normalised_score(&a.0, a.1, length_penalty)
                .partial_cmp(&normalised_score(&b.0, b.1, length_penalty))
                .unwrap_or(Ordering::Equal)
        })
        .expect("beam search produced no candidates");
    Ok(best.0[1..].to_vec())
}
