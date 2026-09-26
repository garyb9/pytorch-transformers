use std::time::Instant;

use anyhow::Result;
use candle_core::Device;

use crate::infer::greedy_decode;
use crate::model::Transformer;
use crate::tokenizer::TokenizerWrapper;

pub struct BenchOutcome {
    pub outputs: Vec<String>,
    pub times: Vec<f64>,
}

pub struct BenchParams {
    pub bos: u32,
    pub eos: u32,
    pub max_len: usize,
    pub warmup: usize,
    pub reps: usize,
}

pub fn bench_translate(
    model: &Transformer,
    tokenizer: &TokenizerWrapper,
    texts: &[String],
    params: &BenchParams,
    device: &Device,
) -> Result<BenchOutcome> {
    let run = || -> Result<Vec<String>> {
        let mut outputs = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = tokenizer.encode(text)?;
            let generated =
                greedy_decode(model, &ids, params.bos, params.eos, params.max_len, device)?;
            outputs.push(tokenizer.decode(&generated)?.trim().to_string());
        }
        Ok(outputs)
    };

    for _ in 0..params.warmup {
        run()?;
    }

    let mut times = Vec::with_capacity(params.reps);
    let mut outputs = Vec::new();
    for _ in 0..params.reps {
        let start = Instant::now();
        outputs = run()?;
        times.push(start.elapsed().as_secs_f64());
    }

    Ok(BenchOutcome { outputs, times })
}

pub fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((fraction * (ordered.len() as f64 - 1.0)).round() as usize).min(ordered.len() - 1);
    ordered[index]
}
