use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{ops, VarBuilder, VarMap};
use rand::RngExt;

use crate::config::ModelConfig;
use crate::dataset::{make_batch, TranslationDataset};
use crate::model::{LangPair, Transformer};

pub struct TrainOptions {
    pub data_dir: PathBuf,
    pub split: String,
    pub direction: String,
    pub seq_len: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub label_smoothing: f64,
}

pub fn scheduled_lr(step: usize, warmup_steps: usize, base_lr: f64) -> f64 {
    let warmup = warmup_steps.max(1) as f64;
    let current = (step + 1) as f64;
    let peak = warmup.powf(-0.5);
    let scale = current.powf(-0.5).min(current * warmup.powf(-1.5));
    base_lr * scale / peak
}

pub fn label_smoothed_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: u32,
    smoothing: f64,
) -> Result<Tensor> {
    let (_rows, vocab) = logits.dims2()?;
    let log_probs = ops::log_softmax(logits, D::Minus1)?;
    let gathered = log_probs
        .gather(&targets.unsqueeze(1)?, D::Minus1)?
        .squeeze(1)?
        .neg()?;
    let smoothed = (log_probs.sum(D::Minus1)?.neg()? / vocab as f64)?;
    let per_row = (gathered.affine(1.0 - smoothing, 0.0)? + smoothed.affine(smoothing, 0.0)?)?;

    let mask_values: Vec<f32> = targets
        .to_vec1::<u32>()?
        .iter()
        .map(|token| if *token == ignore_index { 0.0 } else { 1.0 })
        .collect();
    let mask = Tensor::from_vec(mask_values, per_row.shape(), per_row.device())?;
    let total = (&per_row * &mask)?.sum_all()?.to_scalar::<f32>()?;
    let count = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
    Ok(Tensor::new((total / count) as f32, per_row.device())?)
}

pub fn train_model(
    config: &ModelConfig,
    options: &TrainOptions,
    device: &Device,
    pad: u32,
    bos: u32,
    eos: u32,
    out_dir: &Path,
) -> Result<f32> {
    let dataset = TranslationDataset::from_dir(
        &options.data_dir,
        &options.split,
        options.seq_len,
        &options.direction,
    )?;
    if dataset.is_empty() {
        anyhow::bail!("empty training split: {}", options.split);
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Transformer::load(config, vb, device)?;

    let params = ParamsAdamW {
        lr: options.lr,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    let mut rng = rand::rng();
    let mut last_loss = f32::NAN;
    for step in 0..options.steps {
        optimizer.set_params(ParamsAdamW {
            lr: scheduled_lr(step, options.warmup_steps, options.lr),
            ..Default::default()
        });
        let indices: Vec<usize> = (0..options.batch_size)
            .map(|_| rng.random_range(0..dataset.len()))
            .collect();
        let batch = make_batch(&dataset, &indices, pad, bos, eos, device)?;
        let logits = model.forward(
            &batch.encoder_input,
            &batch.decoder_input,
            Some(&batch.encoder_mask),
            Some(&batch.decoder_mask),
            LangPair::default(),
            true,
        )?;
        let (rows, seq_len, vocab) = logits.dims3()?;
        let flat_logits = logits.reshape((rows * seq_len, vocab))?;
        let flat_labels = batch.label.reshape((rows * seq_len,))?;
        let loss =
            label_smoothed_cross_entropy(&flat_logits, &flat_labels, pad, options.label_smoothing)?;
        optimizer.backward_step(&loss)?;
        last_loss = loss.to_scalar::<f32>()?;
        if step % 10 == 0 {
            eprintln!("step {step} loss {last_loss:.4}");
        }
    }

    std::fs::create_dir_all(out_dir)?;
    varmap.save(out_dir.join("weights.safetensors"))?;
    let mut payload = serde_json::to_value(config)?;
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "direction".to_string(),
            serde_json::json!(options.direction),
        );
    }
    std::fs::write(
        out_dir.join("model.json"),
        serde_json::to_string_pretty(&payload)?,
    )?;
    Ok(last_loss)
}
