use anyhow::Result;
use candle_core::Device;
use pytorch_transformers_rs::{scheduled_lr, train_model, ModelConfig, TrainOptions};

#[test]
fn scheduled_lr_peaks_at_warmup_then_decays() {
    let peak = (0..10)
        .map(|step| scheduled_lr(step, 10, 1e-3))
        .fold(0.0f64, f64::max);
    assert!((peak - 1e-3).abs() < 1e-4, "peak {peak}");
    assert!(scheduled_lr(0, 10, 1e-3) < peak);
    assert!(scheduled_lr(20, 10, 1e-3) < peak);
}

#[test]
fn training_smoke_writes_checkpoint() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir)?;

    let mut lines = String::new();
    for index in 0..24u32 {
        let src = [10 + index % 5, 20 + index % 7, 30 + index % 3];
        let tgt = [40 + index % 4, 50 + index % 6, 60 + index % 2];
        lines.push_str(&format!(
            "{{\"src\":{:?},\"tgt\":{:?},\"origin\":\"synthetic\",\"pair_id\":{index}}}\n",
            src, tgt
        ));
    }
    std::fs::write(data_dir.join("train-00000.jsonl"), lines)?;
    let manifest = serde_json::json!({
        "vocab_size": 172,
        "shards": {"train": ["train-00000.jsonl"]},
    });
    std::fs::write(
        data_dir.join("manifest.json"),
        serde_json::to_string(&manifest)?,
    )?;

    let config = ModelConfig {
        src_vocab_size: 172,
        tgt_vocab_size: 172,
        src_seq_len: 8,
        tgt_seq_len: 8,
        d_model: 16,
        n_layers: 1,
        n_heads: 2,
        d_ff: 32,
        dropout: 0.0,
        layer_norm_eps: 1e-6,
        residual_mode: "post".to_string(),
        tie_embeddings: false,
        lang_embedding: false,
        direction: None,
    };
    let options = TrainOptions {
        data_dir: data_dir.clone(),
        split: "train".to_string(),
        direction: "en-ja".to_string(),
        seq_len: 8,
        batch_size: 4,
        steps: 20,
        lr: 1e-2,
        warmup_steps: 5,
        label_smoothing: 0.1,
    };
    let out_dir = dir.path().join("run");

    let loss = train_model(&config, &options, &Device::Cpu, 0, 2, 3, &out_dir)?;
    assert!(loss.is_finite(), "loss should be finite, got {loss}");
    assert!(out_dir.join("weights.safetensors").exists());
    assert!(out_dir.join("model.json").exists());
    Ok(())
}
