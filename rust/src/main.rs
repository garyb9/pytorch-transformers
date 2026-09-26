use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand};
use pytorch_transformers_rs::{
    beam_search, bench_translate, greedy_decode, percentile, train_model, BeamParams, BenchParams,
    ModelConfig, TokenizerWrapper, TrainOptions, Transformer,
};

#[derive(Parser)]
#[command(
    name = "ptr",
    about = "EN<->JP transformer inference and training (candle)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Translate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        max_len: usize,
        #[arg(long, default_value_t = 1)]
        beam: usize,
        #[arg(long, default_value = "auto")]
        device: String,
    },
    Train {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "train")]
        split: String,
        #[arg(long, default_value = "en-ja")]
        direction: String,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 100)]
        warmup_steps: usize,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        out: PathBuf,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long = "text")]
        text: Vec<String>,
        #[arg(long, default_value_t = 32)]
        max_len: usize,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value = "auto")]
        device: String,
    },
}

fn resolve_device(name: &str) -> Result<Device> {
    Ok(match name {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        "metal" => Device::new_metal(0)?,
        _ => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
    })
}

fn load_model(model: &Path, config: &Path, device: &Device) -> Result<Transformer> {
    let config = ModelConfig::from_json_file(config)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, device)? };
    Transformer::load(&config, vb, device)
}

fn read_inputs(text: Option<String>, file: Option<PathBuf>) -> Result<Vec<String>> {
    if let Some(value) = text {
        return Ok(vec![value]);
    }
    let content = match file {
        Some(path) => std::fs::read_to_string(path)?,
        None => {
            let mut buffer = String::new();
            std::io::stdin().read_to_string(&mut buffer)?;
            buffer
        }
    };
    Ok(content.lines().map(str::to_string).collect())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Translate {
            model,
            config,
            tokenizer,
            text,
            file,
            max_len,
            beam,
            device,
        } => {
            let device = resolve_device(&device)?;
            let model = load_model(&model, &config, &device)?;
            let tokenizer = TokenizerWrapper::from_file(&tokenizer)?;
            let bos = tokenizer.bos_id()?;
            let eos = tokenizer.eos_id()?;
            for line in read_inputs(text, file)? {
                if line.trim().is_empty() {
                    continue;
                }
                let ids = tokenizer.encode(&line)?;
                let output = if beam > 1 {
                    beam_search(
                        &model,
                        &ids,
                        bos,
                        eos,
                        &BeamParams {
                            max_len,
                            beam_size: beam,
                            length_penalty: 0.6,
                        },
                        &device,
                    )?
                } else {
                    greedy_decode(&model, &ids, bos, eos, max_len, &device)?
                };
                println!("{}", tokenizer.decode(&output)?.trim());
            }
        }
        Command::Train {
            config,
            data_dir,
            tokenizer,
            split,
            direction,
            seq_len,
            batch_size,
            steps,
            lr,
            warmup_steps,
            device,
            out,
        } => {
            let device = resolve_device(&device)?;
            let model_config = ModelConfig::from_json_file(&config)?;
            let tokenizer = TokenizerWrapper::from_file(&tokenizer)?;
            let options = TrainOptions {
                data_dir,
                split,
                direction,
                seq_len,
                batch_size,
                steps,
                lr,
                warmup_steps,
                label_smoothing: 0.1,
            };
            let loss = train_model(
                &model_config,
                &options,
                &device,
                tokenizer.pad_id()?,
                tokenizer.bos_id()?,
                tokenizer.eos_id()?,
                &out,
            )?;
            println!("final loss {loss:.4}");
        }
        Command::Bench {
            model,
            config,
            tokenizer,
            text,
            max_len,
            reps,
            warmup,
            device,
        } => {
            let device = resolve_device(&device)?;
            let model = load_model(&model, &config, &device)?;
            let tokenizer = TokenizerWrapper::from_file(&tokenizer)?;
            let texts = if text.is_empty() {
                vec!["hello world".to_string(), "こんにちは世界".to_string()]
            } else {
                text
            };
            let outcome = bench_translate(
                &model,
                &tokenizer,
                &texts,
                &BenchParams {
                    bos: tokenizer.bos_id()?,
                    eos: tokenizer.eos_id()?,
                    max_len,
                    warmup,
                    reps,
                },
                &device,
            )?;
            let timings = serde_json::json!({
                "stack": "rust",
                "device": device_label(&device),
                "reps": reps,
                "warmup": warmup,
                "median_s": percentile(&outcome.times, 0.5),
                "p95_s": percentile(&outcome.times, 0.95),
                "min_s": outcome.times.iter().cloned().fold(f64::INFINITY, f64::min),
                "outputs": outcome.outputs,
            });
            println!("{}", serde_json::to_string(&timings)?);
        }
    }
    Ok(())
}

fn device_label(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}
