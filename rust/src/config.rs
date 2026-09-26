use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

fn default_seq_len() -> usize {
    256
}

fn default_d_model() -> usize {
    512
}

fn default_n_layers() -> usize {
    6
}

fn default_n_heads() -> usize {
    8
}

fn default_d_ff() -> usize {
    2048
}

fn default_layer_norm_eps() -> f64 {
    1e-6
}

fn default_residual_mode() -> String {
    "post".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
    #[serde(default = "default_seq_len")]
    pub src_seq_len: usize,
    #[serde(default = "default_seq_len")]
    pub tgt_seq_len: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_d_ff")]
    pub d_ff: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_residual_mode")]
    pub residual_mode: String,
    #[serde(default)]
    pub tie_embeddings: bool,
    #[serde(default)]
    pub lang_embedding: bool,
}

impl ModelConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn is_pre_norm(&self) -> bool {
        self.residual_mode == "pre"
    }
}
