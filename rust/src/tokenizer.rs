use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("failed to load tokenizer {path:?}: {error}"))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|error| anyhow::anyhow!("detokenization failed: {error}"))
    }

    fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .with_context(|| format!("token not found: {token}"))
    }

    pub fn pad_id(&self) -> Result<u32> {
        self.token_id("[PAD]")
    }

    pub fn bos_id(&self) -> Result<u32> {
        self.token_id("[BOS]")
    }

    pub fn eos_id(&self) -> Result<u32> {
        self.token_id("[EOS]")
    }
}
