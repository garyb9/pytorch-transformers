use anyhow::{bail, Result};
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, ops, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;

fn maybe_dropout(x: &Tensor, p: f32, training: bool) -> Result<Tensor> {
    if training && p > 0.0 {
        Ok(ops::dropout(x, p)?)
    } else {
        Ok(x.clone())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LangPair {
    pub src: Option<u32>,
    pub tgt: Option<u32>,
}

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: f32,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(d_model, d_ff, vb.pp("linear1"))?,
            linear2: linear(d_ff, d_model, vb.pp("linear2"))?,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let hidden = self.linear1.forward(x)?.relu()?;
        let hidden = maybe_dropout(&hidden, self.dropout, training)?;
        Ok(self.linear2.forward(&hidden)?)
    }
}

pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    n_heads: usize,
    d_k: usize,
    dropout: f32,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        if !d_model.is_multiple_of(n_heads) {
            bail!("d_model ({d_model}) must be divisible by n_heads ({n_heads})");
        }
        Ok(Self {
            w_q: linear(d_model, d_model, vb.pp("w_q"))?,
            w_k: linear(d_model, d_model, vb.pp("w_k"))?,
            w_v: linear(d_model, d_model, vb.pp("w_v"))?,
            w_o: linear(d_model, d_model, vb.pp("w_o"))?,
            n_heads,
            d_k: d_model / n_heads,
            dropout,
        })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        Ok(x.reshape((batch, seq_len, self.n_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let query = self.split_heads(&self.w_q.forward(q)?)?;
        let key = self.split_heads(&self.w_k.forward(k)?)?;
        let value = self.split_heads(&self.w_v.forward(v)?)?;
        let scale = (self.d_k as f64).sqrt();
        let scores = (query.matmul(&key.transpose(2, 3)?.contiguous()?)? / scale)?;
        let scores = match mask {
            Some(mask) => {
                let broadcast = mask.broadcast_as(scores.shape())?;
                let neg = Tensor::full(-1e9f32, scores.shape().clone(), scores.device())?;
                broadcast.where_cond(&neg, &scores)?
            }
            None => scores,
        };
        let weights = ops::softmax(&scores, D::Minus1)?;
        let weights = maybe_dropout(&weights, self.dropout, training)?;
        let context = weights.matmul(&value)?;
        let (batch, _heads, seq_len, _) = context.dims4()?;
        let context = context.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.n_heads * self.d_k,
        ))?;
        Ok(self.w_o.forward(&context)?)
    }
}

pub struct EncoderBlock {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: f32,
    pre_norm: bool,
}

impl EncoderBlock {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let dropout = config.dropout as f32;
        Ok(Self {
            self_attn: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                dropout,
                vb.pp("self_attn"),
            )?,
            ffn: FeedForward::new(config.d_model, config.d_ff, dropout, vb.pp("ffn"))?,
            norm1: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm1"))?,
            norm2: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm2"))?,
            dropout,
            pre_norm: config.is_pre_norm(),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        let attention = if self.pre_norm {
            let h = self.norm1.forward(x)?;
            self.self_attn.forward(&h, &h, &h, mask, training)?
        } else {
            self.self_attn.forward(x, x, x, mask, training)?
        };
        let attention = maybe_dropout(&attention, self.dropout, training)?;
        let x = if self.pre_norm {
            (x + &attention)?
        } else {
            self.norm1.forward(&(x + &attention)?)?
        };

        let ffn = if self.pre_norm {
            let h = self.norm2.forward(&x)?;
            self.ffn.forward(&h, training)?
        } else {
            self.ffn.forward(&x, training)?
        };
        let ffn = maybe_dropout(&ffn, self.dropout, training)?;
        if self.pre_norm {
            Ok((&x + &ffn)?)
        } else {
            Ok(self.norm2.forward(&(&x + &ffn)?)?)
        }
    }
}

pub struct DecoderBlock {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: f32,
    pre_norm: bool,
}

impl DecoderBlock {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let dropout = config.dropout as f32;
        Ok(Self {
            self_attn: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                dropout,
                vb.pp("self_attn"),
            )?,
            cross_attn: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                dropout,
                vb.pp("cross_attn"),
            )?,
            ffn: FeedForward::new(config.d_model, config.d_ff, dropout, vb.pp("ffn"))?,
            norm1: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm1"))?,
            norm2: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm2"))?,
            norm3: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm3"))?,
            dropout,
            pre_norm: config.is_pre_norm(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let self_attention = if self.pre_norm {
            let h = self.norm1.forward(x)?;
            self.self_attn.forward(&h, &h, &h, tgt_mask, training)?
        } else {
            self.self_attn.forward(x, x, x, tgt_mask, training)?
        };
        let self_attention = maybe_dropout(&self_attention, self.dropout, training)?;
        let x = if self.pre_norm {
            (x + &self_attention)?
        } else {
            self.norm1.forward(&(x + &self_attention)?)?
        };

        let cross_attention = if self.pre_norm {
            let h = self.norm2.forward(&x)?;
            self.cross_attn
                .forward(&h, encoder_output, encoder_output, src_mask, training)?
        } else {
            self.cross_attn
                .forward(&x, encoder_output, encoder_output, src_mask, training)?
        };
        let cross_attention = maybe_dropout(&cross_attention, self.dropout, training)?;
        let x = if self.pre_norm {
            (&x + &cross_attention)?
        } else {
            self.norm2.forward(&(&x + &cross_attention)?)?
        };

        let ffn = if self.pre_norm {
            let h = self.norm3.forward(&x)?;
            self.ffn.forward(&h, training)?
        } else {
            self.ffn.forward(&x, training)?
        };
        let ffn = maybe_dropout(&ffn, self.dropout, training)?;
        if self.pre_norm {
            Ok((&x + &ffn)?)
        } else {
            Ok(self.norm3.forward(&(&x + &ffn)?)?)
        }
    }
}

pub struct Transformer {
    src_embed: Embedding,
    tgt_embed: Embedding,
    lang_embed: Option<Embedding>,
    src_pos: Tensor,
    tgt_pos: Tensor,
    encoder_layers: Vec<EncoderBlock>,
    decoder_layers: Vec<DecoderBlock>,
    encoder_norm: LayerNorm,
    decoder_norm: LayerNorm,
    tgt_proj: Linear,
    d_model: usize,
    dropout: f32,
}

impl Transformer {
    pub fn load(config: &ModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let d_model = config.d_model;
        let src_embed = embedding(config.src_vocab_size, d_model, vb.pp("src_embed"))?;
        let tgt_embed = if config.tie_embeddings {
            let weight = vb
                .pp("src_embed")
                .get((config.src_vocab_size, d_model), "weight")?;
            Embedding::new(weight, d_model)
        } else {
            embedding(config.tgt_vocab_size, d_model, vb.pp("tgt_embed"))?
        };
        let src_pos = sinusoidal(config.src_seq_len, d_model, device)?;
        let tgt_pos = sinusoidal(config.tgt_seq_len, d_model, device)?;
        let lang_embed = if config.lang_embedding {
            Some(embedding(2, d_model, vb.pp("lang_embed"))?)
        } else {
            None
        };

        let mut encoder_layers = Vec::with_capacity(config.n_layers);
        let mut decoder_layers = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            encoder_layers.push(EncoderBlock::new(
                config,
                vb.pp(format!("encoder.layers.{index}")),
            )?);
            decoder_layers.push(DecoderBlock::new(
                config,
                vb.pp(format!("decoder.layers.{index}")),
            )?);
        }
        let encoder_norm = layer_norm(d_model, config.layer_norm_eps, vb.pp("encoder.norm"))?;
        let decoder_norm = layer_norm(d_model, config.layer_norm_eps, vb.pp("decoder.norm"))?;

        let tgt_proj = if config.tie_embeddings {
            let weight = vb
                .pp("src_embed")
                .get((config.src_vocab_size, d_model), "weight")?;
            let bias = vb.pp("tgt_proj").get(config.tgt_vocab_size, "bias")?;
            Linear::new(weight, Some(bias))
        } else {
            linear(d_model, config.tgt_vocab_size, vb.pp("tgt_proj"))?
        };

        Ok(Self {
            src_embed,
            tgt_embed,
            lang_embed,
            src_pos,
            tgt_pos,
            encoder_layers,
            decoder_layers,
            encoder_norm,
            decoder_norm,
            tgt_proj,
            d_model,
            dropout: config.dropout as f32,
        })
    }

    fn add_lang(&self, x: &Tensor, lang: Option<u32>) -> Result<Tensor> {
        match (self.lang_embed.as_ref(), lang) {
            (Some(embed), Some(id)) => {
                let ids = Tensor::from_vec(vec![id], (1,), x.device())?;
                let vector = embed.forward(&ids)?.unsqueeze(1)?;
                Ok(x.broadcast_add(&vector)?)
            }
            _ => Ok(x.clone()),
        }
    }

    fn add_positional(&self, x: &Tensor, positional: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let pe = positional.narrow(1, 0, seq_len)?;
        Ok(x.broadcast_add(&pe)?)
    }

    pub fn encode(
        &self,
        src: &Tensor,
        src_mask: Option<&Tensor>,
        src_lang: Option<u32>,
        training: bool,
    ) -> Result<Tensor> {
        let scale = (self.d_model as f64).sqrt();
        let mut x = (self.src_embed.forward(src)? * scale)?;
        x = self.add_positional(&x, &self.src_pos)?;
        x = self.add_lang(&x, src_lang)?;
        x = maybe_dropout(&x, self.dropout, training)?;
        for layer in &self.encoder_layers {
            x = layer.forward(&x, src_mask, training)?;
        }
        Ok(self.encoder_norm.forward(&x)?)
    }

    pub fn decode(
        &self,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt: &Tensor,
        tgt_mask: Option<&Tensor>,
        tgt_lang: Option<u32>,
        training: bool,
    ) -> Result<Tensor> {
        let scale = (self.d_model as f64).sqrt();
        let mut x = (self.tgt_embed.forward(tgt)? * scale)?;
        x = self.add_positional(&x, &self.tgt_pos)?;
        x = self.add_lang(&x, tgt_lang)?;
        x = maybe_dropout(&x, self.dropout, training)?;
        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output, src_mask, tgt_mask, training)?;
        }
        Ok(self.decoder_norm.forward(&x)?)
    }

    pub fn project(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.tgt_proj.forward(x)?)
    }

    pub fn max_src_len(&self) -> Result<usize> {
        Ok(self.src_pos.dim(1)?)
    }

    pub fn max_tgt_len(&self) -> Result<usize> {
        Ok(self.tgt_pos.dim(1)?)
    }

    pub fn forward(
        &self,
        src: &Tensor,
        tgt: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        langs: LangPair,
        training: bool,
    ) -> Result<Tensor> {
        let encoder_output = self.encode(src, src_mask, langs.src, training)?;
        let decoder_output = self.decode(
            &encoder_output,
            src_mask,
            tgt,
            tgt_mask,
            langs.tgt,
            training,
        )?;
        self.project(&decoder_output)
    }
}

pub fn sinusoidal(seq_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * d_model];
    for position in 0..seq_len {
        for index in 0..d_model / 2 {
            let angle = position as f32 / 10000f32.powf(2.0 * index as f32 / d_model as f32);
            data[position * d_model + 2 * index] = angle.sin();
            data[position * d_model + 2 * index + 1] = angle.cos();
        }
    }
    Ok(Tensor::from_vec(data, (1, seq_len, d_model), device)?)
}

pub fn src_fill_mask(ids: &Tensor, pad_id: u32) -> Result<Tensor> {
    let (batch, seq_len) = ids.dims2()?;
    let values = ids.to_vec2::<u32>()?;
    let mut data = vec![0u8; batch * seq_len];
    for row in 0..batch {
        for column in 0..seq_len {
            if values[row][column] == pad_id {
                data[row * seq_len + column] = 1;
            }
        }
    }
    Ok(Tensor::from_vec(
        data,
        (batch, 1, 1, seq_len),
        ids.device(),
    )?)
}

pub fn tgt_fill_mask(ids: &Tensor, pad_id: u32) -> Result<Tensor> {
    let (batch, seq_len) = ids.dims2()?;
    let values = ids.to_vec2::<u32>()?;
    let mut data = vec![0u8; batch * seq_len * seq_len];
    for row in 0..batch {
        for query in 0..seq_len {
            for key in 0..seq_len {
                if key > query || values[row][key] == pad_id {
                    data[row * seq_len * seq_len + query * seq_len + key] = 1;
                }
            }
        }
    }
    Ok(Tensor::from_vec(
        data,
        (batch, 1, seq_len, seq_len),
        ids.device(),
    )?)
}
