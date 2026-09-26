use anyhow::{bail, Result};
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, ops, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(d_model, d_ff, vb.pp("linear1"))?,
            linear2: linear(d_ff, d_model, vb.pp("linear2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(x)?.relu()?;
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
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
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
    pre_norm: bool,
}

impl EncoderBlock {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(config.d_model, config.n_heads, vb.pp("self_attn"))?,
            ffn: FeedForward::new(config.d_model, config.d_ff, vb.pp("ffn"))?,
            norm1: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm1"))?,
            norm2: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm2"))?,
            pre_norm: config.is_pre_norm(),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.residual(x, mask)?;
        self.residual_ffn(&x)
    }

    fn residual(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if self.pre_norm {
            let h = self.norm1.forward(x)?;
            let attn = self.self_attn.forward(&h, &h, &h, mask)?;
            Ok((x + &attn)?)
        } else {
            let attn = self.self_attn.forward(x, x, x, mask)?;
            let summed = (x + &attn)?;
            Ok(self.norm1.forward(&summed)?)
        }
    }

    fn residual_ffn(&self, x: &Tensor) -> Result<Tensor> {
        if self.pre_norm {
            let h = self.norm2.forward(x)?;
            let ffn = self.ffn.forward(&h)?;
            Ok((x + &ffn)?)
        } else {
            let ffn = self.ffn.forward(x)?;
            let summed = (x + &ffn)?;
            Ok(self.norm2.forward(&summed)?)
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
    pre_norm: bool,
}

impl DecoderBlock {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(config.d_model, config.n_heads, vb.pp("self_attn"))?,
            cross_attn: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                vb.pp("cross_attn"),
            )?,
            ffn: FeedForward::new(config.d_model, config.d_ff, vb.pp("ffn"))?,
            norm1: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm1"))?,
            norm2: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm2"))?,
            norm3: layer_norm(config.d_model, config.layer_norm_eps, vb.pp("norm3"))?,
            pre_norm: config.is_pre_norm(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = if self.pre_norm {
            let h = self.norm1.forward(x)?;
            let attn = self.self_attn.forward(&h, &h, &h, tgt_mask)?;
            (x + &attn)?
        } else {
            let attn = self.self_attn.forward(x, x, x, tgt_mask)?;
            self.norm1.forward(&(x + &attn)?)?
        };

        let x = if self.pre_norm {
            let h = self.norm2.forward(&x)?;
            let attn = self
                .cross_attn
                .forward(&h, encoder_output, encoder_output, src_mask)?;
            (&x + &attn)?
        } else {
            let attn = self
                .cross_attn
                .forward(&x, encoder_output, encoder_output, src_mask)?;
            self.norm2.forward(&(&x + &attn)?)?
        };

        if self.pre_norm {
            let h = self.norm3.forward(&x)?;
            let ffn = self.ffn.forward(&h)?;
            Ok((&x + &ffn)?)
        } else {
            let ffn = self.ffn.forward(&x)?;
            Ok(self.norm3.forward(&(&x + &ffn)?)?)
        }
    }
}

pub struct Transformer {
    src_embed: Embedding,
    tgt_embed: Embedding,
    src_pos: Tensor,
    tgt_pos: Tensor,
    encoder_layers: Vec<EncoderBlock>,
    decoder_layers: Vec<DecoderBlock>,
    encoder_norm: LayerNorm,
    decoder_norm: LayerNorm,
    tgt_proj: Linear,
    d_model: usize,
}

impl Transformer {
    pub fn load(config: &ModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        if config.lang_embedding {
            bail!("lang_embedding is reserved for mixed-direction mode");
        }
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
            src_pos,
            tgt_pos,
            encoder_layers,
            decoder_layers,
            encoder_norm,
            decoder_norm,
            tgt_proj,
            d_model,
        })
    }

    fn add_positional(&self, x: &Tensor, positional: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let pe = positional.narrow(1, 0, seq_len)?;
        Ok(x.broadcast_add(&pe)?)
    }

    pub fn encode(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Result<Tensor> {
        let scale = (self.d_model as f64).sqrt();
        let mut x = (self.src_embed.forward(src)? * scale)?;
        x = self.add_positional(&x, &self.src_pos)?;
        for layer in &self.encoder_layers {
            x = layer.forward(&x, src_mask)?;
        }
        Ok(self.encoder_norm.forward(&x)?)
    }

    pub fn decode(
        &self,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt: &Tensor,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let scale = (self.d_model as f64).sqrt();
        let mut x = (self.tgt_embed.forward(tgt)? * scale)?;
        x = self.add_positional(&x, &self.tgt_pos)?;
        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output, src_mask, tgt_mask)?;
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
    ) -> Result<Tensor> {
        let encoder_output = self.encode(src, src_mask)?;
        let decoder_output = self.decode(&encoder_output, src_mask, tgt, tgt_mask)?;
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
