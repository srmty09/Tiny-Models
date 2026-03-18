import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import torch.optim as optim


@dataclass
class TinyLlamaConfig:
  hidden_dim:int = 768
  n_heads:int = 12
  n_layer:int = 12
  head_dim:int = 64
  batch_size: int = 32
  max_position_embedding: int = 256
  rms_norm_eps:float = 1e-5
  attention_dropout: float = 0.1
  rope_theta:int = 1e4
  device:str = "cuda" if torch.cuda.is_available() else "cpu"
  intermediate_dim: int = 1024
  n_kv_heads:int = 4
  vocab_size: int  = 10000
  padding_idx: int = 2

  epochs: int = 10
  lr: float = 3e-4
  min_lr: float = 3e-5
  weight_decay: float = 0.1
  warmup_steps: int = 100
  max_grad_norm: float = 1.0
  log_interval: int = 50
  eval_interval: int = 500
  val_batches: int = 50
  save_interval: int = 0
  grad_accum_steps: int = 1
  use_amp: bool = True
  use_compile: bool = True
  wandb_project: str = "tiny-llama"

  def __post_init__(self):
    self.head_dim = self.hidden_dim // self.n_heads


class TinyLlamaRMSNorm(nn.Module):
  def __init__(self,config:TinyLlamaConfig):
    super().__init__()
    self.cfg = config
    self.eps = config.rms_norm_eps

    # gamma
    self.weights = nn.Parameter(torch.ones(config.hidden_dim))


  def forward(self,hidden_state):
    variance = torch.pow(hidden_state,2).mean(-1,keepdim=True)
    hidden_state = hidden_state * torch.rsqrt(variance+self.eps)
    return self.weights*hidden_state

class TinyLlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.config = config
        inv_freq = self._compute_default_rope_parameters(self.config, device=self.config.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_default_rope_parameters(self, config, device):
        base = config.rope_theta
        head_dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).to(device, dtype=torch.float) / head_dim))
        return inv_freq

    @torch.no_grad()
    def forward(self, position_ids):
        with torch.amp.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = torch.einsum("i, j -> ij", position_ids.reshape(-1).float(), self.inv_freq.float())
            freqs = freqs.view(position_ids.shape[0], position_ids.shape[1], -1)
            freqs_cis = torch.polar(torch.ones_like(freqs), angle=freqs)
            return freqs_cis

def apply_rotary_emb(xq, xk, freq_cis):
    # xq shape: (batch, seq, n_heads, head_dim)
    # freq_cis shape: (batch, seq, head_dim/2)

    # Reshape xq and xk to (..., head_dim/2, 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freq_cis needs to broadcast from (batch, seq, head_dim/2) to (batch, seq, n_heads, head_dim/2)
    freq_cis = freq_cis.unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freq_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class TinyLlamaMLP(nn.Module):
  def __init__(self,config:TinyLlamaConfig):
    super().__init__()
    self.cfg = config
    self.gate_proj = nn.Linear(config.hidden_dim,config.intermediate_dim)
    self.up_proj = nn.Linear(config.hidden_dim,config.intermediate_dim)
    self.down_proj = nn.Linear(config.intermediate_dim,config.hidden_dim)
    self.act = nn.SiLU()


  def forward(self,x):
    out = self.down_proj(self.act(self.gate_proj(x))*self.up_proj(x))
    return out


def repeat_kv(hidden_state,n_rep):
  """
  here hidden_state is the input k-v which we want to repeat
  n_rep is the number of repeatation:
        i.e. the query has total n_head but we have only k number of head for k and v hence we need to repeat
        => n_rep = n_head/k
  """
  batch_dim, seq_len, kv_heads, kv_head_dim = hidden_state.shape
  # first unsqueeze
  # shape: (batch_size, seq_len, kv_head, kv_head_dim)
  hidden_state = hidden_state.unsqueeze(3)


  # expand:
  hidden_state = hidden_state.expand(batch_dim,seq_len,kv_heads,n_rep,kv_head_dim)

  # merge
  hidden_state = hidden_state.reshape(batch_dim,seq_len,kv_heads*n_rep,kv_head_dim)


  return hidden_state



class TinyLlamaGQA(nn.Module):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.hidden_dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

        self.k_cache = None
        self.v_cache = None

    def forward(self,
                hidden_state,
                freqs_cis,
                attention_mask=None
                  ):
        bd, sl, hd = hidden_state.shape

        q = self.wq(hidden_state).view(bd, sl, self.n_heads, self.head_dim)
        k = self.wk(hidden_state).view(bd, sl, self.n_kv_heads, self.head_dim)
        v = self.wv(hidden_state).view(bd, sl, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask == None:
          mask = torch.full((sl, sl), float("-inf"), device=hidden_state.device)
          mask = torch.triu(mask, diagonal=1)
        else:
          mask = attention_mask
        attn_weights = attn_weights + mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bd, sl, self.n_heads * self.head_dim)

        return self.wo(out), attn_weights

    def prefill(self, hidden_state, freqs_cis):
        """Process full prompt at once and populate KV cache."""
        bd, sl, hd = hidden_state.shape

        q = self.wq(hidden_state).view(bd, sl, self.n_heads, self.head_dim)
        k = self.wk(hidden_state).view(bd, sl, self.n_kv_heads, self.head_dim)
        v = self.wv(hidden_state).view(bd, sl, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        self.k_cache = k
        self.v_cache = v

        k_exp = repeat_kv(k, self.n_rep)
        v_exp = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k_exp = k_exp.transpose(1, 2)
        v_exp = v_exp.transpose(1, 2)

        attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.full((sl, sl), float("-inf"), device=hidden_state.device)
        mask = torch.triu(mask, diagonal=1)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(bd, sl, self.n_heads * self.head_dim)
        return self.wo(out)

    def inference(self, x, freqs_cis):
        """Decode one token, appending to KV cache."""
        bd, sl, hd = x.shape
        assert sl == 1, "seq_len must be 1 for cached decoding"

        qi = self.wq(x).view(bd, sl, self.n_heads, self.head_dim)
        ki = self.wk(x).view(bd, sl, self.n_kv_heads, self.head_dim)
        vi = self.wv(x).view(bd, sl, self.n_kv_heads, self.head_dim)

        qi, ki = apply_rotary_emb(qi, ki, freqs_cis)

        self.k_cache = torch.cat([self.k_cache, ki], dim=1)
        self.v_cache = torch.cat([self.v_cache, vi], dim=1)

        k = repeat_kv(self.k_cache, self.n_rep)
        v = repeat_kv(self.v_cache, self.n_rep)

        qi = qi.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(qi, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(qi)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bd, sl, self.n_heads * self.head_dim)
        return self.wo(out)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None



class TinyLlamaDecoderLayer(nn.Module):
  def __init__(self,config:TinyLlamaConfig):
    super().__init__()
    self.cfg = config
    self.hidden_dim = config.hidden_dim
    # attention
    self.attn = TinyLlamaGQA(config)
    # mlp
    self.mlp = TinyLlamaMLP(config)
    # pre layer norm
    self.pre_layer_norm = TinyLlamaRMSNorm(config)
    # post layer norm
    self.post_layer_norm = TinyLlamaRMSNorm(config)

  def forward(self,
              hidden_state,
              freqs_cis,
              attention_mask=None,
              ):
    residual = hidden_state
    hidden_state = self.pre_layer_norm(hidden_state)
    atten_out, _ = self.attn(hidden_state, freqs_cis, attention_mask)
    hidden_state = residual + atten_out

    residual = hidden_state
    hidden_state = self.post_layer_norm(hidden_state)
    out_ffn = self.mlp(hidden_state)
    return residual + out_ffn

  def prefill(self, hidden_state, freqs_cis):
    residual = hidden_state
    hidden_state = self.pre_layer_norm(hidden_state)
    attn_out = self.attn.prefill(hidden_state, freqs_cis)
    hidden_state = residual + attn_out

    residual = hidden_state
    hidden_state = self.post_layer_norm(hidden_state)
    return residual + self.mlp(hidden_state)

  def decode(self, hidden_state, freqs_cis):
    residual = hidden_state
    hidden_state = self.pre_layer_norm(hidden_state)
    attn_out = self.attn.inference(hidden_state, freqs_cis)
    hidden_state = residual + attn_out

    residual = hidden_state
    hidden_state = self.post_layer_norm(hidden_state)
    return residual + self.mlp(hidden_state)


class TinyLlamaModel(nn.Module):
  def __init__(self, config: TinyLlamaConfig):
    super().__init__()
    self.cfg = config
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.padding_idx)
    self.layers = nn.ModuleList([TinyLlamaDecoderLayer(config) for _ in range(config.n_layer)])
    self.norm = TinyLlamaRMSNorm(config)
    self.rope = TinyLlamaRotaryEmbedding(config)

  def forward(self, x):
    batch_size, seq_len = x.shape
    x = self.embed_tokens(x)
    position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    freqs_cis = self.rope(position_ids)
    for layer in self.layers:
      x = layer(x, freqs_cis)
    return self.norm(x)

  def prefill(self, x):
    batch_size, seq_len = x.shape
    x = self.embed_tokens(x)
    position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    freqs_cis = self.rope(position_ids)
    for layer in self.layers:
      x = layer.prefill(x, freqs_cis)
    return self.norm(x)

  def decode_step(self, token_id, pos):
    batch_size = token_id.shape[0]
    x = self.embed_tokens(token_id)
    position_ids = torch.tensor([[pos]], device=x.device).expand(batch_size, -1)
    freqs_cis = self.rope(position_ids)
    for layer in self.layers:
      x = layer.decode(x, freqs_cis)
    return self.norm(x)

  def reset_cache(self):
    for layer in self.layers:
      layer.attn.reset_cache()


# for causal next token prediction:
class TinyLlamaForCausalLM(nn.Module):
  def __init__(self, config: TinyLlamaConfig):
    super().__init__()
    self.model = TinyLlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
    self.lm_head.weight = self.model.embed_tokens.weight
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x):
    hidden_states = self.model(x)
    logits = self.lm_head(hidden_states)
    return logits

  @torch.no_grad()
  def generate(self, input_ids, tokenizer, max_new_tokens=128,
               temperature=0.8, top_k=50, top_p=0.9, eos_token_id=None):
    """Streaming generator: yields new text chunks as tokens are decoded."""
    self.eval()
    self.model.reset_cache()
    eos_id = eos_token_id or tokenizer.eos_token_id
    prompt_len = input_ids.shape[1]
    generated_ids = []
    prev_text = ""

    # Prefill: process the entire prompt, populate KV cache
    hidden = self.model.prefill(input_ids)
    logits = self.lm_head(hidden[:, -1, :])
    next_token = _sample(logits, temperature, top_k, top_p)
    generated_ids.append(next_token.item())
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    yield full_text[len(prev_text):]
    prev_text = full_text

    #Decode: one token at a time using KV cache
    for step in range(max_new_tokens - 1):
      pos = prompt_len + step
      if pos >= self.model.cfg.max_position_embedding - 1:
        break
      hidden = self.model.decode_step(next_token, pos)
      logits = self.lm_head(hidden[:, -1, :])
      next_token = _sample(logits, temperature, top_k, top_p)
      if next_token.item() == eos_id:
        break
      generated_ids.append(next_token.item())
      full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
      new_text = full_text[len(prev_text):]
      if new_text:
        yield new_text
        prev_text = full_text

    self.model.reset_cache()


def _sample(logits, temperature=0.8, top_k=50, top_p=0.9):
    if temperature < 1e-8:
      return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
      top_k = min(top_k, logits.size(-1))
      val, _ = torch.topk(logits, top_k)
      logits[logits < val[..., -1:]] = float("-inf")

    if top_p < 1.0:
      sorted_logits, sorted_idx = torch.sort(logits, descending=True)
      cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
      remove = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
      sorted_logits[remove] = float("-inf")
      logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

