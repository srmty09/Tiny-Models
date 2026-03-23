import torch
import torch.nn as nn
from dataclasses import dataclass
import math
import torch.nn.functional as F


@dataclass
class TinyLlamaConfig:
    hidden_dim:int = 768
    n_heads:int = 12
    n_layer:int = 12
    num_experts:int = 8
    top_k:int = 2
    head_dim:int = 64
    batch_size: int = 32
    max_position_embedding: int = 256
    rms_norm_eps:float = 1e-5
    attention_dropout: float = 0.1
    rope_theta:float = 1e4
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

class TinyLlamaRMSNorm(nn.Module):
  def __init__(self,config:TinyLlamaConfig):
    super().__init__()
    self.cfg = config
    self.eps = config.rms_norm_eps
    self.weights = nn.Parameter(torch.ones(config.hidden_dim))

  def forward(self,hidden_state):
    variance = torch.pow(hidden_state,2).mean(-1,keepdim=True)
    hidden_state = hidden_state * torch.rsqrt(variance+self.eps)
    return self.weights*hidden_state


class TinyLlamaExperts(nn.Module):
    def __init__(self,config:TinyLlamaConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_dim
        self.expert_dim = config.intermediate_dim
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts,self.hidden_size,2*self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts,self.expert_dim,self.hidden_size))
        self.act_fn = nn.SiLU()

        nn.init.kaiming_uniform_(self.gate_up_proj.view(self.num_experts, -1))
        nn.init.kaiming_uniform_(self.down_proj.view(self.num_experts, -1))

    def forward(self,hidden_states):
        hidden_states = hidden_states.view(self.num_experts,-1,self.hidden_size)
        gate_up = torch.bmm(hidden_states,self.gate_up_proj)
        gate,up = gate_up.chunk(2,dim=-1)
        gated = up*self.act_fn(gate)
        next_state = torch.bmm(gated,self.down_proj)
        next_state = next_state.view(-1,self.hidden_size)
        return next_state


class TinyLlamaMLP(nn.Module):
    def __init__(self,config:TinyLlamaConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_dim,config.intermediate_dim,bias=False)
        self.up_proj = nn.Linear(config.hidden_dim,config.intermediate_dim,bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim,config.hidden_dim,bias=False)
        self.act_fn = nn.SiLU()

    def forward(self,x):
        gated = self.act_fn(self.gate_proj(x))*self.up_proj(x)
        return self.down_proj(gated)


class TinyLlamaRouter(nn.Linear):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__(config.hidden_dim, config.num_experts, bias=False)
        self.top_k = config.top_k

    def forward(self, hidden_states):
        router_logits = super().forward(hidden_states)
        top_val, top_idx = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, top_idx, top_val)
        router_scores = torch.sigmoid(router_scores.float()).to(router_logits.dtype)
        return router_scores, router_logits


class TinyLlamaMOE(nn.Module):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_dim

        self.router = TinyLlamaRouter(config)
        self.experts = TinyLlamaExperts(config)
        self.shared_expert = TinyLlamaMLP(config)

    def forward(self, hidden_state):
        bd, sl, hd = hidden_state.shape
        hidden_state_flat = hidden_state.view(-1, hd)

        router_scores, router_logits = self.router(hidden_state_flat)

        routed_in = hidden_state_flat.repeat(self.num_experts, 1)
        routed_in = routed_in * router_scores.t().reshape(-1, 1)

        routed_out = self.experts(routed_in)

        shared_out = self.shared_expert(hidden_state_flat)
        out = shared_out + routed_out.view(self.num_experts, -1, hd).sum(dim=0)

        return out.view(bd, sl, hd), router_logits


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
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = freq_cis.unsqueeze(2)
    xq_out = torch.view_as_real(xq_complex * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(hidden_state,n_rep):
    batch_dim, seq_len, kv_heads, kv_head_dim = hidden_state.shape
    hidden_state = hidden_state.unsqueeze(3)
    hidden_state = hidden_state.expand(batch_dim,seq_len,kv_heads,n_rep,kv_head_dim)
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

    def forward(self, hidden_state, freqs_cis, attention_mask=None):
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

        if attention_mask is None:
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
        bd, sl, hd = x.shape
        assert sl == 1

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
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.cfg = config
        self.attn = TinyLlamaGQA(config)
        self.moe = TinyLlamaMOE(config)
        self.pre_layer_norm = TinyLlamaRMSNorm(config)
        self.post_layer_norm = TinyLlamaRMSNorm(config)

    def forward(self, hidden_state, freqs_cis, attention_mask=None):
        residual = hidden_state
        hidden_state = self.pre_layer_norm(hidden_state)
        atten_out, _ = self.attn(hidden_state, freqs_cis, attention_mask)
        hidden_state = residual + atten_out

        residual = hidden_state
        hidden_state = self.post_layer_norm(hidden_state)
        moe_out, router_logits = self.moe(hidden_state)
        return residual + moe_out, router_logits

    def prefill(self, hidden_state, freqs_cis):
        residual = hidden_state
        hidden_state = self.pre_layer_norm(hidden_state)
        attn_out = self.attn.prefill(hidden_state, freqs_cis)
        hidden_state = residual + attn_out

        residual = hidden_state
        hidden_state = self.post_layer_norm(hidden_state)
        moe_out, _ = self.moe(hidden_state)
        return residual + moe_out

    def decode(self, hidden_state, freqs_cis):
        residual = hidden_state
        hidden_state = self.pre_layer_norm(hidden_state)
        attn_out = self.attn.inference(hidden_state, freqs_cis)
        hidden_state = residual + attn_out

        residual = hidden_state
        hidden_state = self.post_layer_norm(hidden_state)
        moe_out, _ = self.moe(hidden_state)
        return residual + moe_out


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
        all_router_logits = []
        for layer in self.layers:
            x, router_logits = layer(x, freqs_cis)
            all_router_logits.append(router_logits)
        return self.norm(x), all_router_logits

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

    def forward(self, x, targets=None):
        hidden_states, all_router_logits = self.model(x)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux_loss = load_balance_loss(all_router_logits, self.model.cfg.num_experts, self.model.cfg.top_k)
            loss = ce_loss + 0.01 * aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=128,
                 temperature=0.8, top_k=50, top_p=0.9, eos_token_id=None):
        self.eval()
        self.model.reset_cache()
        eos_id = eos_token_id or tokenizer.eos_token_id
        prompt_len = input_ids.shape[1]
        generated_ids = []
        prev_text = ""

        hidden = self.model.prefill(input_ids)
        logits = self.lm_head(hidden[:, -1, :])
        next_token = _sample(logits, temperature, top_k, top_p)
        generated_ids.append(next_token.item())
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        yield full_text[len(prev_text):]
        prev_text = full_text

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


def load_balance_loss(all_router_logits, num_experts, top_k):
    total_loss = 0
    for router_logits in all_router_logits:
        router_probs = torch.softmax(router_logits, dim=-1)
        _, top_idx = torch.topk(router_logits, top_k, dim=-1)
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, top_idx, 1)
        tokens_per_expert = expert_mask.mean(0)
        probs_per_expert = router_probs.mean(0)
        total_loss += num_experts * (tokens_per_expert * probs_per_expert).sum()
    return total_loss / len(all_router_logits)


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




torch.random.manual_seed(42)

cfg = TinyLlamaConfig()

model = TinyLlamaForCausalLM(cfg)
print(f"params: {sum(p.numel() for p in model.parameters())//1e6:,}")

x = torch.randint(0, cfg.vocab_size, (2, 16))
targets = torch.randint(0, cfg.vocab_size, (2, 16))

logits, loss = model(x, targets)
print(f"logits: {logits.shape}")
print(f"loss:   {loss.item():.4f}")

logits_nograd, _ = model(x)
print(f"logits no target: {logits_nograd.shape}")