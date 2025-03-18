#copyright joshuah.rainstar@gmail.com 2025
#may not be used without some form of compensation
#PLEASE SEND ME MONEY

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        self.use_rope = config.use_rope
        self.noise_alpha = config.noise_alpha
        

        # ---- Primary Q,K,V ----
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.abs_pos_embedding = nn.Parameter(torch.zeros(1, config.n_head, config.block_size, config.block_size))

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def apply_rope(self, q, k):
        """Applies RoPE to queries and keys."""
        B, H, S, D = q.shape
        rope_freqs = self.rope_freqs.to(q.device, q.dtype)
        positions = torch.arange(S, device=q.device, dtype=q.dtype).unsqueeze(1)
        theta = positions * rope_freqs.unsqueeze(0)  # shape [S, D/2]
        sin_theta, cos_theta = theta.sin(), theta.cos()

        sin_theta = sin_theta.expand(B, H, S, D // 2)
        cos_theta = cos_theta.expand(B, H, S, D // 2)

        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        q = torch.cat([q1*cos_theta - q2*sin_theta, q1*sin_theta + q2*cos_theta], dim=-1)
        k = torch.cat([k1*cos_theta - k2*sin_theta, k1*sin_theta + k2*cos_theta], dim=-1)
        return q, k
        

    def forward(self, x, rope_freqs=None, weights=None):


        B, T, C = x.size()
        weight = weights if weights is not None else self.c_attn.weight
        q, k, v = F.linear(x, weight).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_rope and rope_freqs is not None:
            self.rope_freqs = rope_freqs
            q, k = self.apply_rope(q, k)

                
        # Compute attention for the primary embedding
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                        dropout_p=self.dropout if self.training else 0, is_causal=True)
                
        else:
            print("Nope not today!")
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            position_scores = self.compute_positional_scores(x.shape[1]).to(x.device)  # (T, T) fixed position bias
            att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            att_probs = F.softmax(att_scores, dim=-1)
            att_probs = self.attn_dropout(att_probs)
            att_pos = att_probs @ v
            y = att_pos 

        # Reshape
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class RainstarActivation(nn.Module):
    def __init__(self, gammaval=8):
        super().__init__()
        self.sil = nn.SiLU()
    def forward(self, x):
       neg =  self.sil(x) * (x*torch.sigmoid(x)) + x/(1+torch.abs(x))
       pos =  x -  x/(1+torch.abs(x))
       return (neg *torch.sigmoid(-x)) + (pos * torch.sigmoid(x)) +1

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = RainstarActivation()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False) #KAN style
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_r = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, rope_freqs,c):
            """
            Iteratively apply attention and MLP with residuals over multiple steps.
            """
            prior = x
            x = self.ln_1(x)
            x = self.attn(x, rope_freqs,weights=c)
            # --- Step 2: Compute Absolute Positional Bias ---
            # Apply normalization and MLP (using the residual)
            x = self.ln_2(x)
            x = self.mlp(x)
            posterior = x + prior
            return posterior
    
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    return_features: bool = False  # New flag to return hidden representations
    use_rope: bool = True
    noise_alpha: float = 0.5

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.attentions = nn.ModuleList([CausalSelfAttention(config) for _ in range(config.n_layer)])
        self.ln_attn = nn.ModuleList([LayerNorm(config.n_embd, bias=config.bias) for _ in range(config.n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for _ in range(config.n_layer)])
        self.ln_mlp = LayerNorm(config.n_embd, bias=config.bias)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight  # Tie embeddings

        if config.use_rope:
            head_dim = config.n_embd // config.n_head
            self.register_buffer("rope_freqs", self._build_rope_frequencies(head_dim))
        else:
            self.rope_freqs = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def _build_rope_frequencies(self, head_dim):
        # e.g. head_dim = 32
        half_dim = head_dim // 2  # =16
        return 1.0 / (
            10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.wte(idx) + self.wpe(pos)

        residual = x  # Keep initial residual state

        # Store intermediate attention products
        attention_outputs = []
        #
         #   if i > 0 and (i + 1) % 2 and i < len(self.transformer.residual) - 1:
         #       weight_prev = self.transformer.residual[i - 1].attn.c_attn.weight
         #       weight_next = self.transformer.residual[i + 1].attn.c_attn.weight
         #       c = weight_prev + weight_next
        #        c = c/2
          #  else:
        #        c = None
        # ---- Attention Stage ----
        for i, attn, norm,mlp in zip(range(config.n_layer),self.attentions, self.ln_attn,self.mlps):
            
            x = attn(x,rope_freqs=self.rope_freqs,weights=None)
            x = norm(x)
            if i == 0:
                residual = residual + x #seed the stage
            else:
                residual += mlp(x)

        # Final norm and output
        x = self.ln_mlp(residual)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
        
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True       # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens): 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
