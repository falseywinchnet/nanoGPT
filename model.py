#copyright joshuah.rainstar@gmail.com 2025
#may not be used without some form of compensation
#PLEASE SEND ME MONEY

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DyT(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*0.5)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class NewAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.n_embd
        self.n_heads = config.n_head
        self.dropout = nn.Dropout(config.dropout)
        
        # DyT normalization (replaces LayerNorm)
        self.dyt = DyT(self.emb_dim)
        self.dyt2 = DyT(self.emb_dim)
        self.dyt3 = DyT(self.emb_dim)

        # Positional embedding unique to this block
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        
        self.W_q = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_k = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_v = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        
        # Attention projection (applied after attention)
        self.attn_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        
        # Feedforward layers
        self.expand = nn.Linear(self.emb_dim, 4 * self.emb_dim)
        self.contract = nn.Linear(4 * self.emb_dim, self.emb_dim)
        self.gelu = nn.GELU()

    
    def compute_positional_scores(self, seq_len):
        """Creates a position-dependent bias matrix (T, T)"""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # (1, T)
        relative_position = position - position.transpose(0, 1)  # (T, T)
        
        # Non-learnable position interaction (like in DIET)
        position_scores = -torch.abs(relative_position)  # Closer positions get higher scores
        return position_scores.to(self.pos_embed.device)
        
    def forward(self, x):
        x = x + self.pos_embed  # (B, T, C) #reinforce embeddings

        prior = self.dyt3(x.clone())  # Shape: (B, T, C)

        # Apply dropout (keeps shape unchanged)
        x = self.dropout(x)  # (B, T, C)
        
        # Add block-specific positional embedding (broadcasts over batch and tokens)
        
        # Apply Dynamic Tanh normalization (DyT) → still (B, T, C)
        x = self.dyt(x)
        
        # --- Whitening and Ordinary Attention ---
        # Apply the whitening transform to decorrelate features.
        # This aims to approximate the effect of a Mahalanobis measure.
        
        # Compute queries, keys, values.
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)  # (B, T, C)
        V = self.W_v(x)  # (B, T, C)
        
        # Compute standard scaled dot-product attention
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.emb_dim)  # (B, T, T)
        
        # Compute explicit position-dependent interaction (DIET's approach)
        position_scores = self.compute_positional_scores(x.shape[1])  # (T, T) fixed position bias
        
        # Combine both for final attention scores
        attn_scores = content_scores + position_scores  # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        
        # Multiply attention weights by V to get attended output.
        # The resulting shape remains (B, T, C)
        attn_output = torch.matmul(attn_weights, V)
        
        # Project the attention output back into the embedding space.
        attn_output = self.attn_projection(attn_output)  # (B, T, C)
        
        # Apply second DyT scaling.
        x = self.dyt2(attn_output)  # (B, T, C)
        
        # --- Feedforward Transformation ---
        # Expand → GELU → Contract: (B, T, C) -> (B, T, 4C) -> (B, T, 4C) -> (B, T, C)
        x = self.expand(x)
        x = self.gelu(x)
        x = self.contract(x)
                
        # Residual addition: add the original input back.
        posterior = x + prior  # (B, T, C)
        
        # --- Compute Divergence Loss (JS loss) ---
        # We compute a symmetric KL divergence between the softmax-normalized prior and posterior.
        m = 0.5 * (prior + posterior)
        js_loss = 0.5 * (F.kl_div(prior.log_softmax(dim=-1), m.softmax(dim=-1), reduction='batchmean') +
                         F.kl_div(posterior.log_softmax(dim=-1), m.softmax(dim=-1), reduction='batchmean'))
        
        return posterior, js_loss

    
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
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            residual = nn.ModuleList([NewAttentionBlock(config) for _ in range(config.n_layer)])
        ))
        

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: share the weights between the token embedding and the final output layer.
        self.transformer.wte.weight = self.lm_head.weight

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
        return n_params
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(
        self,
        idx,
        targets=None,
        return_features=None
    ):
        """
        A single forward method that:
          - If refinement_steps=0, just does a normal forward pass => last-position logits
          - If refinement_steps>0, does multiple passes:
              (a) get full-sequence logits,
              (b) apply Gumbel-Softmax on last token => "virtual token",
              (c) append it,
              (d) re-run attention,
              repeated 'refinement_steps' times,
              then slice the final output back to the original length,
              and optionally compute CE loss.
          - If return_features=True, we return hidden states instead of logits.

        Param:
          idx: (b, t) integer token IDs
          targets: (b, t) or None
          refinement_steps: how many "dance" passes to do
          gumbel_tau: gumbel temperature
          hard_st: if True, use straight-through gumbel
          return_features: override config.return_features if set
        """
        if return_features is None:
            return_features = self.config.return_features

        device = idx.device
        b, original_t = idx.shape

        # ---- Normal single forward pass for the given sequence ----
        x, kl_loss = self._run_transformer(idx)  # shape (b, t, n_embd)
        
        if return_features:
            # If we want the final hidden states:
            return x  # shape (b, t, n_embd)
        else:
            # otherwise, return last-position logits by default:
            logits = self.lm_head(x[:, -1:, :])  # (b, 1, vocab_size)
            loss = None
            if targets is not None:
                # compute CE across entire seq
                full_logits = self.lm_head(x)      # (b, t, vocab_size)
                loss = F.cross_entropy(
                    full_logits.view(-1, full_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
                loss = loss + kl_loss * 0.1
            return logits, loss


    def _run_transformer(self, idx):
        """
        1) Convert idx -> embeddings
        2) If override_last_embed is not None, replace the last token's embedding
        3) (Optional) build secondary embeddings
        4) Pass through blocks, return final hidden states (b, t, n_embd)
        """
        b, t = idx.shape
        device = idx.device
        # Standard token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        x = tok_emb 
                 
        total_kl_loss = 0.0
        # Pass through each block
        for i, block in enumerate(self.transformer.residual):
            x, kl_loss = block(x)   
            total_kl_loss += kl_loss
        # Final layernorm
        return x, total_kl_loss
    
        
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
