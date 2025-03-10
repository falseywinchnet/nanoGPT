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

class ComplexConditionalVRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        """
        x_dim: dimension of complex input (i.e. 2 * embed_dim)
        h_dim: hidden state dimension
        z_dim: latent variable dimension
        """
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Inference network: q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, 2 * z_dim),
            nn.Tanh(),
            nn.Linear(2 * z_dim, 2 * z_dim)
        )
        # Prior network: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 2 * z_dim),
            nn.Tanh(),
            nn.Linear(2 * z_dim, 2 * z_dim)
        )
        # Decoder network: decodes x_t from [z_t, h_{t-1}]
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + h_dim, 2 * x_dim),
            nn.Tanh(),
            nn.Linear(2 * x_dim, x_dim)
        )
        # RNN update: updates hidden state from [x_t, z_t]
        self.rnn = nn.GRUCell(x_dim + z_dim, h_dim)
        self.activation = ReferenceActivation()

    def forward(self, x, h_prev):
        """
        Args:
          x: (B, x_dim) — current complex input (real and imag concatenated)
          h_prev: (B, h_dim) — previous hidden state
        Returns:
          x_hat: predicted x (B, x_dim)
          h_new: updated hidden state (B, h_dim)
          z: sampled latent (B, z_dim)
          kl: approximate KL divergence term (B,)
        """
        # Inference: condition on x and h_prev
        enc_input = torch.cat([x, h_prev], dim=-1)
        enc_out = self.enc_net(enc_input)
        mu_enc, log_sigma_enc = enc_out.chunk(2, dim=-1)
        sigma_enc = torch.exp(log_sigma_enc)

        # Sample latent from q(z|x,h)
        eps = torch.randn_like(sigma_enc)
        z = mu_enc + sigma_enc * eps

        # Prior from h_prev
        prior_out = self.prior_net(h_prev)
        mu_prior, log_sigma_prior = prior_out.chunk(2, dim=-1)
        sigma_prior = torch.exp(log_sigma_prior)

        # Decode: predict x_hat
        dec_input = torch.cat([z, h_prev], dim=-1)
        x_hat = self.dec_net(dec_input)

        # Update hidden state
        rnn_input = torch.cat([x, z], dim=-1)
        h_new = self.rnn(rnn_input, h_prev)

        # Compute KL divergence (per data point, ignoring constants)
        kl = 0.5 * torch.sum(sigma_enc**2 + (mu_enc - mu_prior)**2 - 1 - 2 * log_sigma_enc + 2 * log_sigma_prior, dim=-1)
        return x_hat, h_new, z, kl

# ---------------------------------------------------------------------------
# Ingestion phase: process the observed sequence with the conditional VRNN.
def ingest_sequence(cell, x_complex):
    """
    x_complex: (B, T, x_dim) — the input sequence (complex)
    Returns:
       x_hat_seq: (B, T, x_dim) — sequence of decoded predictions (for reconstruction)
       h_final: (B, h_dim) — final hidden state after ingestion
       kl_seq: (B, T) — per-timestep KL divergence terms
    """
    B, T, _ = x_complex.shape
    h = torch.zeros(B, cell.h_dim, device=x_complex.device)
    x_hat_list = []
    kl_list = []
    for t in range(T):
        x_t = x_complex[:, t, :]  # (B, x_dim)
        x_hat, h, z, kl = cell(x_t, h)
        x_hat_list.append(x_hat)
        kl_list.append(kl)
    x_hat_seq = torch.stack(x_hat_list, dim=1)
    kl_seq = torch.stack(kl_list, dim=1)
    return x_hat_seq, h, kl_seq

def auto_regressive_predict(cell, h_init, steps, top_k=5):
    """
    h_init: (B, h_dim) — final hidden state from ingestion
    steps: number of future timesteps to predict
    top_k: branching factor
    
    Returns:
       pred_seq: (B, steps, x_dim) — predicted extension (complex), 
                 obtained by averaging latent z's and then decoding.
                 
    This version collects latent z values (from q(z|x,h)) at each step,
    averages them across the top-k branches, and then uses the decoder (dec_net)
    with a placeholder hidden (e.g. zeros) to produce the final predicted x.
    """
    B = h_init.size(0)
    # Use a zero vector as a starting placeholder input.
    x_start = torch.zeros(B, cell.x_dim, device=h_init.device)
    from collections import deque
    queue = deque()
    # Each element is a tuple:
    # (list of latent z's for each predicted step, current hidden state, cumulative KL, last input used)
    queue.append(([], h_init, 0.0, x_start))
    
    for _ in range(steps):
        next_queue = []
        while queue:
            latent_seq_so_far, h_prev, kl_so_far, last_x = queue.popleft()
            # Run one step of the cell.
            # Note: cell returns (x_hat, h_new, z, kl) where:
            #  - z is the latent variable sampled from q(z|x,h)
            x_hat, h_new, z, kl = cell(last_x, h_prev)
            new_latent_seq = latent_seq_so_far + [z]  # Append latent for this timestep.
            new_kl = kl_so_far + kl.item()  # Accumulate KL divergence (convert to scalar if needed)
            # For the next step, we use the decoded x_hat as input.
            next_queue.append((new_latent_seq, h_new, new_kl, x_hat))
        # Keep only the top_k branches (lowest cumulative KL).
        next_queue = sorted(next_queue, key=lambda tup: tup[2])[:top_k]
        queue = deque(next_queue)
    
    # Now, each branch has a sequence of latent z's, one per predicted timestep.
    # Each latent is of shape (B, z_dim); stacking along time gives (B, steps, z_dim).
    all_latent_seqs = [torch.stack(latent_seq, dim=1) for latent_seq, h, kl, x in queue]
    # Average the latent sequences across branches.
    avg_latent_seq = torch.stack(all_latent_seqs, dim=0).mean(dim=0)  # (B, steps, z_dim)
    
    # Now decode each averaged latent z into x using the decoder network.
    # We need to provide some hidden context; here we use a placeholder (e.g. zeros).
    B, L, z_dim = avg_latent_seq.shape
    h_placeholder = torch.zeros(B, cell.h_dim, device=avg_latent_seq.device)
    decoded_steps = []
    for t in range(L):
        z_t = avg_latent_seq[:, t, :]  # (B, z_dim)
        dec_in = torch.cat([z_t, h_placeholder], dim=-1)  # (B, z_dim + h_dim)
        x_t = cell.dec_net(dec_in)  # (B, x_dim)
        decoded_steps.append(x_t)
    pred_seq = torch.stack(decoded_steps, dim=1)  # (B, steps, x_dim)
    return pred_seq



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
        self.cond_vrnn = ComplexConditionalVRNNCell(
            x_dim = 2 * config.n_embd,  # complex input dimension
            h_dim = 64,
            z_dim = 16
        )
        self.top_k = 5
        self.steps = 5

        # ---- Primary Q,K,V ----
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # ---- Secondary Q,K,V if use_secondary_embed is enabled ----
        self.c_attn_2 = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
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

    def forward(self, x, x2=None, rope_freqs=None):
        """
        x:  (B, T, C) primary embeddings
        x2: (B, T, C) secondary embeddings (if any)
        """
        B, T, C = x.size()
        x_complex = torch.cat([x, x2], dim=-1)  # (B, T, 2C)

        h = torch.zeros(B, self.cond_vrnn.h_dim, device=x.device)
        for t in range(T):
            x_t = x_complex[:, t, :]  # (B, 2C)
            x_hat, h, z, kl = self.cond_vrnn(x_t, h)
            
        pred_complex = auto_regressive_predict(self.cond_vrnn, h, steps=self.steps, top_k=self.top_k)

        pred_real, pred_imag = pred_complex.split(C, dim=-1)  # each: (B, steps, C)
        
        # 4. Append the predicted extensions to the original x and x2.
        x_aug = torch.cat([x, pred_real], dim=1)   # (B, T+steps, C)
        x2_aug = torch.cat([x2, pred_imag], dim=1)  # (B, T+steps, C)
        T_aug = x_aug.size(1)

        # ---- Primary pass Q,K,V ----
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)

        noise = torch.randn_like(q) * self.noise_alpha
        v_pos = v + noise
        v_neg = v - noise

        if self.use_rope and rope_freqs is not None:
            self.rope_freqs = rope_freqs
            q, k = self.apply_rope(q, k)

        # Compute attention for the primary embedding
        if self.flash:
            att_pos = F.scaled_dot_product_attention(q, k, v_pos, attn_mask=None,
                        dropout_p=self.dropout if self.training else 0, is_causal=True)
            att_neg = F.scaled_dot_product_attention(q, k, v_neg, attn_mask=None,
                        dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att_probs = F.softmax(att_scores, dim=-1)
            att_probs = self.attn_dropout(att_probs)
            att_pos = att_probs @ v_pos
            att_neg = att_probs @ v_neg
        
        y_primary = (att_pos + att_neg) / 2  # (B, n_head, T, head_dim)

        # ---- Secondary pass if x2 is given ----
        # c_attn_2 -> Q2,K2,V2
        q2, k2, v2 = self.c_attn_2(x2).split(self.n_embd, dim=2)
        q2 = q2.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)
        k2 = k2.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)
        v2 = v2.view(B, T_aug, self.n_head, C // self.n_head).transpose(1, 2)
        if self.use_rope:
            q2, k3 = self.apply_rope(q2, k2)


        if self.flash:
            att2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None,
                       dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att_scores2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(k2.size(-1)))
            att_scores2 = att_scores2.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att_probs2 = F.softmax(att_scores2, dim=-1)
            att_probs2 = self.attn_dropout(att_probs2)
            att2 = att_probs2 @ v2
            
            # Combine primary + secondary stream
            y_secondary = att2  # shape (B, n_head, T, head_dim)
            y = (y_primary + y_secondary) / 2
        else:
            # No x2 => just use primary
            y = y_primary

        # Reshape
        y = y.transpose(1, 2).contiguous().view(B, T_aug, C)
        y = y[:,:T,:]#truncate
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
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, rope_freqs):
        x = x + self.attn(self.ln_1(x), rope_freqs)
        x = x + self.mlp(self.ln_2(x))       
        return x


class ResBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_r = LayerNorm(config.n_embd, bias=config.bias)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionPair(config)
        self.mlp = MLP(config)
    
    def forward(self, x,x2, rope_freqs, num_iterations=3):
        """
        Iteratively apply attention and MLP with residuals over multiple steps.
        """
        for _ in range(num_iterations):  # Number of recurrence steps
            residual = x  # Capture the original input as the residual

            # Apply normalization and attention (using the residual)
            x = self.ln_1(x)
            x2 = self.ln_r(x2)
            x = self.attn(x,x2, rope_freqs)

            # Apply normalization and MLP (using the residual)
            x = self.ln_2(x)
            x = self.mlp(x)

            # Add the residual after each iteration to refine the result
            x = x + residual

        return x
    
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
        self.ln_x = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_y = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_R = LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            prelude = nn.ModuleList([Block(config) for _ in range(1)]),
            preludey = nn.ModuleList([Block(config) for _ in range(1)]),
            coda = nn.ModuleList([Block(config) for _ in range(1)]),
            residual = nn.ModuleList([ResBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: share the weights between the token embedding and the final output layer.
        self.transformer.wte.weight = self.lm_head.weight
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
            n_params -= self.transformer.wpe.weight.numel()
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


    def compute_phase_embedding(self, tok_emb, pos_emb):
            """
            Computes a secondary embedding for each token by accumulating phase shifts.
            
            For each token position i in x (of shape (B, T, C)):
              - We have two positional embeddings:
                   * A forward rope embedding for tokens j > i.
                   * A backward rope embedding for tokens j < i.
              - For each token j (j != i), we compute a positional indication:
                   If j > i: use forward embedding (function of j - i)
                   If j < i: use backward embedding (function of i - j)
              - Multiply that positional indication (elementwise) by x[j] to get a phase shift contribution.
              - Sum over all j to obtain a cumulative phase shift, φ[i].
              - Form a complex number with:
                   real part = x[i]
                   imaginary part = φ[i]
                and take the magnitude:
                   x2[i] = sqrt( x[i]^2 + φ[i]^2 )
            
            Args:
                tok_emb: Tensor of shape (B, T, C) representing token embeddings.
                pos_emb: Tensor of shape (T,C) with positions (0, 1, ..., T-1).
            
            Returns:
                x2: Tensor of shape (B, T, C), the secondary embeddings.
            """
            B, T, C = tok_emb.shape
            device = tok_emb.device
        
            # Define fixed frequencies per dimension (like standard RoPE frequencies).
            dims = torch.arange(C, device=device, dtype=torch.float32)  # (C,)
            freqs = 1.0 / (10000 ** (dims / C))  # (C,)
        
            # Compute a matrix of relative offsets: offset[i, j] = j - i.
            indices = torch.arange(T, device=device)
            offset_matrix = indices.unsqueeze(0) - indices.unsqueeze(1)  # shape (T, T)
        
            # Create masks for forward and backward directions.
            forward_mask = (offset_matrix > 0).float()  # (T, T)
            backward_mask = (offset_matrix < 0).float()  # (T, T)
        
            # For each pair (i, j) compute the absolute offset:
            # For forward, we care about (j - i) and for backward, (i - j).
            forward_offsets = torch.clamp(offset_matrix, min=0)  # (T, T): zeros for j <= i.
            backward_offsets = torch.clamp(-offset_matrix, min=0)  # (T, T): zeros for j >= i.
        
            # Compute forward and backward positional rope embeddings.
            # Here we use a simple sinusoidal function.
            # Each will have shape (T, T, C): for each i and j, a vector of length C.
            forward_pos_emb = torch.sin(forward_offsets.unsqueeze(-1) * freqs.view(1, 1, C))
            backward_pos_emb = torch.sin(backward_offsets.unsqueeze(-1) * freqs.view(1, 1, C))
            
            # Combine them: for each pair (i,j), use the forward embedding if j>i, backward if j<i.
            phase_emb = forward_mask.unsqueeze(-1) * forward_pos_emb + backward_mask.unsqueeze(-1) * backward_pos_emb
            # For j == i, offset is 0 so sin(0)=0; effectively these contribute nothing.
        
            # Now, for each token position i (for each batch element), accumulate contributions from all j:
            # φ[i, c] = sum_{j != i} [ phase_emb[i,j,c] * x[b,j,c] ]
            # Using einsum: treat phase_emb as (T, T, C) and x as (B, T, C) so that we sum over j.
            phi = torch.einsum('ijc,bjc->bic', phase_emb, tok_emb)
            phi = phi / (T ** 0.5)
            return phi




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
        x = self._run_transformer(idx)  # shape (b, t, n_embd)
        
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
            return logits, loss

    # ------------------------------------------------------------------
    # Internal helper that runs the transformer stacks on a given 
    # index sequence. Optionally override the last token's embedding.
    # Also includes "secondary embeddings" if config.use_secondary_embed.
    # ------------------------------------------------------------------
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
        pos = torch.arange(t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        phase = self.compute_phase_embedding(tok_emb, pos_emb) 

        x = tok_emb + pos_emb.unsqueeze(0) 
        y = tok_emb + phase

        
        dropout_mask = (torch.rand_like(x) > self.config.dropout).float() / (1.0 - self.config.dropout)
        
        # Step 2: Apply dropout mask to token embeddings (x) and later to phase embeddings
        x = x * dropout_mask  
        y = y * dropout_mask              

        for i, block in enumerate(self.transformer.prelude):
            residual = block(x, rope_freqs=self.rope_freqs)  
       
        for i, block in enumerate(self.transformer.preludey):
            residualy = block(y, rope_freqs=self.rope_freqs)  

        # Pass through each block
        for i, block in enumerate(self.transformer.residual):
            x = block(x+residual, residualy, rope_freqs=self.rope_freqs)
            residualy = residualy + self.ln_R(x)

                    
        for i, block in enumerate(self.transformer.coda):
            x = block(x, rope_freqs=self.rope_freqs)       

        # Final layernorm
        x = self.transformer.ln_f(x)
        return x
    
        
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
