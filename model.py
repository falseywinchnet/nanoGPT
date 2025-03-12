#copyright joshuah.rainstar@gmail.com 2025
#may not be used without some form of compensation
#PLEASE SEND ME MONEY

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F



def compute_phase_embedding(tok_emb):
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


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)




class CustomVRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        """
        x_dim: Input embedding dimension (same as Transformer hidden size)
        h_dim: Hidden state size
        z_dim: Latent variable dimension
        """
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Encoder: Produces latent representation z from input and previous state
        # Inference network: q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, 2 * z_dim),
            nn.Tanh(),
            nn.Linear(2 * z_dim, z_dim)
        )
        
        # Decoder: Reconstructs input x from latent z and hidden state
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + h_dim, 2 * x_dim),
            nn.Tanh(),
            nn.Linear(2 * x_dim, x_dim)
        )

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 2 * z_dim),
            nn.Tanh(),
            nn.Linear(2 * z_dim, 2 * z_dim)
        )
        # Custom state transition (instead of GRU)
        self.trans_net = nn.GRUCell(x_dim + z_dim, h_dim)

        # Activation function
        self.activation = nn.GELU()

    def forward(self, x, h_prev):
        """
        Args:
            x: (B, x_dim) – Current input embedding.
            h_prev: (B, h_dim) – Previous hidden state.

        Returns:
            x_hat: (B, x_dim) – Reconstructed x.
            h_new: (B, h_dim) – Updated hidden state.
            z: (B, z_dim) – Sampled latent representation.
        """
        # Encode input and previous state into latent z
        enc_input = torch.cat([x, h_prev], dim=-1)
        z = self.enc_net(enc_input)

        # Prior from h_prev
        prior_out = self.prior_net(h_prev)
        mu_prior, log_sigma_prior = prior_out.chunk(2, dim=-1)
        sigma_prior = torch.exp(log_sigma_prior)

        # Decode latent z and hidden state to reconstruct x_hat
        dec_input = torch.cat([z, h_prev], dim=-1)
        x_hat = self.dec_net(dec_input)

        # Custom state transition (Instead of GRU)
        trans_input = torch.cat([x, z], dim=-1)
        h_new = self.activation(self.trans_net(trans_input))

        return x_hat, h_new, z


class CustomVRNN(nn.Module):
    def __init__(self, seq_len, x_dim, h_dim, z_dim, num_layers=2):
        """
        Variational RNN model that operates as a predictive layer before attention.
        
        Args:
            seq_len: Maximum sequence length.
            x_dim: Transformer hidden size (input dimension).
            h_dim: Hidden state size.
            z_dim: Latent variable size.
            num_layers: Number of stacked VRNN layers.
        """
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        # Stacked VRNN cells
        self.vrnn_cells = nn.ModuleList([
            CustomVRNNCell(x_dim, h_dim, z_dim) 
            for i in range(num_layers)
        ])

    def forward(self, x, h_prev=None):
        """
        Processes a batch of input embeddings through VRNN layers.
        
        Args:
            x: (B, T, x_dim) – Input sequence embeddings.
            h_prev: (List of tensors) – Previous hidden states for each layer.

        Returns:
            x_hat: (B, T, x_dim) – Reconstructed input.
            h_new: List[(B, h_dim)] – Updated hidden states per layer.
            z_seq: (B, T, z_dim) – Latent sequence.
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states if not provided
        if h_prev is None or h_prev[0].size(0) != batch_size:
                h_prev = [torch.zeros(batch_size, self.h_dim, device=x.device) for _ in range(self.num_layers)]

        x_hat_seq, new_h_states, z_seq = [], [], []

        for t in range(seq_len):  # Iterate through sequence
            x_t = x[:, t, :]  # Get the t-th timestep embedding

            new_h = []

            for layer_idx, cell in enumerate(self.vrnn_cells):
            
                x_t, h_t, z_t = cell(x_t, h_prev[layer_idx])
                new_h.append(h_t)  # Update hidden state

            h_prev = new_h  # Store updated hidden states
            x_hat_seq.append(x_t)  # Store reconstructed x
            z_seq.append(z_t)  # Store latent z

        # Stack sequences back into (B, T, *)
        x_hat_seq = torch.stack(x_hat_seq, dim=1)
        z_seq = torch.stack(z_seq, dim=1)

        return x_hat_seq, new_h, z_seq[:, -1, :]  # Only take the last step’s latent and z.



from collections import deque
import torch
import torch.nn.functional as F

def auto_regressive_predict_q(last_q, vrnn, h_init, steps, top_k=5, n_candidates=10, sigma=0.1):
    """
    Auto-regressive prediction in q-vector (transformer embedding) space.
    
    The VRNN decodes to a q-vector (the predicted embedding), and we assume that
    this is the mean of a Gaussian distribution. At each step we sample candidate
    q-vectors around that mean, compute their Gaussian log-probabilities (up to a constant),
    select the top_k most probable candidates, and recursively feed them back into the model.
    
    Args:
        last_q: Tensor of shape (B, x_dim) – last observed q-vector.
        vrnn: A multi-layer VRNN model. Its attribute `vrnn_cells` is a list of cells.
              Each cell takes an input q-vector and a hidden state and returns a tuple:
                (x_hat, new_hidden, latent r-vector).
              Here, x_hat is a q-vector prediction.
        h_init: List of hidden state tensors (one per VRNN cell), each of shape (B, h_dim).
        steps: Number of future time steps to predict.
        top_k: Beam search branching factor (number of candidates to keep per branch per step).
        n_candidates: Number of candidate samples to draw (per branch) before pruning.
        sigma: Standard deviation used for sampling candidate q-vectors from the Gaussian
               centered at the predicted q-vector.
    
    Returns:
        pred_seq_tensor: Tensor of shape (B, steps, x_dim) – the predicted q-vector sequence
                         from the best branch.
        best_prob_history: List (length = steps) of tuples with candidate probability data at each step.
                           Each tuple is (candidate_log_probs, candidate_indices) for that step.
                           (This records the probability matrix before pruning at each step.)
    """
    B, x_dim = last_q.shape
    device = last_q.device
    
    # Each beam branch is a tuple:
    #   (pred_seq, h_list, cum_logp, last_q, prob_history)
    # where:
    #   pred_seq: list of predicted q-vectors (each: (B, x_dim))
    #   h_list: list of hidden state tensors (one per VRNN layer)
    #   cum_logp: cumulative log probability (Tensor of shape (B,))
    #   last_q: current q-vector input for this branch (B, x_dim)
    #   prob_history: list of candidate probability data for each step
    beams = [( [], h_init, torch.zeros(B, device=device), last_q, [] )]
    
    for step in range(steps):
        new_beams = []
        # Expand every branch in the beam.
        for branch in beams:
            pred_seq, h_list, cum_logp, last_q, prob_history = branch
            
            # Run one forward step through all VRNN layers.
            x_in = last_q
            new_h_list = []
            for cell, h in zip(vrnn.vrnn_cells, h_list):
                # Each cell returns (x_hat, new_hidden, r-vector); we only need x_hat and new_hidden.
                x_out, h_new, _ = cell(x_in, h)
                new_h_list.append(h_new)
                x_in = x_out
            # x_in is the VRNN's decoded prediction (a q-vector), which we treat as the mean.
            q_hat = x_in  # shape: (B, x_dim)
            
            # Sample n_candidates candidate q-vectors around q_hat.
            # candidate_q shape: (B, n_candidates, x_dim)
            candidate_q = q_hat.unsqueeze(1) + sigma * torch.randn(B, n_candidates, x_dim, device=device)
            
            # Compute (unnormalized) log probability for each candidate under N(q_hat, sigma^2 I).
            # (Ignoring constant terms, the log probability is proportional to -||candidate - q_hat||^2.)
            candidate_log_prob = -0.5 * torch.sum(((candidate_q - q_hat.unsqueeze(1)) / sigma)**2, dim=-1)  # (B, n_candidates)
            
            # Select the top_k candidates for each branch.
            topk_vals, topk_idx = candidate_log_prob.topk(k=top_k, dim=1)  # each: (B, top_k)
            # Gather the corresponding candidate q-vectors.
            # Expand topk_idx to match candidate_q's last dimension.
            candidate_q_topk = torch.gather(candidate_q, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x_dim))  # (B, top_k, x_dim)
            
            # For each candidate in the top_k, create a new branch.
            for i in range(top_k):
                # Extract the candidate q-vector and its log probability.
                candidate = candidate_q_topk[:, i, :]          # (B, x_dim)
                candidate_logp = topk_vals[:, i]                 # (B,)
                
                new_pred_seq = pred_seq + [candidate]            # Append candidate to branch's sequence.
                new_cum_logp = cum_logp + candidate_logp         # Update cumulative log probability.
                # Append the candidate probability distribution data for this step.
                new_prob_history = prob_history + [(topk_vals, topk_idx)]
                
                # Create new branch with updated hidden states and candidate as new last_q.
                new_branch = (new_pred_seq, new_h_list, new_cum_logp, candidate, new_prob_history)
                new_beams.append(new_branch)
        
        # Prune: sort all new branches by the average cumulative log probability and keep the top_k overall.
        new_beams = sorted(new_beams, key=lambda b: b[2].mean().item(), reverse=True)[:top_k]
        beams = new_beams
    
    # After all steps, select the best branch.
    best_branch = beams[0]
    best_pred_seq, best_h, best_cum_logp, best_last_q, best_prob_history = best_branch
    
    # Stack the predicted q-vectors into a tensor of shape (B, steps, x_dim).
    pred_seq_tensor = torch.stack(best_pred_seq, dim=1)
    
    return pred_seq_tensor


            

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

    def forward(self, x, rope_freqs=None):
        """
        x:  (B, T, C) primary embeddings
        x2: (B, T, C) secondary embeddings (if any)
        """
        B, T, C = x.size()
        #x2 = compute_phase_embedding(x)
        #x = x + x2/T

        
        # 4. Append the predicted extensions to the original x and x2.
        

        # ---- Primary pass Q,K,V ----
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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
        
        y = (att_pos + att_neg) / 2  # (B, n_head, T, head_dim)

        
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
    
    def forward(self, x, rope_freqs, num_iterations=3):
            """
            Iteratively apply attention and MLP with residuals over multiple steps.
            """
            residual = x
            x = self.ln_1(x)
            x = self.attn(x, rope_freqs)
            
            # Apply normalization and MLP (using the residual)
            x = self.ln_2(x)
            x = self.mlp(x)
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
            wph = nn.Linear(config.n_embd, config.n_embd),
            drop = nn.Dropout(config.dropout),
            residual = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.cond_vrnn = CustomVRNN(
                seq_len=25,
                x_dim=config.n_embd,
                h_dim=8,
                z_dim=config.n_embd,
                num_layers=2
        )
        self.h_safe= [torch.zeros(1, self.cond_vrnn.h_dim) for _ in range(self.cond_vrnn.num_layers)]

        self.top_k = 5
        self.steps = 50
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
        x,v_loss = self._run_transformer(idx)  # shape (b, t, n_embd)
        
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
                lambda_vrnn = 0.1  # Hyperparameter to weigh the VRNN loss.
                loss = loss + lambda_vrnn * v_loss
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
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Standard token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        phase = compute_phase_embedding(tok_emb)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        phase_emb = self.transformer.wph(phase) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb #+ phase_emb
        
        if torch.is_grad_enabled():
           h = None #reset anew
           x_hat, h, z = self.cond_vrnn(x[:, max(0,t-25):, :], h)  # Pass batch through VRNN

        else:
            h = [h.to(x.device) for h in self.h_safe]  # Ensure h_safe is on the same device as x 
            x_hat, h, z = self.cond_vrnn(x[:, max(0,t-25):, :] , h)  # Pass last items through VRNN because VRNN PAPER SAYS SO
            
        with torch.no_grad():
                # Start with the last 25 tokens as context
                context = x  # (B, 25, embedding_dim)
                h_new = h  # Maintain hidden state across iterations
            
            
                # Sequentially generate 25 steps, feeding back each step
                for _ in range(25):
                    # Get the last generated token embedding (or context if first step)
                    last_token = x[:, -1:, :]  # (B, 1, embedding_dim)
            
                    # Predict the next token embedding
                    pred_embedding, h_new, _ = self.cond_vrnn(last_token, h_new)  # (B, 1, embedding_dim)
            
                    # Append to the generated sequence
            
                    # Update context: shift left and append new token
                    x = torch.cat([context[:, 1:, :], pred_embedding], dim=1)

               

                
        dropout_mask = (torch.rand_like(x) > self.config.dropout).float() / (1.0 - self.config.dropout)
        x = x * dropout_mask  
       
        # Pass through each block
        for i, block in enumerate(self.transformer.residual):
            x = block(x, rope_freqs=self.rope_freqs)   

        # Final layernorm
        x = self.transformer.ln_f(x)
        x = x[:,:t,:]
        z_prime = x[:, -x_hat.shape[1]:, :]  # for example
        lambda_mse = 0.5
        lambda_cos = 0.5 #heca8se we want patterns
        vrnn_loss = lambda_mse * F.mse_loss(x_hat, z_prime) + lambda_cos * (1 - F.cosine_similarity(x_hat, z_prime, dim=-1).mean())
                
        return x,vrnn_loss
    
        
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
