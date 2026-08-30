"""A from-scratch forward pass for Qwen2.5 (Phase 2).

The teaching goal is the *attention + KV-cache machinery*, not re-deriving every
matmul. So we **borrow the weight-bearing modules** from a loaded HF checkpoint
(embeddings, projections, layernorms, MLP, lm_head — calling them gives correct
weights and biases for free) and **reimplement the parts that matter**: RoPE, the
attention core (scores → causal mask → softmax → weighted sum), and grouped-query
attention (repeat_kv). This first cut does a plain full-sequence forward; the KV
cache and the incremental decode loop build on it in the next rung.

Correctness is pinned by a parity test: our logits must match HF's on a fixed
prompt, and (next rung) our greedy tokens must match ``model.generate`` exactly.
"""
from __future__ import annotations


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class MiniQwen:
    """Owns the forward loop + attention; leans on HF modules for the weights."""

    def __init__(self, model_path: str, dtype=None, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.device = device
        self.dtype = dtype or torch.float32
        hf = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=self.dtype).to(device).eval()
        self.hf = hf                        # keep a handle (weights live here)
        cfg = hf.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        # transformers 5.0 moved the base into cfg.rope_parameters; older configs
        # keep it as cfg.rope_theta. (Qwen2.5 uses 1e6, not the 10000 default.)
        rope_params = getattr(cfg, "rope_parameters", None) or {}
        self.rope_theta = float(
            rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))

        # RoPE inverse frequencies (reimplemented — central to the teaching story).
        inv = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self._inv_freq = inv.to(device)     # [head_dim/2]

    # --- reimplemented pieces ----------------------------------------------
    def _rope(self, positions):
        """Return (cos, sin) each [T, head_dim] for the given token positions."""
        import torch

        freqs = torch.outer(positions.float(), self._inv_freq)   # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                  # [T, head_dim]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _apply_rope(self, x, cos, sin):
        # x: [B, n_heads, T, head_dim]; cos/sin: [T, head_dim]
        return (x * cos) + (_rotate_half(x) * sin)

    def _repeat_kv(self, x):
        # [B, n_kv, T, d] -> [B, n_heads, T, d] by repeating each kv head.
        import torch

        if self.n_kv == self.n_heads:
            return x
        b, kv, t, d = x.shape
        reps = self.n_heads // self.n_kv
        return x[:, :, None, :, :].expand(b, kv, reps, t, d).reshape(b, kv * reps, t, d)

    def _attention(self, q, k, v, start: int = 0):
        """Scaled-dot-product attention with a causal mask over cached + new keys.

        q: [B,H,T,d]; k,v: [B,H,S,d] where S is the total cached length (>= T). A
        query at new-token index i (absolute position ``start + i``) may attend to
        key j iff ``j <= start + i``. With no cache (start=0, S=T) this is the
        plain lower-triangular causal mask; on a decode step (T=1) nothing is
        masked (the one query sees all past keys and itself).
        """
        import torch

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)   # [B,H,T,S]
        t, s = scores.shape[-2], scores.shape[-1]
        i = torch.arange(t, device=q.device).unsqueeze(1)            # query rows
        j = torch.arange(s, device=q.device).unsqueeze(0)            # key cols
        scores = scores.masked_fill(j > (start + i), float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return weights @ v                                           # [B,H,T,d]

    def _decoder_layer(self, layer, idx, h, cos, sin, cache, seq_id, start):
        b, t, _ = h.shape
        residual = h
        x = layer.input_layernorm(h)
        attn = layer.self_attn
        q = attn.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if cache is not None:                       # single-sequence decode path
            k1 = k[0].transpose(0, 1)               # [T, n_kv, d]
            v1 = v[0].transpose(0, 1)
            full_k, full_v = cache.append(seq_id, idx, k1, v1)      # [S, n_kv, d]
            k = full_k.transpose(0, 1).unsqueeze(0)                 # [1, n_kv, S, d]
            v = full_v.transpose(0, 1).unsqueeze(0)
        out = self._attention(q, self._repeat_kv(k), self._repeat_kv(v), start=start)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        h = residual + attn.o_proj(out)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))
        return h

    # --- public forward -----------------------------------------------------
    def forward(self, input_ids, cache=None, seq_id: int = 0, start_pos: int = 0):
        """input_ids: [B, T] -> logits [B, T, vocab].

        With ``cache=None`` this is a plain full-sequence forward (B may be >1).
        With a ``cache`` it's the incremental path (B must be 1): keys/values are
        appended at ``start_pos`` and attention runs over the whole cached prefix.
        """
        import torch

        model = self.hf.model
        with torch.no_grad():
            h = model.embed_tokens(input_ids)
            positions = torch.arange(start_pos, start_pos + input_ids.shape[1],
                                     device=self.device)
            cos, sin = self._rope(positions)
            for idx, layer in enumerate(model.layers):
                h = self._decoder_layer(layer, idx, h, cos, sin, cache, seq_id, start_pos)
            h = model.norm(h)
            return self.hf.lm_head(h)
