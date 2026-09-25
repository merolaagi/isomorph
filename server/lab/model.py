"""Tiny one-layer transformer for modular addition (a + b mod p).

No LayerNorm, on purpose: without it the residual stream has an exact
orthogonal symmetry, so alignment can be verified to machine precision.
Sites exposed for analysis (final position unless noted):
  resid_pre  (B, 3, d)  token + position embedding, all positions
  attn_out   (B, d)     attention output
  resid_mid  (B, d)     residual after attention
  mlp_post   (B, dmlp)  ReLU activations
  resid_post (B, d)     residual after MLP
  logits     (B, p)
"""
import math

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, d_mlp=512, n_ctx=3):
        super().__init__()
        self.p, self.d, self.h, self.dmlp, self.n_ctx = p, d_model, n_heads, d_mlp, n_ctx
        dh = d_model // n_heads
        self.dh = dh
        s = 1 / math.sqrt(d_model)
        self.W_E = nn.Parameter(torch.randn(p + 1, d_model) * s)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * s)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, dh) * s)
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, dh) * s)
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, dh) * s)
        self.W_O = nn.Parameter(torch.randn(n_heads, dh, d_model) / math.sqrt(dh))
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * s)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = nn.Parameter(torch.randn(d_model, p) * s)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)).bool(), persistent=False)

    def config(self):
        return dict(p=self.p, d_model=self.d, n_heads=self.h, d_mlp=self.dmlp, n_ctx=self.n_ctx)

    def embed(self, x):
        return self.W_E[x] + self.W_pos

    def attn(self, resid):
        q = torch.einsum("bpd,hde->bhpe", resid, self.W_Q)
        k = torch.einsum("bpd,hde->bhpe", resid, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", resid, self.W_V)
        sc = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        sc = sc.masked_fill(~self.mask, float("-inf"))
        pat = sc.softmax(-1)
        z = pat @ v
        out = torch.einsum("bhpe,hed->bpd", z, self.W_O)
        return out[:, -1], pat

    def mlp(self, resid_mid):
        post = torch.relu(resid_mid @ self.W_in + self.b_in)
        return post, post @ self.W_out + self.b_out

    def forward(self, x, cache=None):
        return self.run_from("resid_pre", self.embed(x), cache)

    def run_from(self, site, h, cache=None):
        """Continue the forward pass from an intermediate site (used by stitching)."""
        if site == "resid_pre":
            attn_out, pat = self.attn(h)
            resid_mid = h[:, -1] + attn_out
            if cache is not None:
                cache.update(resid_pre=h, attn_out=attn_out, resid_mid=resid_mid, pattern=pat)
            h, site = resid_mid, "resid_mid"
        if site == "resid_mid":
            post, mlp_out = self.mlp(h)
            resid_post = h + mlp_out
            if cache is not None:
                cache.update(mlp_post=post, resid_post=resid_post)
            h, site = resid_post, "resid_post"
        if site == "resid_post":
            logits = h @ self.W_U
            if cache is not None:
                cache["logits"] = logits
            return logits
        raise ValueError(site)


def make_dataset(p, train_frac, seed):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    x = torch.stack([a, b, eq], 1)
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n = int(train_frac * p * p)
    return x, y, perm[:n], perm[n:]


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    m = TinyTransformer(**blob["config"])
    m.load_state_dict(blob["state"])
    m.eval()
    return m
