"""
Doom latent diffusion forcing trainer — single-file implementation.
Trains a causal diffusion transformer (DiT) on DC-AE encoded Doom latents.

Based on: https://github.com/wendlerc/toy-wm
Usage: uv run train.py
"""

import io
import os
import sys
import time
import math
import random
from pathlib import Path
from functools import partial
from multiprocessing import Value

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from tqdm import tqdm
from muon import SingleDeviceMuonWithAuxAdam
import webdataset as wds
from webdataset.filters import _shuffle

# ---------------------------------------------------------------------------
# Configuration (agent-modifiable)
# ---------------------------------------------------------------------------

# Architecture
D_MODEL = 512
N_HEADS = 32
N_BLOCKS = 7
PATCH_SIZE = 2
N_WINDOW = 30
IN_CHANNELS = 32
HEIGHT = 16           # latent height (padded from 15)
WIDTH = 20            # latent width
ROPE_C = 5000
ROPE_TYPE = "vid"          # "1d" = sequential RoPE, "vid" = VidRoPE (x,y,t)
ROPE_D_X = 6              # spatial-x dims (3 pairs)
ROPE_D_Y = 4              # spatial-y dims (2 pairs)
ROPE_D_T = 6              # temporal dims (3 pairs); sum must be <= d_head
ROPE_C_X = 100
ROPE_C_Y = 100
ROPE_C_T = 100
ROPE_FACTOR_X = 0.6283185 # 2*pi/10 (ctx_x)
ROPE_FACTOR_Y = 0.7853982 # 2*pi/8  (ctx_y)
ROPE_FACTOR_T = 0.6283185 # kt=3: 3*2*pi/30
N_REGISTERS = 1
EXPANSION = 3
T_NOISE = 1000        # noise schedule resolution

# Training
BATCH_SIZE = 1
LR1 = 0.01            # Muon lr for body params (>=2D)
LR2 = 4e-4            # Adam lr for gains/biases/embeddings
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0
WARMUP_STEPS = 50
ACTION_DROPOUT = 0.2
GRAD_CLIP = 10.0
DTYPE = t.bfloat16

# Data
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-wm")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FPS = 30
DURATION = 1
NUM_WORKERS = 8

# Budget
TIME_BUDGET = 3600     # 1 hour training


def sample_noise_times(batch, seq, device, dtype):
    """Sample diffusion noise times. Agents: modify this to change the noise schedule.
    Used by both training and validation to ensure consistent evaluation."""
    return F.sigmoid(-1.0 + t.randn(batch, seq, device=device, dtype=dtype))


# =========================================================================
# Neural network building blocks
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(t.ones(d))

    def forward(self, x):
        return x / (((x**2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()) * self.w


class GEGLU(nn.Module):
    def __init__(self, d_in, d_mid, d_out):
        super().__init__()
        self.up_proj = nn.Linear(d_in, d_mid, bias=True)
        self.up_proj.bias.data.zero_()
        self.up_gate = nn.Linear(d_in, d_mid, bias=True)
        self.up_gate.bias.data.zero_()
        self.down = nn.Linear(d_mid, d_out, bias=True)
        self.down.bias.data.zero_()
        self.nonlin = nn.SiLU()

    def forward(self, x):
        return self.down(self.up_proj(x) * self.nonlin(self.up_gate(x)))


class NumericEncoding(nn.Module):
    def __init__(self, C=1e4, dim=64, n_max=10000):
        super().__init__()
        args = t.exp(-math.log(C) * t.arange(0, dim, 2) / dim)
        args = t.arange(n_max)[:, None] * args[None, :]
        pe = t.empty((n_max, dim))
        pe[:, ::2] = t.sin(args)
        pe[:, 1::2] = t.cos(args)
        self.register_buffer("pe", pe)

    def forward(self, num):
        return self.pe[num.long()]


# =========================================================================
# RoPE
# =========================================================================

def compute_trig(d_head, n_ctx, C, factor=1.):
    thetas = factor * t.exp(-math.log(C) * t.arange(0, d_head, 2) / d_head)
    thetas = thetas.repeat([2, 1]).T.flatten()
    positions = t.arange(n_ctx)
    all_thetas = positions.unsqueeze(1) * thetas.unsqueeze(0)
    return t.sin(all_thetas), t.cos(all_thetas)


class RoPE(nn.Module):
    def __init__(self, d_head, n_ctx, C=10000):
        super().__init__()
        sins, coss = compute_trig(d_head, n_ctx, C)
        self.register_buffer('sins', sins.unsqueeze(0).unsqueeze(2))
        self.register_buffer('coss', coss.unsqueeze(0).unsqueeze(2))

    def forward(self, x, offset=0):
        x_perm = t.empty_like(x)
        even = t.arange(0, x.shape[-1], 2)
        odd = t.arange(1, x.shape[-1], 2)
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        return self.coss[:, offset:offset + x.shape[1]] * x + self.sins[:, offset:offset + x.shape[1]] * x_perm


class VidRoPE(nn.Module):
    """Video RoPE with separate spatial (x, y) and temporal (t) rotary embeddings."""
    def __init__(self, d_head, d_x, d_y, d_t, ctx_x, ctx_y, ctx_t,
                 C_x, C_y, C_t, toks_per_frame, n_registers,
                 factor_x=1., factor_y=1., factor_t=1.):
        super().__init__()
        assert d_x + d_y + d_t <= d_head, f"dx + dy + dt > d_head"
        self.d_head = d_head
        self.d_x, self.d_y, self.d_t = d_x, d_y, d_t
        self.toks_per_frame = toks_per_frame
        self.n_registers = n_registers
        sins_x, coss_x = compute_trig(d_x, ctx_x + 1, C_x, factor=factor_x)
        self.register_buffer("sins_x", sins_x.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_x", coss_x.unsqueeze(0).unsqueeze(2))
        sins_y, coss_y = compute_trig(d_y, ctx_y + 1, C_y, factor=factor_y)
        self.register_buffer("sins_y", sins_y.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_y", coss_y.unsqueeze(0).unsqueeze(2))
        sins_t, coss_t = compute_trig(d_t, ctx_t, C_t, factor=factor_t)
        self.register_buffer("sins_t", sins_t.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_t", coss_t.unsqueeze(0).unsqueeze(2))
        n_frames = ctx_t
        pos_x = t.arange(ctx_x).repeat(ctx_y)
        pos_x = t.cat([pos_x, t.tensor([ctx_x], dtype=t.int32)])
        pos_x = pos_x.repeat(n_frames)
        pos_y = t.arange(ctx_y).repeat_interleave(ctx_x)
        pos_y = t.cat([pos_y, t.tensor([ctx_y], dtype=t.int32)])
        pos_y = pos_y.repeat(n_frames)
        pos_t = t.arange(n_frames).repeat_interleave(toks_per_frame)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.register_buffer("pos_t", pos_t)
        # Precompute even/odd indices for each dim to avoid dynamic arange in rotate
        max_d = max(d_x, d_y, d_t)
        self.register_buffer("_even", t.arange(0, max_d, 2))
        self.register_buffer("_odd", t.arange(1, max_d, 2))

    def rotate(self, x, pos_idcs, coss, sins):
        d = x.shape[-1]
        x_perm = t.empty(x.shape, device=x.device, dtype=x.dtype)
        even, odd = self._even[:d // 2], self._odd[:d // 2]
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        return coss[:, pos_idcs] * x + sins[:, pos_idcs] * x_perm

    def forward(self, x, offset=0):
        x = x.clone()
        x[:, :, :, :self.d_x] = self.rotate(
            x[:, :, :, :self.d_x], self.pos_x[:x.shape[1]], self.coss_x, self.sins_x)
        x[:, :, :, self.d_x:self.d_x + self.d_y] = self.rotate(
            x[:, :, :, self.d_x:self.d_x + self.d_y], self.pos_y[:x.shape[1]], self.coss_y, self.sins_y)
        x[:, :, :, self.d_x + self.d_y:self.d_x + self.d_y + self.d_t] = self.rotate(
            x[:, :, :, self.d_x + self.d_y:self.d_x + self.d_y + self.d_t],
            self.pos_t[:x.shape[1]] + (offset // self.toks_per_frame), self.coss_t, self.sins_t)
        return x


# =========================================================================
# Patch / UnPatch
# =========================================================================

class Patch(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        dim = out_channels
        if dim % 32 == 0 and dim > 32:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2),
                nn.SiLU(), nn.GroupNorm(32, dim // 2),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2),
                nn.SiLU(), nn.GroupNorm(32, dim // 2),
            )
        else:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2),
                nn.SiLU(),
            )
        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

    def forward(self, x):
        batch, dur, c, h, w = x.shape
        x = self.init_conv_seq(x.reshape(-1, c, h, w))
        B, C, H, W = x.size()
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        x = self.x_embedder(x)
        return x.reshape(batch, dur, -1, self.out_channels)


class UnPatch(nn.Module):
    def __init__(self, height, width, in_channels=64, out_channels=3, patch_size=2):
        super().__init__()
        self.height, self.width = height, width
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.unpatch = nn.Linear(in_channels, out_channels * patch_size ** 2)

    def forward(self, x):
        batch, dur, seq, d = x.shape
        x = self.unpatch(x.reshape(-1, seq, d))
        c, p = self.out_channels, self.patch_size
        h, w = self.height // p, self.width // p
        x = x.reshape(-1, h, w, p, p, c)
        x = t.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x.reshape(batch, dur, c, self.height, self.width)


# =========================================================================
# Attention (no KV cache)
# =========================================================================

def create_block_causal_mask_mod(block_size):
    def block_causal_mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) | ((q_idx // block_size) == (kv_idx // block_size))
    return block_causal_mask_mod


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, rope=None, use_flex=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.QKV = nn.Linear(d_model, 3 * d_model)
        self.O = nn.Linear(d_model, d_model)
        self.lnq = RMSNorm(self.d_head)
        self.lnk = RMSNorm(self.d_head)
        self.rope = rope
        self.use_flex = use_flex

    def forward(self, x, mask=None, k_cache=None, v_cache=None):
        qkv = self.QKV(x)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        b, s, d = q.shape
        q = q.reshape(b, s, self.n_heads, self.d_head)
        k_new = k_new.reshape(b, s, self.n_heads, self.d_head)
        v_new = v_new.reshape(b, s, self.n_heads, self.d_head)
        if k_cache is not None:
            k = t.cat([k_cache, k_new], dim=1)
            v = t.cat([v_cache, v_new], dim=1)
            offset = k_cache.shape[1]
        else:
            k, v = k_new, v_new
            offset = 0
        q = self.lnq(q).to(dtype=self.QKV.weight.dtype)
        k = self.lnk(k).to(dtype=self.QKV.weight.dtype)
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k)
        if self.use_flex and k_cache is None:
            q_f = q.permute(0, 2, 1, 3)
            k_f = k.permute(0, 2, 1, 3)
            v_f = v.permute(0, 2, 1, 3)
            z = flex_attention(q_f, k_f, v_f, scale=1., block_mask=mask) if mask is not None else flex_attention(q_f, k_f, v_f, scale=1.)
            z = z.permute(0, 2, 1, 3)
        else:
            q_p, k_p, v_p = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            z = F.scaled_dot_product_attention(q_p, k_p, v_p, is_causal=(k_cache is None))
            z = z.permute(0, 2, 1, 3)
        return self.O(z.reshape(b, s, d)), k_new, v_new


# =========================================================================
# Doom1P action embedding
# =========================================================================

class Doom1P(nn.Module):
    def __init__(self, d_model, d_turn=32, n_max_turn_emb=100):
        super().__init__()
        self.action_emb = nn.Linear(14 + d_turn, d_model, bias=False)
        self.angle_emb = NumericEncoding(dim=d_turn, n_max=n_max_turn_emb)
        self.n_max_turn_emb = n_max_turn_emb
        self.register_buffer("uncond_action", t.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=t.float32))

    def forward(self, actions):
        dtype = self.action_emb.weight.dtype
        actions = actions.to(dtype)
        angle = self.angle_emb(((12.5 + actions[:, :, -1]) / 25 * (self.n_max_turn_emb - 1)).int())
        actions = actions.clone()
        actions[actions[:, :, 0] == 1] = t.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=actions.device)
        angle[actions[:, :, 0] == 1] = 0
        return self.action_emb(t.cat([actions[:, :, :-1], angle], dim=-1))

    @property
    def unconditional_action(self):
        return self.uncond_action


# =========================================================================
# DiT model
# =========================================================================

def modulate(x, shift, scale):
    b, s, d = x.shape
    toks_per_frame = s // shift.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]
    return x.reshape(b, s, d)


def gate_fn(x, g):
    b, s, d = x.shape
    toks_per_frame = s // g.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * g[:, :, None, :]
    return x.reshape(b, s, d)


class CausalBlock(nn.Module):
    def __init__(self, d_model, expansion, n_heads, rope=None, use_flex=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.selfattn = Attention(d_model, n_heads, rope=rope, use_flex=use_flex)
        self.norm2 = RMSNorm(d_model)
        self.geglu = GEGLU(d_model, expansion * d_model, d_model)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(self, z, cond, mask_self, cached_k=None, cached_v=None):
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = z
        z = modulate(self.norm1(z), mu1, sigma1)
        z, k_new, v_new = self.selfattn(z, mask=mask_self, k_cache=cached_k, v_cache=cached_v)
        z = residual + gate_fn(z, c1)
        residual = z
        z = modulate(self.norm2(z), mu2, sigma2)
        z = self.geglu(z)
        z = residual + gate_fn(z, c2)
        return z, k_new, v_new


class CausalDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.height = HEIGHT
        self.width = WIDTH
        self.in_channels = IN_CHANNELS
        self.n_window = N_WINDOW
        self.d_model = D_MODEL
        self.n_heads = N_HEADS
        self.d_head = D_MODEL // N_HEADS
        self.n_blocks = N_BLOCKS
        self.patch_size = PATCH_SIZE
        self.n_registers = N_REGISTERS
        self.toks_per_frame = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE) + N_REGISTERS

        if ROPE_TYPE == "vid":
            self.rope_seq = VidRoPE(
                self.d_head, ROPE_D_X, ROPE_D_Y, ROPE_D_T,
                WIDTH // PATCH_SIZE, HEIGHT // PATCH_SIZE, N_WINDOW,
                ROPE_C_X, ROPE_C_Y, ROPE_C_T,
                self.toks_per_frame, N_REGISTERS,
                factor_x=ROPE_FACTOR_X, factor_y=ROPE_FACTOR_Y, factor_t=ROPE_FACTOR_T,
            )
        else:
            rope_tmax = N_WINDOW * self.toks_per_frame
            self.rope_seq = RoPE(self.d_head, rope_tmax, C=ROPE_C)

        self.blocks = nn.ModuleList([
            CausalBlock(D_MODEL, EXPANSION, N_HEADS, rope=self.rope_seq, use_flex=True)
            for _ in range(N_BLOCKS)
        ])
        self.patch = Patch(in_channels=IN_CHANNELS, out_channels=D_MODEL, patch_size=PATCH_SIZE)
        self.norm = RMSNorm(D_MODEL)
        self.unpatch = UnPatch(HEIGHT, WIDTH, in_channels=D_MODEL, out_channels=IN_CHANNELS, patch_size=PATCH_SIZE)
        self.action_emb = Doom1P(D_MODEL)
        self.registers = nn.Parameter(t.randn(N_REGISTERS, D_MODEL) * 1 / D_MODEL ** 0.5)
        self.time_emb = NumericEncoding(dim=D_MODEL, n_max=T_NOISE)
        self.time_emb_mixer = nn.Linear(D_MODEL, D_MODEL)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(D_MODEL, 2 * D_MODEL, bias=True))
        self.mask = create_block_mask(
            create_block_causal_mask_mod(self.toks_per_frame),
            B=None, H=None,
            Q_LEN=self.toks_per_frame * N_WINDOW,
            KV_LEN=self.toks_per_frame * N_WINDOW,
        )

    def forward(self, z, actions, ts, cached_k=None, cached_v=None):
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])
        a = self.action_emb(actions)
        ts_scaled = (ts.float() * (T_NOISE - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        z = self.patch(z)
        zr = t.cat((z, self.registers[None, None].repeat(z.shape[0], z.shape[1], 1, 1)), dim=2)
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d)
        # Use flex mask only for full-window training; skip for single-frame or cached inference
        if cached_k is not None or zr.shape[1] != self.toks_per_frame * N_WINDOW:
            mask_self = None
        else:
            mask_self = self.mask
        k_updates, v_updates = [], []
        for bidx, block in enumerate(self.blocks):
            ks = cached_k[bidx] if cached_k is not None else None
            vs = cached_v[bidx] if cached_v is not None else None
            zr, k_new, v_new = block(zr, cond, mask_self, cached_k=ks, cached_v=vs)
            k_updates.append(k_new.unsqueeze(0))
            v_updates.append(v_new.unsqueeze(0))
        k_updates = t.cat(k_updates, dim=0)
        v_updates = t.cat(v_updates, dim=0)
        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, durzr, seqzr, d)
        return self.unpatch(zr[:, :, :-N_REGISTERS]), k_updates, v_updates


# =========================================================================
# Optimizer
# =========================================================================

def get_muon(model, lr1, lr2, betas, weight_decay):
    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]
    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]
    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=lr1, weight_decay=weight_decay),
        dict(params=hidden_gains_biases + list(other_weights), use_muon=False, lr=lr2, betas=betas, weight_decay=weight_decay),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def lr_lambda(step, max_steps, warmup_steps=200):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0  # constant LR after warmup (Round 1 best)


# =========================================================================
# WebDataset doom loader
# =========================================================================

class _SharedEpoch:
    def __init__(self, epoch=0):
        self.shared_epoch = Value('i', epoch)
    def set_value(self, epoch): self.shared_epoch.value = epoch
    def get_value(self): return self.shared_epoch.value


class _ResampledShards(IterableDataset):
    def __init__(self, urls, epoch=-1):
        super().__init__()
        self.urls = list(urls)
        self.epoch = epoch
        self.rng = random.Random()
    def __iter__(self):
        if isinstance(self.epoch, _SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1; epoch = self.epoch
        seed = (get_worker_info().seed if get_worker_info() else 0) + epoch
        self.rng.seed(seed)
        while True:
            yield dict(url=self.rng.choice(self.urls))


class _ExplodeClips(wds.PipelineStage):
    def __init__(self, clip_len, rng):
        self.clip_len = clip_len
        self.rng = rng

    def run(self, src):
        for sample in src:
            raw_p1 = sample["latents_p1.npy"]
            n_frames = raw_p1.shape[0]
            latents_p1 = np.zeros((n_frames, 32, 16, 20), dtype=np.float32)
            latents_p1[:, :, 1:, :] = raw_p1
            n_clips = n_frames // self.clip_len
            if n_clips == 0:
                continue
            actions_p1 = np.zeros((n_frames, 15), dtype=np.float32)
            actions_p1[:, 1:] = sample.get("actions_p1.npy", np.zeros((n_frames, 14), dtype=np.float32))
            has_p2 = "latents_p2.npy" in sample
            if has_p2:
                raw_p2 = sample["latents_p2.npy"]
                latents_p2 = np.zeros((n_frames, 32, 16, 20), dtype=np.float32)
                latents_p2[:, :, 1:, :] = raw_p2
                actions_p2 = np.zeros((n_frames, 15), dtype=np.float32)
                actions_p2[:, 1:] = sample.get("actions_p2.npy", np.zeros((n_frames, 14), dtype=np.float32))
            starts = list(range(0, n_clips * self.clip_len, self.clip_len))
            self.rng.shuffle(starts)
            for start in starts:
                end = start + self.clip_len
                clip = {"latents_p1": latents_p1[start:end], "actions_p1": actions_p1[start:end]}
                if has_p2:
                    clip["latents_p2"] = latents_p2[start:end]
                    clip["actions_p2"] = actions_p2[start:end]
                yield clip


def _decode_npy(sample):
    for key in list(sample.keys()):
        if key.endswith(".npy") and isinstance(sample[key], bytes):
            sample[key] = np.load(io.BytesIO(sample[key]))
    return sample


def _collate(batch):
    return {k: t.from_numpy(np.stack([b[k] for b in batch])) for k in batch[0].keys()}


def _log_and_continue(exn):
    import logging
    logging.warning(f'WebDataset error: {repr(exn)}')
    return True


def get_doom_loader(shard_urls, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    clip_len = FPS * DURATION + 1
    shared_epoch = _SharedEpoch(0)
    clip_rng = random.Random(42)
    pipeline = [
        _ResampledShards(shard_urls, epoch=shared_epoch),
        wds.tarfile_to_samples(handler=_log_and_continue),
        wds.map(_decode_npy, handler=_log_and_continue),
        _ExplodeClips(clip_len, clip_rng),
        wds.batched(batch_size, partial=False, collation_fn=_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)
    n_samples = len(shard_urls) * 13 * (8000 // clip_len)
    num_worker_batches = math.ceil(n_samples / batch_size / max(1, num_workers))
    dataset = dataset.with_epoch(num_worker_batches)
    return wds.WebLoader(dataset, batch_size=None, shuffle=False,
                         num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)


def iterate_doom(loader):
    """Yield (latents, actions) from doom loader, interleaving P1/P2."""
    for batch in loader:
        for player in ("p1", "p2"):
            lk, ak = f"latents_{player}", f"actions_{player}"
            if lk not in batch:
                continue
            yield batch[lk][:, 1:].float(), batch[ak][:, :-1].float()


# =========================================================================
# Checkpoint management (top-K by val_loss)
# =========================================================================

CKPT_DIR = os.path.join(CACHE_DIR, "checkpoints")
CKPT_TOP_K = 5

def save_checkpoint(model, val_loss, step, agent_name="default"):
    """Save checkpoint and keep only top-K best by val_loss."""
    agent_dir = os.path.join(CKPT_DIR, agent_name)
    os.makedirs(agent_dir, exist_ok=True)
    # Save new checkpoint
    fname = f"ckpt-step={step:06d}-val={val_loss:.6f}.pt"
    path = os.path.join(agent_dir, fname)
    state = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
    t.save({"model": state, "val_loss": val_loss, "step": step}, path)
    print(f"  Saved checkpoint: {fname}")
    # Prune to top-K
    ckpts = sorted(Path(agent_dir).glob("ckpt-*.pt"))
    if len(ckpts) > CKPT_TOP_K:
        # Parse val_loss from filename, keep lowest
        scored = []
        for c in ckpts:
            try:
                vl = float(c.stem.split("val=")[1])
                scored.append((vl, c))
            except (IndexError, ValueError):
                scored.append((float('inf'), c))
        scored.sort(key=lambda x: x[0])
        for _, c in scored[CKPT_TOP_K:]:
            c.unlink()
            print(f"  Pruned checkpoint: {c.name}")


# =========================================================================
# >>> READ-ONLY BELOW THIS LINE — do NOT modify anything below <<<
# The evaluation code (KV cache, sampling, VAE, metrics, AR eval) is fixed.
# Only modify the model, training loop, and hyperparameters above.
# =========================================================================

# =========================================================================
# KV Cache (simple, for AR eval)
# =========================================================================

class SimpleKVCache:
    """Naive growing KV cache for AR inference."""
    def __init__(self):
        self.keys = None   # (n_layers, B, T, H, D)
        self.values = None

    def get(self):
        return self.keys, self.values

    def extend(self, k_new, v_new):
        """k_new/v_new: (n_layers, B, T_new, H, D)"""
        if self.keys is None:
            self.keys = k_new
            self.values = v_new
        else:
            self.keys = t.cat([self.keys, k_new], dim=2)
            self.values = t.cat([self.values, v_new], dim=2)
        # Keep only last (N_WINDOW-1) * toks_per_frame tokens
        max_tokens = (N_WINDOW - 1) * ((HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE) + N_REGISTERS)
        if self.keys.shape[2] > max_tokens:
            self.keys = self.keys[:, :, -max_tokens:]
            self.values = self.values[:, :, -max_tokens:]

    def reset(self):
        self.keys = None
        self.values = None


# =========================================================================
# Sampling (for AR eval)
# =========================================================================

@t.no_grad()
def sample_one_frame(model, noise, action, n_steps=10, cache=None):
    """Denoise one frame using Euler method. Returns (denoised, k_new, v_new)."""
    ts = 1 - t.linspace(0, 1, n_steps + 1, device=noise.device, dtype=noise.dtype)
    ts = 3 * ts / (2 * ts + 1)
    z = noise.clone()
    cached_k, cached_v = cache.get() if cache is not None else (None, None)
    for i in range(len(ts)):
        t_cond = ts[i].reshape(1, 1)
        v_pred, k_new, v_new = model(z, action, t_cond, cached_k=cached_k, cached_v=cached_v)
        if i < len(ts) - 1:
            z = z + (ts[i] - ts[i + 1]) * v_pred
    if cache is not None:
        cache.extend(k_new, v_new)
    return z


# =========================================================================
# VAE decoder + pixel metrics
# =========================================================================

_vae_cache = {}

def _get_vae(device):
    if device not in _vae_cache:
        from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
        import glob
        local = glob.glob("/tmp/dc-ae-cache/snapshots/*/")
        model_path = local[0] if local else "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
        _vae_cache[device] = AutoencoderDC.from_pretrained(model_path, torch_dtype=t.float16).to(device).eval()
    return _vae_cache[device]

def _offload_vae(device):
    vae = _vae_cache.pop(device, None)
    if vae is not None:
        vae.to("cpu")
        _vae_cache["cpu"] = vae
    t.cuda.empty_cache()

def decode_latents_to_rgb(latents, device):
    """Decode (B, T, 32, 16, 20) latents to (B, T, 3, H, W) float [0,1]."""
    vae = _get_vae(device)
    B, T, C, H, W = latents.shape
    flat = latents.reshape(B * T, C, H, W).to(t.float16)
    if flat.shape[-2] == 16:  # strip height padding
        flat = flat[:, :, 1:, :]
    frames = []
    with t.no_grad():
        for i in range(0, B * T, 8):
            rgb = vae.decode(flat[i:i+8]).sample
            rgb = (rgb.clamp(-1, 1) + 1) / 2  # to [0, 1]
            frames.append(rgb.float())
    return t.cat(frames, dim=0)[:B*T].reshape(B, T, 3, -1, rgb.shape[-1])

def pixel_psnr(pred_rgb, target_rgb):
    """PSNR on float [0,1] images. Returns scalar."""
    mse = F.mse_loss(pred_rgb, target_rgb)
    return (10 * t.log10(1.0 / mse.clamp(min=1e-10))).item()

_lpips_net = None
def pixel_lpips(pred_rgb, target_rgb):
    """Mean LPIPS on float [0,1] images. Returns scalar (lower = more similar)."""
    global _lpips_net
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net='alex', verbose=False).to(pred_rgb.device)
    _lpips_net = _lpips_net.to(pred_rgb.device)
    # LPIPS expects [-1, 1]
    p = pred_rgb * 2 - 1
    tgt = target_rgb * 2 - 1
    with t.no_grad():
        # Process frame by frame to save memory
        dists = []
        for i in range(p.shape[0]):
            d = _lpips_net(p[i:i+1], tgt[i:i+1])
            dists.append(d.item())
    return sum(dists) / len(dists)


# =========================================================================
# AR Evaluation (pixel space)
# =========================================================================

AR_EVAL_SEED = 2024     # deterministic clip selection + sampling noise

@t.no_grad()
def compute_ar_metrics(model, val_loader, device, dtype, n_clips=50, n_steps=10):
    """Compute AR and TF metrics in pixel space (PSNR + LPIPS).

    Fully deterministic: fixed seeds for clip selection, shuffling, and sampling noise.
    Uses num_workers=0 loader to eliminate worker-level RNG variance.
    """
    model.eval()
    ar_psnrs, ar_lpips_vals = [], []
    tf_psnrs, tf_lpips_vals = [], []

    # Deterministic: fix ALL RNG sources
    rng_state = t.random.get_rng_state()
    cuda_rng_state = t.cuda.get_rng_state()
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    t.manual_seed(AR_EVAL_SEED)
    t.cuda.manual_seed(AR_EVAL_SEED)
    random.seed(AR_EVAL_SEED)
    np.random.seed(AR_EVAL_SEED)

    # Create a deterministic loader (num_workers=0 avoids worker-level seed variance)
    val_urls = [str(p) for p in sorted(Path(DATA_DIR).glob("latent-*.tar"))[-1:]]
    det_loader = get_doom_loader(val_urls, batch_size=1, num_workers=0)
    val_iter = iterate_doom(det_loader)

    for clip_idx in range(n_clips):
        try:
            frames, actions = next(val_iter)
        except StopIteration:
            val_iter = iterate_doom(val_loader)
            frames, actions = next(val_iter)

        frames = frames[:1, :N_WINDOW].to(device).to(dtype)
        actions = actions[:1, :N_WINDOW].to(device)
        T_frames = frames.shape[1]
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]

        # Decode GT to pixels
        gt_rgb = decode_latents_to_rgb(frames, device)  # (1, T, 3, H, W)

        # --- AR path ---
        ar_cache = SimpleKVCache()
        a0 = t.zeros(1, 1, 15, device=device, dtype=dtype)
        _, k0, v0 = model(frames[:, :1], a0, t.zeros(1, 1, device=device, dtype=dtype))
        ar_cache.extend(k0, v0)

        ar_latents = [frames[:, :1]]
        for tidx in range(1, T_frames):
            noise = t.randn(1, 1, C, H, W, device=device, dtype=dtype)
            pred = sample_one_frame(model, noise, actions[:, tidx:tidx+1],
                                    n_steps=n_steps, cache=ar_cache)
            ar_latents.append(pred)

        ar_seq = t.cat(ar_latents, dim=1)
        ar_rgb = decode_latents_to_rgb(ar_seq, device)

        # Compute pixel metrics (skip frame 0 which is GT)
        ar_psnr = pixel_psnr(ar_rgb[:, 1:].reshape(-1, *ar_rgb.shape[2:]),
                             gt_rgb[:, 1:].reshape(-1, *gt_rgb.shape[2:]))
        ar_lp = pixel_lpips(ar_rgb[:, 1:].reshape(-1, *ar_rgb.shape[2:]),
                            gt_rgb[:, 1:].reshape(-1, *gt_rgb.shape[2:]))
        ar_psnrs.append(ar_psnr)
        ar_lpips_vals.append(ar_lp)

        # --- TF path ---
        tf_cache = SimpleKVCache()
        _, k0, v0 = model(frames[:, :1], a0, t.zeros(1, 1, device=device, dtype=dtype))
        tf_cache.extend(k0, v0)

        tf_latents = [frames[:, :1]]
        for tidx in range(1, T_frames):
            noise = t.randn(1, 1, C, H, W, device=device, dtype=dtype)
            pred = sample_one_frame(model, noise, actions[:, tidx:tidx+1],
                                    n_steps=n_steps, cache=tf_cache)
            tf_latents.append(pred)
            # Inject GT into cache (teacher forcing) — remove sample's cache entry, add GT's
            tf_cache.keys = tf_cache.keys[:, :, :-k0.shape[2]]
            tf_cache.values = tf_cache.values[:, :, :-v0.shape[2]]
            ck, cv = tf_cache.get()
            _, k_gt, v_gt = model(frames[:, tidx:tidx+1], actions[:, tidx:tidx+1],
                                   t.zeros(1, 1, device=device, dtype=dtype),
                                   cached_k=ck, cached_v=cv)
            tf_cache.extend(k_gt, v_gt)

        tf_seq = t.cat(tf_latents, dim=1)
        tf_rgb = decode_latents_to_rgb(tf_seq, device)

        tf_psnr = pixel_psnr(tf_rgb[:, 1:].reshape(-1, *tf_rgb.shape[2:]),
                             gt_rgb[:, 1:].reshape(-1, *gt_rgb.shape[2:]))
        tf_lp = pixel_lpips(tf_rgb[:, 1:].reshape(-1, *tf_rgb.shape[2:]),
                            gt_rgb[:, 1:].reshape(-1, *gt_rgb.shape[2:]))
        tf_psnrs.append(tf_psnr)
        tf_lpips_vals.append(tf_lp)

        print(f"  clip {clip_idx}: ar_psnr={ar_psnr:.1f} ar_lpips={ar_lp:.3f} | tf_psnr={tf_psnr:.1f} tf_lpips={tf_lp:.3f}")

    # Restore all RNG states so eval doesn't affect training
    t.random.set_rng_state(rng_state)
    t.cuda.set_rng_state(cuda_rng_state)
    random.setstate(py_rng_state)
    np.random.set_state(np_rng_state)

    # Offload VAE to free VRAM
    _offload_vae(device)

    model.train()
    return {
        'ar_auto_psnr': sum(ar_psnrs) / len(ar_psnrs),
        'ar_tf_psnr': sum(tf_psnrs) / len(tf_psnrs),
        'ar_gap': sum(tf_psnrs) / len(tf_psnrs) - sum(ar_psnrs) / len(ar_psnrs),
        'ar_auto_lpips': sum(ar_lpips_vals) / len(ar_lpips_vals),
        'ar_tf_lpips': sum(tf_lpips_vals) / len(tf_lpips_vals),
    }


# =========================================================================
# Validation
# =========================================================================

VAL_BATCH_SIZE = 16    # fixed, do NOT derive from BATCH_SIZE
VAL_N_BATCHES = 20     # fixed number of batches for val
VAL_SEED = 42          # deterministic noise for reproducible val

@t.no_grad()
@t._dynamo.disable
def compute_val_loss(model, val_loader, device, dtype):
    """Deterministic val loss: fixed batch size, fixed noise seed, fixed n_batches."""
    model.eval()
    rng_state = t.random.get_rng_state()
    cuda_rng_state = t.cuda.get_rng_state()
    t.manual_seed(VAL_SEED)
    t.cuda.manual_seed(VAL_SEED)
    losses = []
    val_iter = iterate_doom(val_loader)
    for _ in range(VAL_N_BATCHES):
        try:
            frames, actions = next(val_iter)
        except StopIteration:
            val_iter = iterate_doom(val_loader)
            frames, actions = next(val_iter)
        frames = frames[:, :N_WINDOW].to(device).to(dtype)
        actions = actions[:, :N_WINDOW].to(device)
        with t.autocast(device_type="cuda", dtype=dtype):
            ts = sample_noise_times(frames.shape[0], frames.shape[1], device, dtype)
            z = t.randn_like(frames)
            vel_true = frames - z
            x_t = frames - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _ = model(x_t, actions, ts)
            loss = F.mse_loss(vel_pred.double(), vel_true.double())
        losses.append(loss.item())
    # Restore RNG state so val doesn't affect training
    t.random.set_rng_state(rng_state)
    t.cuda.set_rng_state(cuda_rng_state)
    model.train()
    return sum(losses) / len(losses)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    t.backends.cuda.matmul.fp32_precision = "tf32"
    t.backends.cudnn.conv.fp32_precision = "tf32"

    device = "cuda"
    assert t.cuda.is_available(), "CUDA required"

    # --- Data setup ---
    shard_paths = sorted(Path(DATA_DIR).glob("latent-*.tar"))
    assert len(shard_paths) > 0, f"No shards in {DATA_DIR}. Run: python prepare.py"
    shard_urls = [str(p) for p in shard_paths]
    assert len(shard_urls) >= 2, f"Need at least 2 shards (got {len(shard_urls)}). Run: python prepare.py"
    val_urls = [shard_urls[-1]]
    train_urls = shard_urls[:-1]  # strictly disjoint from val

    print(f"Train shards: {len(train_urls)}, Val shards: {len(val_urls)}")
    train_loader = get_doom_loader(train_urls, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = get_doom_loader(val_urls, batch_size=VAL_BATCH_SIZE, num_workers=2)

    # --- Model ---
    model = CausalDit().to(device).to(DTYPE)
    model = t.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params, dtype={DTYPE}")

    # --- Optimizer ---
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizer = get_muon(raw_model, LR1, LR2, BETAS, WEIGHT_DECAY)
    max_steps = 400  # ~expected steps in 10min budget
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=WARMUP_STEPS))

    # --- Training ---
    print(f"Training for {TIME_BUDGET}s...")
    train_iter = iterate_doom(train_loader)
    step = 0
    running_loss = 0.0
    t0_setup = time.time()
    t0_train = time.time()

    while True:
        elapsed = time.time() - t0_train
        if elapsed >= TIME_BUDGET:
            break

        model.train()
        optimizer.zero_grad()
        try:
            frames, actions = next(train_iter)
        except StopIteration:
            train_iter = iterate_doom(train_loader)
            frames, actions = next(train_iter)

        # Action dropout
        uncond = raw_model.action_emb.unconditional_action
        mask_ts = t.rand(actions.shape[0], actions.shape[1], device=device, dtype=DTYPE) <= ACTION_DROPOUT
        mask = mask_ts.unsqueeze(-1).expand_as(actions)
        actions[mask] = uncond.to(actions).unsqueeze(0).expand(mask_ts.sum().item(), -1).reshape(-1)

        frames = frames[:, :N_WINDOW].to(device).to(DTYPE)
        actions = actions[:, :N_WINDOW].to(device)
        ts = sample_noise_times(frames.shape[0], frames.shape[1], device, DTYPE)

        with t.autocast(device_type="cuda", dtype=DTYPE):
            z = t.randn_like(frames)
            vel_true = frames - z
            x_t = frames - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _ = model(x_t, actions, ts)
            loss = F.mse_loss(vel_pred.double(), vel_true.double())

        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
        if step % 50 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | avg {running_loss:.4f} | {elapsed:.0f}s")
        step += 1

    training_seconds = time.time() - t0_train
    print(f"\nTraining done: {step} steps in {training_seconds:.1f}s")

    # --- Validation (outside time budget) ---
    print("Computing validation loss...")
    t.cuda.empty_cache()
    val_loss = compute_val_loss(raw_model, val_loader, device, DTYPE)

    print("Computing AR metrics (pixel PSNR + LPIPS, 50 clips)...")
    ar_metrics = compute_ar_metrics(raw_model, val_loader, device, DTYPE)

    # --- Save checkpoint (top-K by ar_auto_lpips, lower is better) ---
    import socket
    agent_name = os.environ.get("AGENT_NAME", socket.gethostname())
    save_checkpoint(model, ar_metrics['ar_auto_lpips'], step, agent_name=agent_name)

    total_seconds = time.time() - t0_setup
    peak_vram = t.cuda.max_memory_allocated() / 1e6

    # --- Structured output ---
    print(f"\n---")
    print(f"ar_auto_psnr:      {ar_metrics['ar_auto_psnr']:.2f}")
    print(f"ar_auto_lpips:     {ar_metrics['ar_auto_lpips']:.4f}")
    print(f"ar_tf_psnr:        {ar_metrics['ar_tf_psnr']:.2f}")
    print(f"ar_tf_lpips:       {ar_metrics['ar_tf_lpips']:.4f}")
    print(f"ar_gap:            {ar_metrics['ar_gap']:.2f}")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram:.1f}")
    print(f"num_steps:         {step}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"final_train_loss:  {running_loss:.6f}")
