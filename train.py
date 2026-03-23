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
D_MODEL = 384
N_HEADS = 24
N_BLOCKS = 5
PATCH_SIZE = 2
N_WINDOW = 15
IN_CHANNELS = 32
HEIGHT = 16           # latent height (padded from 15)
WIDTH = 20            # latent width
ROPE_C = 5000
N_REGISTERS = 1
EXPANSION = 4
T_NOISE = 1000        # noise schedule resolution

# Training
BATCH_SIZE = 8
LR1 = 0.02            # Muon lr for body params (>=2D)
LR2 = 3e-4            # Adam lr for gains/biases/embeddings
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 30
ACTION_DROPOUT = 0.1
GRAD_CLIP = 10.0
DTYPE = t.bfloat16

# Data
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-wm")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FPS = 30
DURATION = 1
NUM_WORKERS = 8

# Budget
TIME_BUDGET = 600      # 10 minutes training

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

def compute_trig(d_head, n_ctx, C):
    thetas = t.exp(-math.log(C) * t.arange(0, d_head, 2) / d_head)
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

    def forward(self, x, mask=None):
        qkv = self.QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, s, d = q.shape
        q = q.reshape(b, s, self.n_heads, self.d_head)
        k = k.reshape(b, s, self.n_heads, self.d_head)
        v = v.reshape(b, s, self.n_heads, self.d_head)
        q = self.lnq(q).to(dtype=self.QKV.weight.dtype)
        k = self.lnk(k).to(dtype=self.QKV.weight.dtype)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        if self.use_flex:
            q_f = q.permute(0, 2, 1, 3)
            k_f = k.permute(0, 2, 1, 3)
            v_f = v.permute(0, 2, 1, 3)
            z = flex_attention(q_f, k_f, v_f, scale=1., block_mask=mask) if mask is not None else flex_attention(q_f, k_f, v_f, scale=1.)
            z = z.permute(0, 2, 1, 3)
        else:
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            attn = q @ k.permute(0, 1, 3, 2)
            if mask is not None:
                attn = t.where(mask[:attn.shape[-2], :attn.shape[-1]], attn, float("-inf"))
            z = attn.softmax(dim=-1) @ v
            z = z.permute(0, 2, 1, 3)
        return self.O(z.reshape(b, s, d))


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


def drop_path(x, drop_prob, training):
    if not training or drop_prob == 0.:
        return x
    keep = 1. - drop_prob
    mask = t.bernoulli(t.full((x.shape[0], 1, 1), keep, device=x.device, dtype=x.dtype))
    return x * mask / keep


class CausalBlock(nn.Module):
    def __init__(self, d_model, expansion, n_heads, rope=None, use_flex=True, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.selfattn = Attention(d_model, n_heads, rope=rope, use_flex=use_flex)
        self.norm2 = RMSNorm(d_model)
        self.geglu = GEGLU(d_model, expansion * d_model, d_model)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        self.drop_path_rate = drop_path_rate

    def forward(self, z, cond, mask_self):
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = z
        z = modulate(self.norm1(z), mu1, sigma1)
        z = self.selfattn(z, mask=mask_self)
        z = residual + drop_path(gate_fn(z, c1), self.drop_path_rate, self.training)
        residual = z
        z = modulate(self.norm2(z), mu2, sigma2)
        z = self.geglu(z)
        z = residual + drop_path(gate_fn(z, c2), self.drop_path_rate, self.training)
        return z


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

        rope_tmax = N_WINDOW * self.toks_per_frame
        self.rope_seq = RoPE(self.d_head, rope_tmax, C=ROPE_C)

        drop_rates = [0.1 * i / max(1, N_BLOCKS - 1) for i in range(N_BLOCKS)]
        self.blocks = nn.ModuleList([
            CausalBlock(D_MODEL, EXPANSION, N_HEADS, rope=self.rope_seq, use_flex=True,
                        drop_path_rate=drop_rates[i])
            for i in range(N_BLOCKS)
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

    def forward(self, z, actions, ts):
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])
        a = self.action_emb(actions)
        ts_scaled = (ts.float() * (T_NOISE - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        z = self.patch(z)
        zr = t.cat((z, self.registers[None, None].repeat(z.shape[0], z.shape[1], 1, 1)), dim=2)
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d)
        for block in self.blocks:
            zr = block(zr, cond, self.mask)
        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, durzr, seqzr, d)
        return self.unpatch(zr[:, :, :-N_REGISTERS])


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


def lr_lambda(step, max_steps, warmup_steps=200, constant_fraction=0.6):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    post_warmup = max_steps - warmup_steps
    constant_end = warmup_steps + int(constant_fraction * post_warmup)
    if step < constant_end:
        return 1.0
    progress = float(step - constant_end) / float(max(1, max_steps - constant_end))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            z = t.randn_like(frames)
            vel_true = frames - z
            x_t = frames - ts[:, :, None, None, None] * vel_true
            vel_pred = model(x_t, actions, ts)
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
    max_steps = 11000  # calibrated for ~10700 steps with N_WINDOW=15, BS=8, N_BLOCKS=5
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=WARMUP_STEPS, constant_fraction=0.7))

    # --- Training ---
    print(f"Training for {TIME_BUDGET}s...")
    train_iter = iterate_doom(train_loader)
    step = 0
    running_loss = 0.0
    t0_setup = time.time()
    t0_train = time.time()

    # SWA: collect checkpoints during last 30% of training
    SWA_START_FRAC = 0.7
    SWA_INTERVAL = 500
    swa_states = []
    swa_started = False

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
        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=DTYPE))

        with t.autocast(device_type="cuda", dtype=DTYPE):
            z = t.randn_like(frames)
            vel_true = frames - z
            x_t = frames - ts[:, :, None, None, None] * vel_true
            vel_pred = model(x_t, actions, ts)
            loss = F.mse_loss(vel_pred.double(), vel_true.double())

        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
        if step % 50 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | avg {running_loss:.4f} | {elapsed:.0f}s")

        # SWA checkpoint collection
        if not swa_started and elapsed >= SWA_START_FRAC * TIME_BUDGET:
            swa_started = True
            print(f"  SWA: starting collection at step {step}")
        if swa_started and step % SWA_INTERVAL == 0:
            swa_states.append({k: v.clone().cpu() for k, v in raw_model.state_dict().items()})
            print(f"  SWA: saved checkpoint {len(swa_states)} at step {step}")

        step += 1

    # Save final checkpoint for SWA
    swa_states.append({k: v.clone().cpu() for k, v in raw_model.state_dict().items()})
    print(f"  SWA: saved final checkpoint {len(swa_states)} at step {step}")

    training_seconds = time.time() - t0_train
    print(f"\nTraining done: {step} steps in {training_seconds:.1f}s")

    # --- Validation (outside time budget) ---
    print("Computing validation loss (no SWA)...")
    t.cuda.empty_cache()
    val_loss_raw = compute_val_loss(raw_model, val_loader, device, DTYPE)
    print(f"  val_loss (no SWA): {val_loss_raw:.6f}")

    # Apply SWA: average all collected checkpoints
    if len(swa_states) >= 2:
        print(f"Applying SWA with {len(swa_states)} checkpoints...")
        avg_state = {}
        for k in swa_states[0]:
            avg_state[k] = sum(s[k] for s in swa_states) / len(swa_states)
        raw_model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})
        del swa_states, avg_state

    print("Computing validation loss (SWA)...")
    val_loss = compute_val_loss(raw_model, val_loader, device, DTYPE)
    total_seconds = time.time() - t0_setup
    peak_vram = t.cuda.max_memory_allocated() / 1e6

    # --- Structured output ---
    print(f"\n---")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram:.1f}")
    print(f"num_steps:         {step}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"final_train_loss:  {running_loss:.6f}")
