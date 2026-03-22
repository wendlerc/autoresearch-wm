"""
One-time data preparation for Doom world model experiments.
Downloads DC-AE latent shards from Hugging Face Hub.

Usage:
    python prepare.py                  # download 5 shards (~20 GB)
    python prepare.py --num-shards 1   # download 1 shard (~4 GB, for testing)
    python prepare.py --num-shards -1  # download all shards (~770 GB)

Data is stored in ~/.cache/autoresearch-wm/.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

import requests

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600          # training time budget in seconds (10 minutes)
CLIP_LEN = 30             # frames per clip (1s @ 30fps)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-wm")
DATA_DIR = os.path.join(CACHE_DIR, "data")
HF_REPO = "chrisxx/doom-2players-latents"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
DEFAULT_NUM_SHARDS = 5

# ---------------------------------------------------------------------------
# Shard listing
# ---------------------------------------------------------------------------

def list_shards():
    """List available shard filenames from the HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        all_files = api.list_repo_files(HF_REPO, repo_type="dataset")
        shards = sorted([f for f in all_files if f.startswith("data/latent-") and f.endswith(".tar")])
        return shards
    except Exception as e:
        print(f"Warning: Could not list shards from HF API ({e}), using fallback range")
        return [f"data/latent-{i:06d}.tar" for i in range(1, 187)]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_shard(args):
    """Download a single shard with retries."""
    hf_path, local_path, idx, total = args
    local_path = Path(local_path)

    if local_path.exists():
        return f"  [{idx}/{total}] {local_path.name} already exists, skipping"

    url = f"{HF_BASE_URL}/{hf_path}"
    tmp_path = local_path.with_suffix(".tmp")

    for attempt in range(5):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            os.replace(tmp_path, local_path)
            return f"  [{idx}/{total}] Downloaded {local_path.name}"
        except Exception as e:
            wait = 2 ** attempt
            if attempt < 4:
                time.sleep(wait)
            else:
                if tmp_path.exists():
                    tmp_path.unlink()
                return f"  [{idx}/{total}] FAILED {local_path.name}: {e}"

    return f"  [{idx}/{total}] FAILED {local_path.name}"


def download_data(num_shards=DEFAULT_NUM_SHARDS, num_workers=4):
    """Download doom latent shards."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Listing available shards from {HF_REPO}...")
    all_shards = list_shards()
    print(f"Available: {len(all_shards)} shards (~{len(all_shards) * 4.1:.0f} GB total)")

    if num_shards < 0:
        shards = all_shards
    else:
        shards = all_shards[:num_shards]

    # We reserve the LAST shard as validation (pinned)
    val_shard = all_shards[-1]
    if val_shard not in shards:
        shards.append(val_shard)

    print(f"Downloading {len(shards)} shard(s) (~{len(shards) * 4.1:.0f} GB)...")

    tasks = []
    for i, hf_path in enumerate(shards):
        local_name = hf_path.replace("data/", "")
        local_path = os.path.join(DATA_DIR, local_name)
        tasks.append((hf_path, local_path, i + 1, len(shards)))

    if num_workers > 1 and len(tasks) > 1:
        with Pool(min(num_workers, len(tasks))) as pool:
            for msg in pool.imap_unordered(download_shard, tasks):
                print(msg)
    else:
        for task in tasks:
            print(download_shard(task))

    # Write metadata
    n_downloaded = len(list(Path(DATA_DIR).glob("latent-*.tar")))
    val_local = os.path.join(DATA_DIR, val_shard.replace("data/", ""))
    meta_path = os.path.join(CACHE_DIR, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"data_dir={DATA_DIR}\n")
        f.write(f"val_shard={val_local}\n")
        f.write(f"num_shards={n_downloaded}\n")
        f.write(f"clip_len={CLIP_LEN}\n")
        f.write(f"time_budget={TIME_BUDGET}\n")

    return n_downloaded, val_local


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_data():
    """Quick verification that data looks correct."""
    import tarfile
    import numpy as np

    shards = sorted(Path(DATA_DIR).glob("latent-*.tar"))
    if not shards:
        print("ERROR: No shards found!")
        return False

    shard = shards[0]
    with tarfile.open(shard) as tf:
        members = tf.getnames()
        eps = set()
        for name in members:
            if name.endswith('.latents_p1.npy'):
                eps.add(name.rsplit('.latents_p1.npy', 1)[0])

        if not eps:
            print(f"ERROR: No episodes found in {shard.name}")
            return False

        ep = sorted(eps)[0]
        lat_f = tf.extractfile(f"{ep}.latents_p1.npy")
        lat = np.load(lat_f)
        act_f = tf.extractfile(f"{ep}.actions_p1.npy")
        act = np.load(act_f)

        print(f"Verification OK:")
        print(f"  Shards: {len(shards)}")
        print(f"  Sample episode: {ep}")
        print(f"  Latent shape: {lat.shape} (expected: (N, 32, 15, 20))")
        print(f"  Action shape: {act.shape} (expected: (N, 14))")
        print(f"  Latent dtype: {lat.dtype}, range: [{lat.min():.2f}, {lat.max():.2f}]")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Doom latent dataset")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS,
                        help=f"Number of shards to download (default: {DEFAULT_NUM_SHARDS}, -1 for all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Download parallelism (default: 4)")
    args = parser.parse_args()

    t0 = time.time()
    n_shards, val_shard = download_data(args.num_shards, args.num_workers)
    print(f"\nDownloaded {n_shards} shard(s) in {time.time() - t0:.1f}s")
    print(f"Validation shard: {val_shard}")
    print(f"Data directory: {DATA_DIR}")

    print("\nVerifying data...")
    if verify_data():
        print(f"\nReady! Run: python train.py")
    else:
        print("\nData verification failed!")
        sys.exit(1)
