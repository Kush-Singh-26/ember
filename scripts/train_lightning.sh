#!/usr/bin/env bash
# =============================================================================
# Ember-275M — Training launch wrapper for Lightning AI
# Usage:
#   bash scripts/train_lightning.sh              # single A100 (default)
#   bash scripts/train_lightning.sh --gpu h100   # single H100
#   bash scripts/train_lightning.sh --gpu 4xa100 # 4x A100 DDP
# =============================================================================
set -e

EMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EMBER_DIR"

# ── Parse args ─────────────────────────────────────────────────────────────
GPU="a100"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$GPU" in
    a100)   PROFILE="lightning-a100";   NUM_GPUS=1 ;;
    h100)   PROFILE="lightning-h100";   NUM_GPUS=1 ;;
    4xa100) PROFILE="lightning-4xa100"; NUM_GPUS=4 ;;
    *)
        echo "Unknown GPU type: $GPU. Use: a100 | h100 | 4xa100"
        exit 1
        ;;
esac

# ── Source env vars ────────────────────────────────────────────────────────
source ~/.bashrc 2>/dev/null || true

# Verify required env vars
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN is not set. Run: bash scripts/lightning_setup.sh"
    exit 1
fi
if [ -z "$FORGE_DATA_DIR" ]; then
    echo "❌ FORGE_DATA_DIR is not set. Run: bash scripts/lightning_setup.sh"
    exit 1
fi

# ── Confirm FineWeb-Edu shards are present ─────────────────────────────────
SHARD_COUNT=$(ls "$FORGE_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
if [ "$SHARD_COUNT" -lt 12 ]; then
    echo "⚠️  Only $SHARD_COUNT/12 FineWeb-Edu shards found in $FORGE_DATA_DIR"
    echo "    Run: bash scripts/lightning_setup.sh  to download the rest."
    echo "    Continuing anyway — missing shards will be streamed from HF Hub."
fi

echo "============================================================"
echo " Ember-275M — Lightning AI Training"
echo " Profile : $PROFILE"
echo " GPUs    : $NUM_GPUS"
echo " Data dir: $FORGE_DATA_DIR ($SHARD_COUNT/12 FineWeb shards cached)"
echo " Checkpoint: $(ls outputs/ 2>/dev/null | grep checkpoint | sort -V | tail -1 || echo 'none (fresh start)')"
echo "============================================================"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────
export FORGE_PROFILE="$PROFILE"

if [ "$NUM_GPUS" -eq 1 ]; then
    echo ">>> Launching single-GPU training..."
    python train.py
else
    echo ">>> Launching $NUM_GPUS-GPU DDP training via torchrun..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost \
        --master_port=29500 \
        train.py
fi
