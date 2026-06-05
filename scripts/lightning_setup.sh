#!/usr/bin/env bash
# =============================================================================
# Ember-275M — Lightning AI Studio First-Time Setup
# Run ONCE after cloning the repo into a new Studio.
# Usage: bash scripts/lightning_setup.sh
# =============================================================================
set -e

EMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_DATA_DIR="/teamspace/studios/this_studio/ember-data"
FINEWEB_DIR="$STUDIO_DATA_DIR/fineweb-edu-parquet"
NUM_SHARDS=12

echo "============================================================"
echo " Ember-275M  —  Lightning AI A100 Setup"
echo " Studio data dir: $STUDIO_DATA_DIR"
echo "============================================================"

# ── 1. Install Python dependencies ─────────────────────────────────────────
echo ""
echo ">>> [1/4] Installing Python dependencies..."
pip install uv --quiet
uv pip install -r "$EMBER_DIR/requirements.txt"
# Install lm_forge — adjust the path if you have a local copy
pip install lm-forge --quiet
echo "    ✅ Dependencies installed."

# ── 2. Write persistent environment variables ──────────────────────────────
echo ""
echo ">>> [2/4] Writing persistent env vars to ~/.bashrc..."

# Prompt for HF token if not already set
if [ -z "$HF_TOKEN" ]; then
    echo -n "    Enter your HuggingFace token (hf_...): "
    read -s HF_TOKEN
    echo ""
fi

# Write to bashrc (idempotent — skip if already present)
if ! grep -q "FORGE_DATA_DIR" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

# ── Ember-275M Lightning AI config ──────────────────────────────────────────
export HF_TOKEN="$HF_TOKEN"
export FORGE_NO_SKIP=1
export FORGE_DATA_DIR="$FINEWEB_DIR"
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
EOF
    echo "    ✅ Env vars written to ~/.bashrc"
else
    echo "    ℹ️  Env vars already in ~/.bashrc — skipping."
fi

# Export for the current shell session
export HF_TOKEN="$HF_TOKEN"
export FORGE_NO_SKIP=1
export FORGE_DATA_DIR="$FINEWEB_DIR"

# ── 3. Pre-download FineWeb-Edu shards ─────────────────────────────────────
echo ""
echo ">>> [3/4] Pre-downloading FineWeb-Edu ($NUM_SHARDS shards → $FINEWEB_DIR)..."
echo "    This is a one-time download (~26 GB). Existing shards are skipped."
mkdir -p "$FINEWEB_DIR"

BASE_URL="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD=$(printf "%03d_%05d.parquet" $((i / 10)) $((i % 10)))
    DEST="$FINEWEB_DIR/$SHARD"
    if [ ! -f "$DEST" ]; then
        echo "    Downloading $SHARD..."
        wget -q --show-progress "$BASE_URL/$SHARD" -O "$DEST"
    else
        echo "    $SHARD already exists — skipping."
    fi
done
echo "    ✅ FineWeb-Edu download complete ($NUM_SHARDS shards)."

# ── 4. Pull latest checkpoint from HF Hub ──────────────────────────────────
echo ""
echo ">>> [4/4] Pulling latest checkpoint from HF Hub..."
cd "$EMBER_DIR"
forge pull
echo "    ✅ Checkpoint pulled into ./outputs/"

echo ""
echo "============================================================"
echo " Setup complete! Start training with:"
echo ""
echo "   bash scripts/train_lightning.sh"
echo ""
echo " Or manually:"
echo "   source ~/.bashrc"
echo "   FORGE_PROFILE=lightning-a100 python train.py"
echo "============================================================"
