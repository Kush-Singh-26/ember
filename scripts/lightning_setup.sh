#!/usr/bin/env bash
# Usage: bash scripts/lightning_setup.sh hf_YOUR_TOKEN_HERE
# Does everything: env vars, deps, FineWeb download, checkpoint pull, training launch

set -e

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: bash scripts/lightning_setup.sh hf_YOUR_TOKEN_HERE"
    exit 1
fi

EMBER_DIR="/teamspace/studios/this_studio/ember"
FORGE_DIR="/teamspace/studios/this_studio/lm_forge"
DATA_DIR="/teamspace/studios/this_studio/ember-data/fineweb-edu-parquet"
BASE_URL="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT"

cd "$EMBER_DIR"

echo "=== [1/5] Writing env vars to ~/.bashrc ==="
grep -v "Ember-275M\|HF_TOKEN\|FORGE_NO_SKIP\|FORGE_DATA_DIR\|NCCL_SOCKET_IFNAME\|GLOO_SOCKET_IFNAME" ~/.bashrc > /tmp/bashrc_clean || true
cp /tmp/bashrc_clean ~/.bashrc
echo "" >> ~/.bashrc
echo "# Ember-275M Lightning AI" >> ~/.bashrc
echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
echo "export FORGE_NO_SKIP=1" >> ~/.bashrc
echo "export FORGE_DATA_DIR=$DATA_DIR" >> ~/.bashrc
echo "export NCCL_SOCKET_IFNAME=lo" >> ~/.bashrc
echo "export GLOO_SOCKET_IFNAME=lo" >> ~/.bashrc
source ~/.bashrc
echo "    Done."

echo "=== [2/5] Installing dependencies ==="
pip install uv --quiet
uv pip install -r requirements.txt --quiet
uv pip install -e "$FORGE_DIR" --quiet
echo "    Done."

echo "=== [3/5] Downloading FineWeb-Edu shards (12 x ~2GB) ==="
mkdir -p "$DATA_DIR"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    MAJOR=$((i / 10))
    MINOR=$((i % 10))
    NAME=$(printf "%03d_%05d.parquet" $MAJOR $MINOR)
    DEST="$DATA_DIR/$NAME"
    if [ -f "$DEST" ]; then
        echo "    $NAME already exists, skipping."
    else
        echo "    Downloading $NAME ..."
        wget -q --show-progress "$BASE_URL/$NAME" -O "$DEST"
    fi
done
echo "    Done."

echo "=== [4/5] Pulling latest checkpoint from HF Hub ==="
forge pull
echo "    Done."

echo "=== [5/5] Launching training in tmux ==="
tmux new-session -d -s ember "cd $EMBER_DIR && FORGE_PROFILE=lightning-a100 python train.py 2>&1 | tee /tmp/ember_train.log"
echo ""
echo "================================================"
echo " Training is running in tmux session 'ember'"
echo ""
echo " Watch live:  tmux attach -t ember"
echo " View log:    tail -f /tmp/ember_train.log"
echo " Detach:      Ctrl+b then d"
echo "================================================"
