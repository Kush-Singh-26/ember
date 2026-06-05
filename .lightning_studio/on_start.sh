#!/usr/bin/env bash
# =============================================================================
# .lightning_studio/on_start.sh
# Runs automatically every time the Lightning AI Studio starts.
# NOTE: Do NOT put pip installs here — they persist on the Studio disk.
#       This file is only for things that need to run on every boot.
# =============================================================================

# Re-source env vars from bashrc (secrets set via Lightning UI are
# already injected by the platform; this catches anything in ~/.bashrc)
source ~/.bashrc 2>/dev/null || true

# Log startup confirmation
echo "[on_start.sh] Ember-275M Studio started."
echo "  FORGE_DATA_DIR : ${FORGE_DATA_DIR:-'(not set — run scripts/lightning_setup.sh)'}"
echo "  HF_TOKEN       : ${HF_TOKEN:+set (hidden)}"
