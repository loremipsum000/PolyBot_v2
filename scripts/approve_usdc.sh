#!/usr/bin/env bash
# One-time USDC.e approval for Polymarket CTF/Exchange.
# Usage: ./scripts/approve_usdc.sh
# Requires: .env with RPC_URL (or ALCHEMY_RPC_URL), CHAIN_ID, PRIVATE_KEY.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

python3 approve_usdc.py

