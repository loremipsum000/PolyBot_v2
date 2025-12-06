#!/bin/bash

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
fi

# Run the taker (bundle arbitrage) entrypoint
python3 strategies/taker/run_taker.py "$@"

