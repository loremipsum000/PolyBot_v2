#!/bin/bash

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
fi

# Configuration overrides
export MAKER_MAX_BUNDLE_COST=0.98
export MAKER_ORDER_SIZE=20
export MAKER_REFRESH_INTERVAL=2

# Run the maker strategy entrypoint
python3 strategies/maker/run_maker.py
