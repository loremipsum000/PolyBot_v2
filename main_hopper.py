#!/usr/bin/env python3
"""
Unified entry point for Bundle Arbitrage Bot.

STRATEGY OVERHAUL (Dec 5, 2025):
Based on PNL_ANALYSIS_REPORT.md analysis of gabagool22's actual profitable trades.

Previous (WRONG) Strategy:
- Momentum-based directional betting
- Buy one side based on price trend
- Result: This is what gabagool22 did when LOSING money for 5+ weeks

New (CORRECT) Strategy - Bundle Arbitrage:
- Buy BOTH YES and NO for less than $1.00 combined
- Guaranteed profit = (1.00 - bundle_cost) × min(yes_shares, no_shares)
- No directional risk - market-neutral hedging
- Result: This is what made gabagool22 profitable starting Dec 5

Key Thresholds (from gabagool22's winning trades):
- Max bundle cost: $0.98 (HARD STOP)
- Target bundle cost: $0.97 (sweet spot)
- Abort threshold: $0.99 (kill switch)
- Max position imbalance: 10%
- Max YES price: $0.75
- Max NO price: $0.35
"""

import asyncio
import signal
import sys

from market_hopper import MarketHopper
from config import (
    BA_MAX_BUNDLE_COST, BA_TARGET_BUNDLE_COST, BA_ABORT_BUNDLE_COST,
    BA_MAX_IMBALANCE, BA_MAX_TOTAL_EXPOSURE,
    BA_MAX_YES_PRICE, BA_MAX_NO_PRICE,
    BA_ORDER_SIZE, BA_SWEEP_BURST_SIZE,
    ENABLE_BUNDLE_ARBITRAGE,
)


async def main(target_fills: int = 20):
    """Main entry point for the bundle arbitrage bot"""
    
    print("="*65)
    print("🐙 GABAGOOL BOT - BUNDLE ARBITRAGE MODE")
    print("="*65)
    print()
    print("Strategy: Bundle Arbitrage (gabagool22's Dec 5 profitable strategy)")
    print("Core: Buy BOTH YES and NO for less than $1.00")
    print("Profit: (1.00 - bundle_cost) × hedged_shares")
    print()
    print("="*65)
    print("BUNDLE ARBITRAGE CONFIGURATION")
    print("="*65)
    print(f"  Enabled:            {ENABLE_BUNDLE_ARBITRAGE}")
    print()
    print("Bundle Cost Thresholds:")
    print(f"  Max Bundle Cost:    ${BA_MAX_BUNDLE_COST:.2f}  (HARD STOP)")
    print(f"  Target Bundle:      ${BA_TARGET_BUNDLE_COST:.2f}  (Sweet spot)")
    print(f"  Abort Threshold:    ${BA_ABORT_BUNDLE_COST:.2f}  (Kill switch)")
    print()
    print("Position Limits:")
    print(f"  Max Imbalance:      {BA_MAX_IMBALANCE*100:.0f}%")
    print(f"  Max Exposure:       ${BA_MAX_TOTAL_EXPOSURE:.0f}")
    print()
    print("Entry Price Thresholds:")
    print(f"  Max YES Price:      ${BA_MAX_YES_PRICE:.2f}")
    print(f"  Max NO Price:       ${BA_MAX_NO_PRICE:.2f}")
    print()
    print("Execution:")
    print(f"  Order Size:         {BA_ORDER_SIZE} shares")
    print(f"  Burst Size:         {BA_SWEEP_BURST_SIZE} orders")
    print("="*65)
    print()
    
    if not ENABLE_BUNDLE_ARBITRAGE:
        print("⚠️  Bundle arbitrage is DISABLED in config!")
        print("Set ENABLE_BUNDLE_ARBITRAGE=true to enable.")
        return
    
    hopper = MarketHopper()
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\n[Hopper] Shutting down...")
        hopper.stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await hopper.run(target_fills=target_fills)
    except KeyboardInterrupt:
        print("\n[Hopper] Interrupted by user")
        hopper.stop_event.set()
    except Exception as e:
        print(f"\n[Hopper] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("[Hopper] Goodbye!")


if __name__ == "__main__":
    # Parse command line args
    target = 20  # Default target fills
    
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [target_fills]")
            print(f"  target_fills: Number of sweeps to execute (default: 20)")
            sys.exit(1)
    
    asyncio.run(main(target_fills=target))
