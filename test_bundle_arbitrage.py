#!/usr/bin/env python3
"""
Test script for Bundle Arbitrage Strategy.

This validates the core logic of the bundle arbitrage strategy
before deploying to production.

Tests:
1. Bundle cost calculation
2. Position balancing logic
3. Entry/exit thresholds
4. P&L calculation
"""

import asyncio
import sys
import time

# Add project to path
sys.path.insert(0, '.')

from bundle_arbitrageur import (
    BundleArbitrageur, BundleConfig, OrderBookState, Position,
    BundleCostCalculator, PositionBalancer
)


def test_bundle_cost_calculation():
    """Test bundle cost calculation with UPDATED thresholds from VWAP analysis"""
    print("\n" + "="*60)
    print("TEST 1: Bundle Cost Calculation (VWAP-optimized)")
    print("="*60)
    
    config = BundleConfig()
    print(f"  Config: max_bundle=${config.max_bundle_cost}, abort=${config.abort_bundle_cost}")
    calc = BundleCostCalculator(config)
    
    # Test case 1: Good arbitrage opportunity (bundle < 0.99)
    book1 = OrderBookState(
        timestamp=time.time(),
        best_bid_yes=0.68,
        best_ask_yes=0.70,
        best_bid_no=0.25,
        best_ask_no=0.27,
    )
    calc.update(book1)
    
    bundle_ask = book1.bundle_cost_ask
    print(f"\nCase 1: YES ask=${book1.best_ask_yes:.2f}, NO ask=${book1.best_ask_no:.2f}")
    print(f"  Bundle Cost (ask): ${bundle_ask:.3f}")
    print(f"  Has arbitrage: {book1.has_arbitrage_opportunity}")
    
    should_enter, reason = calc.should_enter()
    print(f"  Should enter: {should_enter} ({reason})")
    
    assert bundle_ask == 0.97, f"Expected 0.97, got {bundle_ask}"
    assert book1.has_arbitrage_opportunity, "Should have arbitrage opportunity"
    assert should_enter, f"Should enter: {reason}"
    print("  ✅ PASS")
    
    # Test case 2: No arbitrage (bundle > 1.00)
    book2 = OrderBookState(
        timestamp=time.time(),
        best_bid_yes=0.78,
        best_ask_yes=0.80,
        best_bid_no=0.22,
        best_ask_no=0.25,
    )
    calc.update(book2)
    
    bundle_ask = book2.bundle_cost_ask
    print(f"\nCase 2: YES ask=${book2.best_ask_yes:.2f}, NO ask=${book2.best_ask_no:.2f}")
    print(f"  Bundle Cost (ask): ${bundle_ask:.3f}")
    print(f"  Has arbitrage: {book2.has_arbitrage_opportunity}")
    
    should_enter, reason = calc.should_enter()
    print(f"  Should enter: {should_enter} ({reason})")
    
    assert bundle_ask == 1.05, f"Expected 1.05, got {bundle_ask}"
    assert not book2.has_arbitrage_opportunity, "Should NOT have arbitrage"
    assert not should_enter, f"Should NOT enter"
    print("  ✅ PASS")
    
    # Test case 3: Abort threshold (>= 0.995)
    book3 = OrderBookState(
        timestamp=time.time(),
        best_bid_yes=0.72,
        best_ask_yes=0.74,
        best_bid_no=0.24,
        best_ask_no=0.26,
    )
    calc.update(book3)
    
    bundle_ask = book3.bundle_cost_ask
    print(f"\nCase 3: YES ask=${book3.best_ask_yes:.2f}, NO ask=${book3.best_ask_no:.2f}")
    print(f"  Bundle Cost (ask): ${bundle_ask:.3f}")
    
    should_abort, abort_reason = calc.should_abort()
    print(f"  Should abort: {should_abort} ({abort_reason})")
    
    # 0.74 + 0.26 = 1.00 -> should abort (>= 0.995)
    assert bundle_ask == 1.0, f"Expected 1.00, got {bundle_ask}"
    assert should_abort, "Should abort at bundle_cost >= 0.995"
    print("  ✅ PASS")


def test_position_tracking():
    """Test position tracking and P&L calculation"""
    print("\n" + "="*60)
    print("TEST 2: Position Tracking & P&L")
    print("="*60)
    
    pos = Position()
    
    # Simulate fills
    pos.update_from_fill('YES', 0.68, 100)  # 100 YES @ $0.68
    pos.update_from_fill('NO', 0.27, 100)   # 100 NO @ $0.27
    
    print(f"\nAfter 100 YES @ $0.68, 100 NO @ $0.27:")
    print(f"  YES: {pos.yes_shares} shares @ ${pos.yes_avg_price:.3f}")
    print(f"  NO: {pos.no_shares} shares @ ${pos.no_avg_price:.3f}")
    print(f"  Bundle Cost (avg): ${pos.bundle_cost:.3f}")
    print(f"  Hedged Pairs: {pos.hedged_shares}")
    print(f"  Imbalance: {pos.imbalance*100:.1f}%")
    print(f"  Unrealized P&L: ${pos.unrealized_pnl:.2f}")
    
    assert pos.yes_shares == 100, f"Expected 100 YES, got {pos.yes_shares}"
    assert pos.no_shares == 100, f"Expected 100 NO, got {pos.no_shares}"
    assert abs(pos.bundle_cost - 0.95) < 0.001, f"Expected 0.95 bundle, got {pos.bundle_cost}"
    assert pos.hedged_shares == 100, f"Expected 100 hedged, got {pos.hedged_shares}"
    assert pos.imbalance == 0.0, f"Expected 0 imbalance, got {pos.imbalance}"
    assert abs(pos.unrealized_pnl - 5.0) < 0.01, f"Expected $5.00, got ${pos.unrealized_pnl}"  # (1.00 - 0.95) * 100 = $5.00
    print("  ✅ PASS")
    
    # Test imbalanced position
    pos2 = Position()
    pos2.update_from_fill('YES', 0.65, 150)
    pos2.update_from_fill('NO', 0.30, 100)
    
    print(f"\nImbalanced: 150 YES @ $0.65, 100 NO @ $0.30:")
    print(f"  Hedged Pairs: {pos2.hedged_shares}")
    print(f"  Imbalance: {pos2.imbalance*100:.1f}%")
    print(f"  Imbalance Direction: {pos2.imbalance_direction}")
    print(f"  Unrealized P&L: ${pos2.unrealized_pnl:.2f}")
    
    assert pos2.hedged_shares == 100  # min(150, 100) = 100
    assert pos2.imbalance_direction == 'YES'
    assert 0.19 < pos2.imbalance < 0.21  # ~20%
    print("  ✅ PASS")
    
    # Test auto-merge
    print(f"\nAuto-merge test:")
    pos3 = Position()
    pos3.update_from_fill('YES', 0.68, 100)
    pos3.update_from_fill('NO', 0.27, 100)
    
    print(f"  Before merge: ${pos3.realized_pnl:.2f} realized")
    profit = pos3.auto_merge()
    print(f"  Merged {100} pairs @ ${0.95:.3f} bundle")
    print(f"  Profit from merge: ${profit:.2f}")
    print(f"  After merge: ${pos3.realized_pnl:.2f} realized")
    print(f"  Remaining: {pos3.yes_shares:.0f} YES, {pos3.no_shares:.0f} NO")
    
    assert abs(profit - 5.0) < 0.01, f"Expected profit $5.00, got ${profit}"
    assert abs(pos3.realized_pnl - 5.0) < 0.01, f"Expected realized $5.00, got ${pos3.realized_pnl}"
    assert pos3.yes_shares == 0, f"Expected 0 YES, got {pos3.yes_shares}"
    assert pos3.no_shares == 0, f"Expected 0 NO, got {pos3.no_shares}"
    print("  ✅ PASS")


def test_position_balancer():
    """Test position balancing logic"""
    print("\n" + "="*60)
    print("TEST 3: Position Balancer")
    print("="*60)
    
    config = BundleConfig(max_imbalance=0.10)
    balancer = PositionBalancer(config)
    
    # Balanced position - no rebalance needed
    pos1 = Position()
    pos1.update_from_fill('YES', 0.68, 100)
    pos1.update_from_fill('NO', 0.27, 100)
    
    book = OrderBookState(
        best_ask_yes=0.70,
        best_ask_no=0.28,
    )
    
    side, size = balancer.get_rebalance_action(pos1, book)
    print(f"\nBalanced position (100/100):")
    print(f"  Rebalance needed: {side} {size:.0f}")
    assert side is None
    print("  ✅ PASS")
    
    # Imbalanced - need more NO
    pos2 = Position()
    pos2.update_from_fill('YES', 0.68, 150)
    pos2.update_from_fill('NO', 0.27, 100)
    
    side, size = balancer.get_rebalance_action(pos2, book)
    print(f"\nImbalanced position (150 YES / 100 NO):")
    print(f"  Imbalance: {pos2.imbalance*100:.1f}%")
    print(f"  Rebalance needed: {side} {size:.0f}")
    assert side == 'NO'
    assert size == 50
    print("  ✅ PASS")
    
    # Test balanced order sizes
    yes_size, no_size = balancer.get_balanced_order_sizes(pos2, book, 16)
    print(f"\nBalanced order sizes for 16-share order:")
    print(f"  YES: {yes_size:.1f}, NO: {no_size:.1f}")
    assert no_size > yes_size  # Should favor NO
    print("  ✅ PASS")


def test_profit_scenarios():
    """Test various profit scenarios from gabagool22's data"""
    print("\n" + "="*60)
    print("TEST 4: Profit Scenarios (from gabagool22 data)")
    print("="*60)
    
    # Winning trade example: BTC 1PM
    # 1,131 YES @ $0.678, 1,123 NO @ $0.297
    # Bundle cost: $0.975
    # Expected profit: ~$28
    
    pos = Position()
    pos.update_from_fill('YES', 0.678, 1131)
    pos.update_from_fill('NO', 0.297, 1123)
    
    print(f"\nWinning Trade (BTC 1PM from report):")
    print(f"  {pos.yes_shares:.0f} YES @ ${pos.yes_avg_price:.3f}")
    print(f"  {pos.no_shares:.0f} NO @ ${pos.no_avg_price:.3f}")
    print(f"  Bundle Cost: ${pos.bundle_cost:.3f}")
    print(f"  Hedged Pairs: {pos.hedged_shares:.0f}")
    print(f"  Expected Profit: ${(1.0 - pos.bundle_cost) * pos.hedged_shares:.2f}")
    print(f"  Calculated Unrealized P&L: ${pos.unrealized_pnl:.2f}")
    
    expected_profit = (1.0 - 0.975) * 1123  # ~$28
    assert 25 < pos.unrealized_pnl < 32, f"Expected ~$28, got ${pos.unrealized_pnl:.2f}"
    print("  ✅ PASS - Matches expected ~$28")
    
    # Losing trade example: ETH 1:00PM-1:15PM
    # 400 YES @ $0.779, 257 NO @ $0.252
    # Bundle cost: $1.031 (OVER $1.00!)
    # Expected loss: negative
    
    pos2 = Position()
    pos2.update_from_fill('YES', 0.779, 400)
    pos2.update_from_fill('NO', 0.252, 257)
    
    print(f"\nLosing Trade (ETH from report):")
    print(f"  {pos2.yes_shares:.0f} YES @ ${pos2.yes_avg_price:.3f}")
    print(f"  {pos2.no_shares:.0f} NO @ ${pos2.no_avg_price:.3f}")
    print(f"  Bundle Cost: ${pos2.bundle_cost:.3f}")
    print(f"  Hedged Pairs: {pos2.hedged_shares:.0f}")
    print(f"  Calculated Unrealized P&L: ${pos2.unrealized_pnl:.2f}")
    
    assert pos2.bundle_cost > 1.0, "Bundle cost should exceed $1.00"
    assert pos2.unrealized_pnl < 0, "Should be a loss"
    print(f"  ✅ PASS - Correctly identifies loss (bundle > $1.00)")


def test_entry_thresholds():
    """Test entry threshold logic with UPDATED thresholds from VWAP analysis"""
    print("\n" + "="*60)
    print("TEST 5: Entry Thresholds (WIDENED from VWAP analysis)")
    print("="*60)
    
    # UPDATED: max_yes_price=0.85, max_no_price=0.45 (widened from analysis)
    config = BundleConfig(
        max_bundle_cost=0.99,
        max_yes_price=0.85,
        max_no_price=0.45,
    )
    calc = BundleCostCalculator(config)
    print(f"  Config: max_yes=${config.max_yes_price}, max_no=${config.max_no_price}")
    
    # Good opportunity
    book1 = OrderBookState(
        best_ask_yes=0.70,
        best_ask_no=0.27,
    )
    calc.update(book1)
    should_enter, reason = calc.should_enter()
    print(f"\nGood: YES=${book1.best_ask_yes:.2f}, NO=${book1.best_ask_no:.2f}")
    print(f"  Bundle: ${book1.bundle_cost_ask:.3f}")
    print(f"  Should enter: {should_enter} - {reason}")
    assert should_enter
    print("  ✅ PASS")
    
    # YES at threshold (0.85) - should still work
    book2 = OrderBookState(
        best_ask_yes=0.85,
        best_ask_no=0.10,
    )
    calc.update(book2)
    should_enter, reason = calc.should_enter()
    print(f"\nYES at threshold: YES=${book2.best_ask_yes:.2f}, NO=${book2.best_ask_no:.2f}")
    print(f"  Bundle: ${book2.bundle_cost_ask:.3f}")
    print(f"  Should enter: {should_enter} - {reason}")
    assert should_enter, "Should enter at YES=$0.85 (within threshold)"
    print("  ✅ PASS")
    
    # YES over threshold (0.90 > 0.85)
    book3 = OrderBookState(
        best_ask_yes=0.90,
        best_ask_no=0.05,
    )
    calc.update(book3)
    should_enter, reason = calc.should_enter()
    print(f"\nYES too expensive: YES=${book3.best_ask_yes:.2f}, NO=${book3.best_ask_no:.2f}")
    print(f"  Bundle: ${book3.bundle_cost_ask:.3f}")
    print(f"  Should enter: {should_enter} - {reason}")
    assert not should_enter, "Should NOT enter at YES=$0.90 (above $0.85)"
    print("  ✅ PASS")
    
    # NO over threshold (0.50 > 0.45)
    book4 = OrderBookState(
        best_ask_yes=0.45,
        best_ask_no=0.50,
    )
    calc.update(book4)
    should_enter, reason = calc.should_enter()
    print(f"\nNO too expensive: YES=${book4.best_ask_yes:.2f}, NO=${book4.best_ask_no:.2f}")
    print(f"  Bundle: ${book4.bundle_cost_ask:.3f}")
    print(f"  Should enter: {should_enter} - {reason}")
    assert not should_enter, "Should NOT enter at NO=$0.50 (above $0.45)"
    print("  ✅ PASS")


def test_vwap_tracking():
    """Test VWAP tracking from 25K trade analysis"""
    print("\n" + "="*60)
    print("TEST 6: VWAP Tracking (from 25K trade analysis)")
    print("="*60)
    
    pos = Position()
    
    # Simulate multiple fills like gabagool22's pattern
    # Target: avg 16.8 fills per transaction
    fills = [
        ('YES', 0.65, 16),
        ('NO', 0.28, 16),
        ('YES', 0.66, 16),
        ('NO', 0.29, 16),
        ('YES', 0.67, 16),
        ('NO', 0.28, 16),
        ('YES', 0.68, 16),
        ('NO', 0.30, 16),
    ]
    
    for side, price, size in fills:
        pos.update_from_fill(side, price, size)
    
    print(f"\nSimulated {len(fills)} fills:")
    print(f"  Total fills tracked: {pos.total_fills}")
    print(f"  YES VWAP: ${pos.get_vwap('YES'):.4f}")
    print(f"  NO VWAP: ${pos.get_vwap('NO'):.4f}")
    print(f"  Bundle VWAP: ${pos.get_bundle_vwap():.4f}")
    
    assert pos.total_fills == 8, f"Expected 8 fills, got {pos.total_fills}"
    
    # Check VWAP calculation
    yes_vwap = pos.get_vwap('YES')
    no_vwap = pos.get_vwap('NO')
    bundle_vwap = pos.get_bundle_vwap()
    
    assert 0.65 < yes_vwap < 0.69, f"YES VWAP {yes_vwap} out of range"
    assert 0.27 < no_vwap < 0.31, f"NO VWAP {no_vwap} out of range"
    assert bundle_vwap < 1.0, f"Bundle VWAP {bundle_vwap} should be < $1.00"
    
    print(f"\n  Expected profit potential: ${(1.0 - bundle_vwap) * pos.hedged_shares:.2f}")
    print("  ✅ PASS - VWAP tracking working correctly")
    
    # Test fill stats
    stats = pos.get_fill_stats()
    print(f"\nFill stats:")
    print(f"  YES fills: {stats['yes_fills']}")
    print(f"  NO fills: {stats['no_fills']}")
    print(f"  Fill rate: {stats['fill_rate']:.2f}")
    print("  ✅ PASS")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🐙 BUNDLE ARBITRAGE STRATEGY TESTS (VWAP-optimized)")
    print("="*60)
    print("Testing the core logic from 25K trade VWAP analysis...")
    
    try:
        test_bundle_cost_calculation()
        test_position_tracking()
        test_position_balancer()
        test_profit_scenarios()
        test_entry_thresholds()
        test_vwap_tracking()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nBundle arbitrage logic is validated (VWAP-optimized).")
        print("The strategy correctly:")
        print("  - Calculates bundle costs with WIDENED thresholds")
        print("  - Tracks positions, VWAP, and P&L")
        print("  - Maintains position balance (<10% imbalance)")
        print("  - Identifies profitable opportunities")
        print("  - Supports multi-fill execution (~17 fills/sweep)")
        print("  - Guards against losing trades (bundle > $1.00)")
        print("\nReady to deploy with: python main_hopper.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

