"""
Bundle Arbitrageur - Gabagool22's Actual Profitable Strategy

This module implements the bundle arbitrage strategy discovered from analyzing
gabagool22's profitable trades starting Dec 5, 2025.

Core Insight:
- Buy BOTH YES and NO shares for less than $1.00 combined
- Guaranteed profit = (1.00 - bundle_cost) × min(yes_shares, no_shares)
- No directional risk - market-neutral hedging

Key Differences from Our Previous Approach:
- OLD: Momentum-based directional betting (one side)
- NEW: Simultaneous dual-side buying (both sides)
- OLD: Risk = full directional exposure
- NEW: Risk = only if bundle_cost > $1.00

References:
- docs/PNL_ANALYSIS_REPORT.md - Full gabagool22 analysis
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from py_clob_client.constants import POLYGON

from config import (
    PRIVATE_KEY, POLYGON_ADDRESS,
    ENABLE_ONCHAIN_MERGE, ENABLE_ONCHAIN_REDEEM,
)
from chain_client import CTFSettlementClient
from trade_logger import TradeLogger


# =============================================================================
# BUNDLE ARBITRAGE CONFIGURATION
# Updated from 25,000 trade VWAP analysis - PNL_ANALYSIS_REPORT.md
# =============================================================================

@dataclass
class BundleConfig:
    """Configuration for bundle arbitrage strategy"""
    
    # Bundle Cost Thresholds (UPDATED based on actual VWAP data)
    # Gabagool22's avg bundle VWAP is $0.987, so we widened thresholds
    max_bundle_cost: float = 0.99       # Was 0.98, avg VWAP is 0.987
    target_bundle_cost: float = 0.985   # Was 0.97, realistic target
    abort_bundle_cost: float = 0.995    # Tighter - losses happen at >1.00
    
    # Position Balance (winners avg 7.6% imbalance, losers 28.3%)
    max_imbalance: float = 0.10         # Max 10% difference between YES/NO
    min_position_size: float = 100      # Don't bother with tiny positions
    max_total_exposure: float = 2000    # Max total capital at risk
    
    # Entry Price Thresholds (WIDENED based on actual price ranges)
    # Gabagool22 buys Up $0.09-$0.91, Down $0.11-$0.89
    max_yes_price: float = 0.92         # He pays up to ~0.91 on strong side
    max_no_price: float = 0.30          # Require cheap hedge leg
    cheap_side_threshold: float = 0.30  # Require at least one side at/under this
    
    # Execution (avg 16.8 fills per transaction - AGGRESSIVE multi-fill)
    order_size: float = 16              # Confirmed 16 shares standard
    sweep_burst_size: int = 20          # Was 50, per-burst wave
    sweep_interval_ms: float = 5        # Was 10ms, faster execution
    min_order_value: float = 1.0        # Minimum $1 order value
    fills_per_sweep: int = 17           # Target ~16.8 fills (from VWAP analysis)
    merge_min_hedged: float = 500       # Minimum hedged shares before on-chain merge
    merge_min_unrealized: float = 10.0  # Minimum unrealized PnL ($) before on-chain merge
    
    # Time-based scoring (early morning has better arbitrage)
    morning_bonus_start: int = 9        # UTC hour
    morning_bonus_end: int = 11         # UTC hour
    morning_bonus_mult: float = 1.1     # 10% score bonus


# =============================================================================
# POSITION TRACKER (with VWAP tracking)
# =============================================================================

@dataclass
class FillRecord:
    """Single fill record for VWAP calculation"""
    timestamp: float
    side: str  # 'YES' or 'NO'
    price: float
    size: float


@dataclass
class Position:
    """
    Tracks current position in a market with VWAP tracking.
    
    From 25K trade analysis:
    - Avg Bundle VWAP: $0.987
    - VWAP is more accurate than simple average for execution quality
    """
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_cost: float = 0.0
    no_cost: float = 0.0
    
    # Running averages (VWAP)
    yes_avg_price: float = 0.0
    no_avg_price: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    
    # Fill history for VWAP analysis
    fills: List = field(default_factory=list)
    total_fills: int = 0
    
    def update_from_fill(self, side: str, price: float, size: float):
        """Update position after a fill with VWAP tracking"""
        # Record fill for analysis
        self.fills.append(FillRecord(
            timestamp=time.time(),
            side=side.upper(),
            price=price,
            size=size
        ))
        self.total_fills += 1
        
        if side.upper() == 'YES':
            self.yes_shares += size
            self.yes_cost += price * size
            self.yes_avg_price = self.yes_cost / self.yes_shares if self.yes_shares > 0 else 0
        else:  # NO
            self.no_shares += size
            self.no_cost += price * size
            self.no_avg_price = self.no_cost / self.no_shares if self.no_shares > 0 else 0
    
    def get_vwap(self, side: str = None) -> float:
        """
        Calculate VWAP for a side or combined bundle.
        Returns volume-weighted average price.
        """
        if not self.fills:
            return 0.0
        
        if side:
            fills = [f for f in self.fills if f.side == side.upper()]
        else:
            fills = self.fills
        
        if not fills:
            return 0.0
        
        total_value = sum(f.price * f.size for f in fills)
        total_volume = sum(f.size for f in fills)
        
        return total_value / total_volume if total_volume > 0 else 0.0
    
    def get_bundle_vwap(self) -> float:
        """Get combined bundle VWAP (YES VWAP + NO VWAP)"""
        yes_vwap = self.get_vwap('YES')
        no_vwap = self.get_vwap('NO')
        return yes_vwap + no_vwap
    
    def get_fill_stats(self) -> Dict:
        """Get fill statistics for analysis"""
        if not self.fills:
            return {'total_fills': 0}
        
        yes_fills = [f for f in self.fills if f.side == 'YES']
        no_fills = [f for f in self.fills if f.side == 'NO']
        
        return {
            'total_fills': self.total_fills,
            'yes_fills': len(yes_fills),
            'no_fills': len(no_fills),
            'yes_vwap': self.get_vwap('YES'),
            'no_vwap': self.get_vwap('NO'),
            'bundle_vwap': self.get_bundle_vwap(),
            'fill_rate': len(yes_fills) / len(no_fills) if no_fills else 0,
        }
    
    @property
    def bundle_cost(self) -> float:
        """Current average bundle cost"""
        return self.yes_avg_price + self.no_avg_price
    
    @property
    def hedged_shares(self) -> float:
        """Number of fully hedged share pairs"""
        return min(self.yes_shares, self.no_shares)
    
    @property
    def imbalance(self) -> float:
        """Position imbalance as percentage (0.0 to 1.0)"""
        if self.yes_shares == 0 and self.no_shares == 0:
            return 0.0
        total = self.yes_shares + self.no_shares
        diff = abs(self.yes_shares - self.no_shares)
        return diff / total if total > 0 else 0.0
    
    @property
    def imbalance_direction(self) -> str:
        """Which side has more shares"""
        if self.yes_shares > self.no_shares:
            return 'YES'
        elif self.no_shares > self.yes_shares:
            return 'NO'
        return 'BALANCED'
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L if we merged all hedged pairs at $1.00"""
        if self.hedged_shares <= 0:
            return 0.0
        
        # Profit per hedged pair = $1.00 - bundle_cost
        if self.bundle_cost < 1.0:
            return (1.0 - self.bundle_cost) * self.hedged_shares
        else:
            # Loss if bundle cost > $1.00
            return (1.0 - self.bundle_cost) * self.hedged_shares
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_exposure(self) -> float:
        """Total capital at risk"""
        return self.yes_cost + self.no_cost
    
    def auto_merge(self) -> float:
        """
        Auto-merge hedged pairs and realize P&L.
        Returns the realized profit from this merge.
        """
        if self.hedged_shares <= 0 or self.bundle_cost >= 1.0:
            return 0.0
        
        shares_to_merge = self.hedged_shares
        profit_per_pair = 1.0 - self.bundle_cost
        realized = profit_per_pair * shares_to_merge
        
        # Reduce positions
        self.yes_shares -= shares_to_merge
        self.no_shares -= shares_to_merge
        self.yes_cost -= self.yes_avg_price * shares_to_merge
        self.no_cost -= self.no_avg_price * shares_to_merge
        self.realized_pnl += realized
        
        # Recalculate averages (should be close to 0 after full merge)
        self.yes_avg_price = self.yes_cost / self.yes_shares if self.yes_shares > 0 else 0
        self.no_avg_price = self.no_cost / self.no_shares if self.no_shares > 0 else 0
        
        return realized


# =============================================================================
# BUNDLE COST CALCULATOR
# =============================================================================

@dataclass
class OrderBookState:
    """Snapshot of orderbook state"""
    timestamp: float = 0.0
    
    # YES side
    best_bid_yes: float = 0.0
    best_ask_yes: float = 1.0
    best_bid_yes_size: float = 0.0
    best_ask_yes_size: float = 0.0
    depth_yes_bids: int = 0
    depth_yes_asks: int = 0
    
    # NO side  
    best_bid_no: float = 0.0
    best_ask_no: float = 1.0
    best_bid_no_size: float = 0.0
    best_ask_no_size: float = 0.0
    depth_no_bids: int = 0
    depth_no_asks: int = 0
    
    @property
    def bundle_cost_ask(self) -> float:
        """Bundle cost if we buy at ask prices (taker)"""
        return self.best_ask_yes + self.best_ask_no
    
    @property
    def bundle_cost_bid(self) -> float:
        """Bundle cost using bid prices (for P&L estimation)"""
        return self.best_bid_yes + self.best_bid_no
    
    @property
    def spread_yes(self) -> float:
        return self.best_ask_yes - self.best_bid_yes
    
    @property
    def spread_no(self) -> float:
        return self.best_ask_no - self.best_bid_no
    
    @property
    def has_liquidity(self) -> bool:
        """Check if market has tradeable liquidity"""
        yes_ok = 0.02 < self.best_ask_yes < 0.98
        no_ok = 0.02 < self.best_ask_no < 0.98
        return yes_ok or no_ok
    
    @property
    def has_arbitrage_opportunity(self) -> bool:
        """Check if bundle arbitrage is possible"""
        return self.bundle_cost_ask < 0.98


class BundleCostCalculator:
    """
    Real-time bundle cost tracking and opportunity detection.
    
    Key insight from gabagool22:
    - PROFITABLE when: bundle_cost < $1.00
    - SAFE entry when: bundle_cost < $0.98
    - ABORT when: bundle_cost > $0.99
    """
    
    def __init__(self, config: BundleConfig):
        self.config = config
        self.history: deque = deque(maxlen=100)
        self.current: Optional[OrderBookState] = None
        
    def update(self, book_state: OrderBookState):
        """Update with new orderbook data"""
        self.current = book_state
        self.history.append(book_state)
    
    def should_enter(self) -> Tuple[bool, str]:
        """
        Determine if we should enter a position.
        Returns (should_enter, reason)
        """
        if not self.current:
            return False, "No market data"
        
        bc = self.current.bundle_cost_ask
        
        if bc >= self.config.abort_bundle_cost:
            return False, f"Bundle cost ${bc:.3f} >= abort threshold ${self.config.abort_bundle_cost}"
        
        if bc >= self.config.max_bundle_cost:
            return False, f"Bundle cost ${bc:.3f} >= max ${self.config.max_bundle_cost}"
        
        if not self.current.has_liquidity:
            return False, "No liquidity"
        
        # Require at least one cheap side to mirror gabagool2's pattern
        cheap_side = min(self.current.best_ask_yes, self.current.best_ask_no)
        if cheap_side > self.config.cheap_side_threshold:
            return False, f"Cheapest side ${cheap_side:.3f} > cheap threshold ${self.config.cheap_side_threshold}"
        
        # Check individual price thresholds
        if self.current.best_ask_yes > self.config.max_yes_price:
            return False, f"YES ask ${self.current.best_ask_yes:.3f} > max ${self.config.max_yes_price}"
        
        if self.current.best_ask_no > self.config.max_no_price:
            return False, f"NO ask ${self.current.best_ask_no:.3f} > max ${self.config.max_no_price}"
        
        return True, f"Bundle cost ${bc:.3f} - OPPORTUNITY!"
    
    def should_abort(self) -> Tuple[bool, str]:
        """Check if we should stop accumulating"""
        if not self.current:
            return True, "No market data"
        
        bc = self.current.bundle_cost_ask
        if bc >= self.config.abort_bundle_cost:
            return True, f"Bundle cost ${bc:.3f} >= abort ${self.config.abort_bundle_cost}"
        
        return False, ""


# =============================================================================
# POSITION BALANCER
# =============================================================================

class PositionBalancer:
    """
    Ensures YES and NO positions stay balanced.
    
    From gabagool22 analysis:
    - Winning trades: avg imbalance 7.6%
    - Losing trades: avg imbalance 28.3%
    - Target: < 10% imbalance
    """
    
    def __init__(self, config: BundleConfig):
        self.config = config
    
    def get_rebalance_action(self, position: Position, book: OrderBookState) -> Tuple[Optional[str], float]:
        """
        Determine which side to buy to maintain balance.
        Returns (side_to_buy, max_size) or (None, 0) if balanced.
        """
        if position.imbalance <= self.config.max_imbalance:
            return None, 0
        
        # Calculate how many shares needed to rebalance
        if position.yes_shares > position.no_shares:
            # Need more NO
            needed = position.yes_shares - position.no_shares
            side = 'NO'
            price = book.best_ask_no
            max_price = self.config.max_no_price
        else:
            # Need more YES
            needed = position.no_shares - position.yes_shares
            side = 'YES'
            price = book.best_ask_yes
            max_price = self.config.max_yes_price
        
        # Don't rebalance if price is too high
        if price > max_price:
            return None, 0
        
        return side, needed
    
    def get_balanced_order_sizes(self, position: Position, book: OrderBookState, base_size: float) -> Tuple[float, float]:
        """
        Calculate order sizes that maintain balance.
        Returns (yes_size, no_size)
        """
        # Start with equal sizes
        yes_size = base_size
        no_size = base_size
        
        # Adjust based on current imbalance
        if position.yes_shares > position.no_shares:
            # Favor NO to catch up
            ratio = position.no_shares / position.yes_shares if position.yes_shares > 0 else 1.0
            if ratio < 0.9:  # More than 10% behind
                yes_size = base_size * 0.5
                no_size = base_size * 1.5
        elif position.no_shares > position.yes_shares:
            # Favor YES to catch up
            ratio = position.yes_shares / position.no_shares if position.no_shares > 0 else 1.0
            if ratio < 0.9:
                yes_size = base_size * 1.5
                no_size = base_size * 0.5
        
        return yes_size, no_size


# =============================================================================
# DUAL SIDE SWEEPER (Core Execution Engine)
# Updated for multi-fill execution (avg 16.8 fills per transaction)
# =============================================================================

class DualSideSweeper:
    """
    Executes bundle arbitrage by sweeping BOTH sides of the orderbook.
    
    UPDATED from 25K trade VWAP analysis:
    - Avg 16.8 fills per transaction
    - Multi-fill execution pattern
    - Aggressive sweeping of both order books
    
    Key insight: Gabagool22 places ~17 orders per sweep wave,
    alternating YES/NO to maintain balance.
    """
    
    def __init__(self, clob_client: ClobClient, config: BundleConfig, logger: TradeLogger):
        self.client = clob_client
        self.config = config
        self.logger = logger
        
        # Rate limiting
        self.last_order_time: float = 0.0
        self.orders_in_burst: int = 0
        self.burst_start_time: float = 0.0
        
        # Stats
        self.total_orders_placed: int = 0
        self.total_fills: int = 0
        self.yes_fills: int = 0
        self.no_fills: int = 0
        
        # Multi-fill tracking
        self.sweep_fill_counts: List[int] = []  # Track fills per sweep
    
    def _can_place_order(self) -> bool:
        """Check burst rate limits"""
        now = time.time()
        
        # Reset burst counter after 1 second
        if now - self.burst_start_time > 1.0:
            self.orders_in_burst = 0
            self.burst_start_time = now
        
        if self.orders_in_burst >= self.config.sweep_burst_size:
            return False
        
        ms_since_last = (now - self.last_order_time) * 1000
        if ms_since_last < self.config.sweep_interval_ms:
            return False
        
        return True
    
    async def place_order(
        self,
        token_id: str,
        side: str,  # 'YES' or 'NO'
        price: float,
        size: float,
        market_slug: str = "",
        condition_id: str = "",
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Place a FOK order on one side.
        Returns (order_id, status, filled_size)
        """
        if not self._can_place_order():
            return None, None, 0.0
        
        # Ensure minimum order value
        if price * size < self.config.min_order_value:
            size = max(5, int(self.config.min_order_value / price) + 1)
        
        try:
            def _create_and_post():
                order = self.client.create_order(OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=float(size),
                    side=BUY,
                ))
                return self.client.post_order(order, OrderType.IOC)
            
            resp = await asyncio.to_thread(_create_and_post)
            
            # Update rate limiting
            self.last_order_time = time.time()
            self.orders_in_burst += 1
            self.total_orders_placed += 1
            
            order_id = resp.get('orderID') or resp.get('id')
            status = resp.get('status', 'unknown')
            
            if order_id and status == 'matched':
                self.total_fills += 1
                if side == 'YES':
                    self.yes_fills += 1
                else:
                    self.no_fills += 1
                
                # Log the fill
                outcome = 'Up' if side == 'YES' else 'Down'
                self.logger.log_order_placed(
                    market_slug=market_slug,
                    condition_id=condition_id,
                    token_id=token_id,
                    outcome=outcome,
                    side='BUY',
                    price=price,
                    size=float(size),
                    order_id=order_id,
                    status=status,
                    market_context={'bundle_arbitrage': True, 'multi_fill': True},
                )
                self.logger.update_position_from_fill(outcome, 'BUY', price, float(size))
                
                return order_id, status, size
            
            return order_id, status, 0.0
            
        except Exception as e:
            err = str(e)
            if "min" not in err.lower() and "403" not in err:
                print(f"[Sweeper] Order error: {e}")
            return None, None, 0.0
    
    async def sweep_both_sides(
        self,
        yes_token: str,
        no_token: str,
        book: OrderBookState,
        position: Position,
        balancer: PositionBalancer,
        market_slug: str = "",
        condition_id: str = "",
    ) -> Tuple[int, int, float, float]:
        """
        Execute MULTI-FILL sweep on BOTH YES and NO sides.
        
        UPDATED execution pattern (from 25K trade analysis):
        - Target ~17 fills per sweep (avg 16.8 from gabagool22)
        - Alternate YES/NO orders to maintain balance
        - Use smaller order sizes for better fill rates
        
        Returns (yes_fills, no_fills, yes_volume, no_volume)
        """
        # Price checks
        yes_price = min(book.best_ask_yes, self.config.max_yes_price)
        no_price = min(book.best_ask_no, self.config.max_no_price)
        
        # Skip if prices too expensive
        place_yes = yes_price <= self.config.max_yes_price
        place_no = no_price <= self.config.max_no_price
        
        if not place_yes and not place_no:
            return 0, 0, 0.0, 0.0
        
        # Multi-fill execution: target ~17 fills per sweep
        # Split total order size across multiple smaller orders
        target_fills = self.config.fills_per_sweep  # 17
        order_size = self.config.order_size  # 16 shares
        
        # Calculate size per fill to achieve ~17 fills while maintaining volume
        size_per_order = max(1, order_size)  # Match observed 16-share clips
        
        # Get balanced sizes based on current position
        base_yes, base_no = balancer.get_balanced_order_sizes(
            position, book, size_per_order
        )
        
        # Build alternating order list (YES, NO, YES, NO, ...)
        # This mimics gabagool22's balanced execution
        orders_to_place = []
        
        # Alternate between YES and NO, respecting balance
        yes_count = 0
        no_count = 0
        max_per_side = target_fills // 2 + 1
        yes_top_liquidity = book.best_ask_yes_size if book.best_ask_yes_size > 0 else float("inf")
        no_top_liquidity = book.best_ask_no_size if book.best_ask_no_size > 0 else float("inf")
        yes_limit = max_per_side if yes_top_liquidity == float("inf") else int(yes_top_liquidity // base_yes)
        no_limit = max_per_side if no_top_liquidity == float("inf") else int(no_top_liquidity // base_no)
        yes_limit = max(0, min(max_per_side, yes_limit))
        no_limit = max(0, min(max_per_side, no_limit))
        
        for i in range(target_fills):
            # Decide which side based on balance
            if position.yes_shares > position.no_shares * 1.1:
                # Favor NO to catch up
                if place_no and no_count < no_limit:
                    orders_to_place.append(('NO', no_price, base_no))
                    no_count += 1
                elif place_yes and yes_count < yes_limit:
                    orders_to_place.append(('YES', yes_price, base_yes))
                    yes_count += 1
            elif position.no_shares > position.yes_shares * 1.1:
                # Favor YES to catch up
                if place_yes and yes_count < yes_limit:
                    orders_to_place.append(('YES', yes_price, base_yes))
                    yes_count += 1
                elif place_no and no_count < no_limit:
                    orders_to_place.append(('NO', no_price, base_no))
                    no_count += 1
            else:
                # Balanced - alternate
                if i % 2 == 0 and place_yes and yes_count < yes_limit:
                    orders_to_place.append(('YES', yes_price, base_yes))
                    yes_count += 1
                elif place_no and no_count < no_limit:
                    orders_to_place.append(('NO', no_price, base_no))
                    no_count += 1
        
        if not orders_to_place:
            return 0, 0, 0.0, 0.0
        
        # Execute all orders concurrently for speed
        async def _execute_order(side, price, size):
            token_id = yes_token if side == 'YES' else no_token
            return (side, await self.place_order(
                token_id, side, price, size, market_slug, condition_id
            ))
        
        tasks = [_execute_order(side, price, size) for side, price, size in orders_to_place]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        yes_fills_count = 0
        no_fills_count = 0
        yes_volume = 0.0
        no_volume = 0.0
        
        for result in results:
            if isinstance(result, Exception):
                continue
            
            side, (order_id, status, filled_size) = result
            if status == 'matched' and filled_size > 0:
                if side == 'YES':
                    yes_fills_count += 1
                    yes_volume += filled_size
                    price = yes_price
                else:
                    no_fills_count += 1
                    no_volume += filled_size
                    price = no_price
                
                # Update position
                position.update_from_fill(side, price, filled_size)
        
        # Track fills per sweep for analysis
        total_fills = yes_fills_count + no_fills_count
        self.sweep_fill_counts.append(total_fills)
        
        if total_fills > 0:
            avg_fills = sum(self.sweep_fill_counts) / len(self.sweep_fill_counts)
            print(f"[Sweeper] Multi-fill: {total_fills} fills this sweep (avg: {avg_fills:.1f})")
        
        return yes_fills_count, no_fills_count, yes_volume, no_volume


# =============================================================================
# MAIN BUNDLE ARBITRAGEUR CLASS
# =============================================================================

class BundleArbitrageur:
    """
    Main orchestrator for bundle arbitrage strategy.
    
    Implements gabagool22's profitable strategy discovered Dec 5, 2025:
    1. Monitor orderbooks for both YES and NO
    2. Calculate real-time bundle cost
    3. When bundle_cost < 0.98: sweep BOTH sides
    4. Maintain position balance (< 10% imbalance)
    5. Abort if bundle_cost > 0.99
    6. Auto-merge hedged pairs for profit
    """
    
    def __init__(self, config: Optional[BundleConfig] = None):
        self.config = config or BundleConfig()
        
        # Components
        self.calculator = BundleCostCalculator(self.config)
        self.balancer = PositionBalancer(self.config)
        self.position = Position()
        self.sweeper: Optional[DualSideSweeper] = None
        self.settlement_client: Optional[CTFSettlementClient] = None
        
        # CLOB client
        self.clob_client: Optional[ClobClient] = None
        
        # Logging
        self.logger = TradeLogger(log_dir="data/bundle_logs")
        
        # Market state
        self.yes_token: str = ""
        self.no_token: str = ""
        self.market_slug: str = ""
        self.condition_id: str = ""
        
        # Stats
        self.sweeps_executed: int = 0
        self.sweeps_skipped: int = 0
        self.last_log_time: float = 0.0
        
    def init_clob_client(self):
        """Initialize CLOB client"""
        if self.clob_client:
            return
        
        print("[BundleArb] 🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=2,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[BundleArb] ✅ CLOB client authenticated")
        
        # Initialize sweeper
        self.sweeper = DualSideSweeper(self.clob_client, self.config, self.logger)
        if (ENABLE_ONCHAIN_MERGE or ENABLE_ONCHAIN_REDEEM) and not self.settlement_client:
            self.settlement_client = CTFSettlementClient()
    
    def set_market(self, yes_token: str, no_token: str, market_slug: str = "", condition_id: str = ""):
        """Set the market tokens to trade"""
        self.yes_token = yes_token
        self.no_token = no_token
        self.market_slug = market_slug
        self.condition_id = condition_id
        print(f"[BundleArb] 🎯 Market set: {market_slug[:40]}...")
        print(f"[BundleArb]    YES: {yes_token[:20]}...")
        print(f"[BundleArb]    NO:  {no_token[:20]}...")
    
    def update_book(self, book: OrderBookState):
        """Update orderbook state"""
        self.calculator.update(book)
    
    async def execute_sweep(self) -> bool:
        """
        Execute one sweep cycle of the bundle arbitrage strategy.
        
        Returns True if any fills occurred.
        """
        if not self.clob_client or not self.sweeper:
            self.init_clob_client()
        
        if not self.yes_token or not self.no_token:
            return False
        
        book = self.calculator.current
        if not book:
            return False
        
        # Check if we should enter
        should_enter, reason = self.calculator.should_enter()
        
        # Also check abort condition
        should_abort, abort_reason = self.calculator.should_abort()
        if should_abort:
            if self.sweeps_executed > 0:  # Only log if we were trading
                print(f"[BundleArb] 🛑 ABORT: {abort_reason}")
            return False
        
        # Check exposure limits
        if self.position.total_exposure >= self.config.max_total_exposure:
            self._log_state(f"⚠️ Max exposure ${self.position.total_exposure:.2f} reached")
            return False
        
        if not should_enter:
            self.sweeps_skipped += 1
            # Log occasionally
            now = time.time()
            if now - self.last_log_time > 10:
                self._log_state(f"⏸️ Waiting: {reason}")
                self.last_log_time = now
            return False
        
        # EXECUTE SWEEP!
        print(f"[BundleArb] 🎯 SWEEP! Bundle=${book.bundle_cost_ask:.3f} (YES=${book.best_ask_yes:.3f} NO=${book.best_ask_no:.3f})")
        
        yes_fills, no_fills, yes_vol, no_vol = await self.sweeper.sweep_both_sides(
            self.yes_token,
            self.no_token,
            book,
            self.position,
            self.balancer,
            self.market_slug,
            self.condition_id,
        )
        
        self.sweeps_executed += 1
        total_fills = yes_fills + no_fills
        
        if total_fills > 0:
            # Auto-merge if profitable
            if self.position.bundle_cost < 1.0 and self.position.hedged_shares > 0:
                merged_profit = await self._settle_hedged_pairs()
                if merged_profit > 0:
                    print(f"[BundleArb] 💰 AUTO-MERGE: +${merged_profit:.2f} realized")
            
            self._log_state(f"✅ Sweep #{self.sweeps_executed}: {yes_fills}Y/{no_fills}N fills, {yes_vol:.0f}/{no_vol:.0f} vol")
            return True
        
        return False
    
    def _log_state(self, prefix: str = ""):
        """Log current state"""
        pos = self.position
        book = self.calculator.current
        
        bc_str = f"${book.bundle_cost_ask:.3f}" if book else "N/A"
        imb_str = f"{pos.imbalance*100:.1f}%" if pos.yes_shares + pos.no_shares > 0 else "N/A"
        
        print(f"[BundleArb] {prefix}")
        print(f"[BundleArb]    Bundle: {bc_str} | Pos: {pos.yes_shares:.0f}Y/{pos.no_shares:.0f}N | Imb: {imb_str}")
        print(f"[BundleArb]    AvgCost: ${pos.bundle_cost:.3f} | P&L: ${pos.total_pnl:+.2f} (${pos.realized_pnl:+.2f} real)")
    
    def print_summary(self):
        """Print session summary"""
        pos = self.position
        
        print("\n" + "="*65)
        print("🐙 BUNDLE ARBITRAGE SESSION SUMMARY")
        print("="*65)
        print(f"Sweeps Executed:      {self.sweeps_executed}")
        print(f"Sweeps Skipped:       {self.sweeps_skipped}")
        if self.sweeper:
            print(f"Total Orders:         {self.sweeper.total_orders_placed}")
            print(f"Total Fills:          {self.sweeper.total_fills}")
            print(f"YES Fills:            {self.sweeper.yes_fills}")
            print(f"NO Fills:             {self.sweeper.no_fills}")
        print("-"*65)
        print(f"Position YES:         {pos.yes_shares:.2f} @ ${pos.yes_avg_price:.4f}")
        print(f"Position NO:          {pos.no_shares:.2f} @ ${pos.no_avg_price:.4f}")
        print(f"Hedged Pairs:         {pos.hedged_shares:.2f}")
        print(f"Bundle Cost (avg):    ${pos.bundle_cost:.4f}")
        print(f"Position Imbalance:   {pos.imbalance*100:.1f}%")
        print("-"*65)
        print(f"Total Exposure:       ${pos.total_exposure:.2f}")
        print(f"Unrealized P&L:       ${pos.unrealized_pnl:+.2f}")
        print(f"Realized P&L:         ${pos.realized_pnl:+.2f}")
        print(f"TOTAL P&L:            ${pos.total_pnl:+.2f}")
        if pos.total_exposure > 0:
            roi = (pos.total_pnl / pos.total_exposure) * 100
            print(f"ROI:                  {roi:+.2f}%")
        print("="*65)

    async def _settle_hedged_pairs(self) -> float:
        """
        Merge hedged YES/NO pairs on-chain (if enabled) and update local P&L.
        Returns realized profit.
        """
        hedged = self.position.hedged_shares
        if hedged <= 0 or self.position.bundle_cost >= 1.0:
            return 0.0

        # Respect merge thresholds to avoid spamming on-chain txs
        if hedged < self.config.merge_min_hedged and self.position.unrealized_pnl < self.config.merge_min_unrealized:
            return 0.0

        if self.settlement_client and ENABLE_ONCHAIN_MERGE:
            result = await self.settlement_client.merge_positions(self.condition_id, hedged)
            if result.tx_hash:
                print(f"[BundleArb] 🔗 On-chain merge submitted: {result.tx_hash}")
            else:
                print(f"[BundleArb] ⚠️ On-chain merge skipped: {result.error}")

        # Update local accounting regardless so we do not double-count exposure.
        realized = self.position.auto_merge()
        return realized

    async def redeem_winning_tokens(self, condition_id: str, index_sets: List[int]) -> Optional[str]:
        """
        Redeem resolved winning tokens for USDC if enabled. Returns tx hash.
        """
        if not self.settlement_client or not ENABLE_ONCHAIN_REDEEM:
            return None

        result = await self.settlement_client.redeem_positions(condition_id, index_sets)
        if result.tx_hash:
            print(f"[BundleArb] 🏁 Redemption submitted for {condition_id}: {result.tx_hash}")
        else:
            print(f"[BundleArb] ⚠️ Redemption skipped: {result.error}")
        return result.tx_hash if result.tx_hash else None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_bundle_arbitrageur(config: Optional[BundleConfig] = None) -> BundleArbitrageur:
    """Factory function to create a BundleArbitrageur instance"""
    return BundleArbitrageur(config)

