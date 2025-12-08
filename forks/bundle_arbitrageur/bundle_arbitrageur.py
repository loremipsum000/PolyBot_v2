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
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import deque
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from py_clob_client.constants import POLYGON

from config import (
    PRIVATE_KEY,
    POLYGON_ADDRESS,
    SIGNATURE_TYPE,
    ENABLE_ONCHAIN_REDEEM,
    BA_MAX_BUNDLE_COST,
    BA_TARGET_BUNDLE_COST,
    BA_ABORT_BUNDLE_COST,
    BA_BID_BUNDLE_THRESHOLD,
    BA_MAX_IMBALANCE,
    BA_MIN_POSITION_SIZE,
    BA_MAX_TOTAL_EXPOSURE,
    BA_MAX_YES_PRICE,
    BA_MAX_NO_PRICE,
    BA_CHEAP_SIDE_THRESHOLD,
    BA_CHEAP_YES_BID,
    BA_CHEAP_NO_BID,
    BA_ORDER_SIZE,
    BA_CLIP_LADDER,
    BA_BURST_CHILD_COUNT,
    BA_SWEEP_BURST_SIZE,
    BA_SWEEP_INTERVAL_MS,
    BA_MIN_ORDER_VALUE,
    BA_FILLS_PER_SWEEP,
    BA_SPREAD_MIN,
    BA_DEPTH_MIN,
    BA_MORNING_BONUS_START,
    BA_MORNING_BONUS_END,
    BA_MORNING_BONUS_MULT,
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
    max_bundle_cost: float = BA_MAX_BUNDLE_COST       # Was 0.98, avg VWAP is 0.987
    target_bundle_cost: float = BA_TARGET_BUNDLE_COST   # Was 0.97, realistic target
    abort_bundle_cost: float = BA_ABORT_BUNDLE_COST    # Tighter - losses happen at >1.00
    bid_bundle_threshold: float = BA_BID_BUNDLE_THRESHOLD # Trigger on bid_sum for bursts (cheap bundle)
    
    # Position Balance (winners avg 7.6% imbalance, losers 28.3%)
    max_imbalance: float = BA_MAX_IMBALANCE         # Max 10% difference between YES/NO
    min_position_size: float = BA_MIN_POSITION_SIZE      # Don't bother with tiny positions
    max_total_exposure: float = BA_MAX_TOTAL_EXPOSURE      # Max total capital at risk (kept tiny for tests)
    
    # Entry Price Thresholds
    max_yes_price: float = BA_MAX_YES_PRICE
    max_no_price: float = BA_MAX_NO_PRICE
    cheap_side_threshold: float = BA_CHEAP_SIDE_THRESHOLD  # used for cheap-side triggers (may be disabled)
    cheap_yes_bid: float = BA_CHEAP_YES_BID         # retained but unused once cheap triggers are removed
    cheap_no_bid: float = BA_CHEAP_NO_BID          # retained but unused once cheap triggers are removed
    
    # Execution (avg 16.8 fills per transaction - AGGRESSIVE multi-fill)
    order_size: float = BA_ORDER_SIZE               # Legacy single clip (fallback)
    clip_ladder: Tuple[float, ...] = BA_CLIP_LADDER  # bursty fixed clips, tiny
    burst_child_count: int = BA_BURST_CHILD_COUNT          # orders per trigger (pairs or singles)
    sweep_burst_size: int = BA_SWEEP_BURST_SIZE           # per-second rate cap
    sweep_interval_ms: float = BA_SWEEP_INTERVAL_MS        # Was 10ms, faster execution
    min_order_value: float = BA_MIN_ORDER_VALUE        # Minimum $1 order value
    fills_per_sweep: int = BA_FILLS_PER_SWEEP            # tiny burst, for testing
    spread_min: float = BA_SPREAD_MIN            # require enough spread (avoid micro-spread noise)
    depth_min: float = BA_DEPTH_MIN              # min depth at touch per leg (shares)
    # Time-based scoring (early morning has better arbitrage)
    morning_bonus_start: int = BA_MORNING_BONUS_START        # UTC hour
    morning_bonus_end: int = BA_MORNING_BONUS_END         # UTC hour
    morning_bonus_mult: float = BA_MORNING_BONUS_MULT     # 10% score bonus


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
    

class BundleJsonLogger:
    """
    Lightweight JSONL logger for bundle arbitrage events.
    Writes to data/bundle_logs/bundle_session_<ts>.jsonl
    """

    def __init__(self, log_dir: str = "data/bundle_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"bundle_session_{self.session_id}.jsonl"
        self._lock = threading.Lock()
        # Session start marker
        self.log({"event_type": "session_start"})

    def log(self, record: Dict):
        rec = record.copy()
        rec.setdefault("ts", int(time.time()))
        rec.setdefault("session_id", self.session_id)
        try:
            with self._lock:
                with open(self.session_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            # Don't raise from logger; just print for visibility
            print(f"[BundleLog] write error: {e}")


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
        action, reason = self.get_trigger_action()
        return action is not None, reason
    
    def should_abort(self) -> Tuple[bool, str]:
        """Check if we should stop accumulating"""
        if not self.current:
            return True, "No market data"
        
        bc = self.current.bundle_cost_ask
        if bc >= self.config.abort_bundle_cost:
            return True, f"Bundle cost ${bc:.3f} >= abort ${self.config.abort_bundle_cost}"
        
        return False, ""

    def get_trigger_action(self) -> Tuple[Optional[str], str]:
        """
        Determine which trigger fired.
        Returns (action, reason) where action in { 'bundle', None }
        
        TAKER STRATEGY: We BUY at the ASK, so bundle_cost_ask is what matters.
        bundle_cost_bid is irrelevant (that's what we'd get if we SOLD).
        """
        if not self.current:
            return None, "No market data"
        
        book = self.current
        bc_ask = book.bundle_cost_ask
        
        # Basic liquidity/price sanity
        depth_yes = min(book.best_bid_yes_size, book.best_ask_yes_size)
        depth_no = min(book.best_bid_no_size, book.best_ask_no_size)
        depth_ok = (depth_yes >= self.config.depth_min) and (depth_no >= self.config.depth_min)
        if not depth_ok:
            return None, f"Depth too thin (YES:{depth_yes:.1f} NO:{depth_no:.1f})"
        if not book.has_liquidity:
            return None, "Book frozen/empty"
        
        # Price caps (per-leg)
        if book.best_ask_yes > self.config.max_yes_price:
            return None, f"YES ask ${book.best_ask_yes:.3f} > max ${self.config.max_yes_price}"
        if book.best_ask_no > self.config.max_no_price:
            return None, f"NO ask ${book.best_ask_no:.3f} > max ${self.config.max_no_price}"

        # PRIMARY TRIGGER: Bundle arbitrage opportunity (ask side only)
        if bc_ask <= self.config.max_bundle_cost:
            profit_per_pair = 1.00 - bc_ask
            return "bundle", f"Ask bundle ${bc_ask:.3f} (profit ${profit_per_pair:.3f}/pair)"
        
        # STRICT NEUTRAL: no one-sided cheap triggers
        return None, f"No opportunity (bundle_ask=${bc_ask:.3f}, need <=${self.config.max_bundle_cost:.2f})"


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
    
    def get_balanced_order_sizes(self, position: Position, book: OrderBookState, base_size: float) -> Tuple[float, float]:
        """
        Calculate order sizes that maintain balance.
        Returns (yes_size, no_size)
        """
        # If balanced enough, buy equal clips
        if position.imbalance <= self.config.max_imbalance:
            return base_size, base_size
        # If imbalanced, buy ONLY the light side
        if position.yes_shares > position.no_shares:
            return 0.0, base_size  # need NO
        else:
            return base_size, 0.0  # need YES


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
    
    def __init__(self, clob_client: ClobClient, config: BundleConfig, logger: TradeLogger, event_logger=None):
        self.client = clob_client
        self.config = config
        self.logger = logger
        self.event_logger = event_logger
        
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
    
    @staticmethod
    def _safe_min(values: List[float]) -> float:
        vals = [v for v in values if v is not None and v > 0]
        return min(vals) if vals else 0.0
    
    def _select_clip(self, ladder: Tuple[float, ...], *depths: float) -> float:
        """Pick the largest clip that fits available depth (2x clip) and depth_min."""
        avail = self._safe_min(list(depths))
        for size in sorted(ladder, reverse=True):
            if avail >= max(self.config.depth_min, size * 2):
                return size
        # fallback to smallest clip
        return sorted(ladder)[0] if ladder else self.config.order_size
    
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
        force_panic: bool = False,
    ) -> Tuple[int, int, float, float]:
        """
        Execute MULTI-FILL sweep on BOTH YES and NO sides.
        
        UPDATED execution pattern (from 25K trade analysis):
        - Target ~17 fills per sweep (avg 16.8 from gabagool22)
        - Alternate YES/NO orders to maintain balance
        - Use smaller order sizes for better fill rates
        
        Returns (yes_fills, no_fills, yes_volume, no_volume)
        """
        # Require both sides present
        if book.best_ask_yes <= 0 or book.best_ask_no <= 0 or not book.has_liquidity:
            print(f"[Sweeper] ⛔ BLOCKED: Missing/empty book (YES:{book.best_ask_yes} NO:{book.best_ask_no})")
            return 0, 0, 0.0, 0.0

        is_panic = force_panic or (position.imbalance > self.config.max_imbalance)

        orders_to_place: List[Tuple[str, float, float]] = []

        if is_panic:
            # Panic hedge: ignore bundle/price caps; hedge only the light side
            missing_side = "NO" if position.yes_shares > position.no_shares else "YES"
            hedge_price = book.best_ask_no if missing_side == "NO" else book.best_ask_yes
            if hedge_price <= 0 or hedge_price >= 0.99:
                print(f"[Sweeper] ⚠️ Panic hedge expensive/invalid ({hedge_price:.3f}); waiting")
                return 0, 0, 0.0, 0.0

            clip_size = self._select_clip(self.config.clip_ladder, book.best_ask_yes_size, book.best_ask_no_size)

            # Panic safety: cap burst notional
            PANIC_MAX_BURST_VALUE = 500.0
            est_burst_value = clip_size * hedge_price * self.config.burst_child_count
            if est_burst_value > PANIC_MAX_BURST_VALUE:
                safe_count = max(1, int(PANIC_MAX_BURST_VALUE // max(1e-9, clip_size * hedge_price)))
                target_pairs = safe_count
            else:
                target_pairs = max(1, self.config.burst_child_count)

            orders_to_place = [(missing_side, hedge_price, clip_size)] * target_pairs
            print(f"[Sweeper] 🚨 PANIC: Firing {target_pairs}x {missing_side} @ ${hedge_price:.3f}")

        else:
            # Normal bundle entry with caps
            bundle_cost = book.best_ask_yes + book.best_ask_no
            if bundle_cost > self.config.max_bundle_cost:
                print(f"[Sweeper] ⛔ BLOCKED: Bundle ${bundle_cost:.3f} > max ${self.config.max_bundle_cost:.3f}")
                return 0, 0, 0.0, 0.0

            yes_price = min(book.best_ask_yes, self.config.max_yes_price)
            no_price = min(book.best_ask_no, self.config.max_no_price)
            if yes_price > self.config.max_yes_price or no_price > self.config.max_no_price:
                print(f"[Sweeper] ⛔ BLOCKED: Price cap exceeded (YES:{yes_price:.3f}>{self.config.max_yes_price:.3f} or NO:{no_price:.3f}>{self.config.max_no_price:.3f})")
                return 0, 0, 0.0, 0.0

            profit_potential = 1.00 - bundle_cost
            print(f"[Sweeper] ✅ ENTERING: Bundle ${bundle_cost:.3f} (profit potential ${profit_potential:.3f}/pair)")

            clip_size = self._select_clip(self.config.clip_ladder, book.best_ask_yes_size, book.best_ask_no_size)
            base_yes, base_no = balancer.get_balanced_order_sizes(
                position, book, clip_size
            )

            target_pairs = max(1, self.config.burst_child_count)
            # Apply exposure cap only in standard mode
            pair_notional = (yes_price + no_price) * min(base_yes, base_no)
            remaining_exposure = max(0.0, self.config.max_total_exposure - position.total_exposure)
            if pair_notional > 0:
                max_pairs_by_cap = max(1, int(remaining_exposure // pair_notional))
                target_pairs = min(target_pairs, max_pairs_by_cap)

            start_side = 'NO' if position.yes_shares > position.no_shares else 'YES'
            for _ in range(target_pairs):
                if start_side == 'YES':
                    orders_to_place.append(('YES', yes_price, base_yes))
                    orders_to_place.append(('NO', no_price, base_no))
                else:
                    orders_to_place.append(('NO', no_price, base_no))
                    orders_to_place.append(('YES', yes_price, base_yes))
        
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
                
                # Log per-fill event
                token_for_log = yes_token if side == 'YES' else no_token
                self.logger.log_order_placed(
                    market_slug=market_slug,
                    condition_id=condition_id,
                    token_id=token_for_log,
                    outcome='Up' if side == 'YES' else 'Down',
                    side='BUY',
                    price=price,
                    size=float(filled_size),
                    order_id=order_id or "unknown",
                    status=status,
                    market_context={'bundle_arbitrage': True, 'multi_fill': True},
                )
                self.logger.update_position_from_fill('Up' if side == 'YES' else 'Down', 'BUY', price, float(filled_size))
                # JSON event log for analysis
                if self.event_logger:
                    self.event_logger.log({
                        "event_type": "fill",
                        "market_slug": market_slug,
                        "condition_id": condition_id,
                        "side": side,
                        "price": price,
                        "size": float(filled_size),
                        "order_id": order_id,
                        "position": {
                            "yes_shares": position.yes_shares,
                            "no_shares": position.no_shares,
                            "bundle_cost": position.bundle_cost,
                            "imbalance": position.imbalance,
                        },
                    })
        
        # Track fills per sweep for analysis
        total_fills = yes_fills_count + no_fills_count
        self.sweep_fill_counts.append(total_fills)
        
        if total_fills > 0:
            avg_fills = sum(self.sweep_fill_counts) / len(self.sweep_fill_counts)
            print(f"[Sweeper] Multi-fill: {total_fills} fills this sweep (avg: {avg_fills:.1f})")
        
        return yes_fills_count, no_fills_count, yes_volume, no_volume

    async def sweep_one_side(
        self,
        side: str,
        yes_token: str,
        no_token: str,
        book: OrderBookState,
        position: Position,
        market_slug: str = "",
        condition_id: str = "",
    ) -> Tuple[int, int, float, float]:
        """
        Burst orders on a single side (cheap-side tap).
        Returns (yes_fills, no_fills, yes_volume, no_volume)
        """
        token_id = yes_token if side == "YES" else no_token
        price = book.best_ask_yes if side == "YES" else book.best_ask_no
        depth_side = book.best_ask_yes_size if side == "YES" else book.best_ask_no_size
        if price <= 0 or depth_side <= 0:
            return 0, 0, 0.0, 0.0

        # Imbalance guard: avoid worsening imbalance beyond threshold
        if position.imbalance >= self.config.max_imbalance:
            if (side == "YES" and position.yes_shares > position.no_shares) or (
                side == "NO" and position.no_shares > position.yes_shares
            ):
                return 0, 0, 0.0, 0.0
        
        clip_size = self._select_clip(self.config.clip_ladder, depth_side)
        remaining_exposure = max(0.0, self.config.max_total_exposure - position.total_exposure)
        max_orders_by_cap = max(1, int(remaining_exposure // (price * clip_size)) if price * clip_size > 0 else 1)
        target_orders = min(max(1, self.config.burst_child_count), max_orders_by_cap)
        orders_to_place = [(side, price, clip_size)] * target_orders
        
        async def _execute(order_side, p, s):
            return (order_side, await self.place_order(token_id, order_side, p, s, market_slug, condition_id))
        
        tasks = [_execute(side, price, clip_size) for _ in orders_to_place]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        yes_fills_count = 0
        no_fills_count = 0
        yes_volume = 0.0
        no_volume = 0.0
        
        for result in results:
            if isinstance(result, Exception):
                continue
            order_side, (order_id, status, filled_size) = result
            if status == "matched" and filled_size > 0:
                if order_side == "YES":
                    yes_fills_count += 1
                    yes_volume += filled_size
                else:
                    no_fills_count += 1
                    no_volume += filled_size
                position.update_from_fill(order_side, price, filled_size)
                self.logger.log_order_placed(
                    market_slug=market_slug,
                    condition_id=condition_id,
                    token_id=token_id,
                    outcome='Up' if order_side == 'YES' else 'Down',
                    side='BUY',
                    price=price,
                    size=float(filled_size),
                    order_id=order_id or "unknown",
                    status=status,
                    market_context={'bundle_arbitrage': True, 'single_leg': True},
                )
                self.logger.update_position_from_fill('Up' if order_side == 'YES' else 'Down', 'BUY', price, float(filled_size))
        
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
        self.event_logger = BundleJsonLogger(log_dir="data/bundle_logs")
        
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
            signature_type=SIGNATURE_TYPE,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[BundleArb] ✅ CLOB client authenticated")
        
        # Initialize sweeper
        self.sweeper = DualSideSweeper(self.clob_client, self.config, self.logger, event_logger=self.event_logger)
        if ENABLE_ONCHAIN_REDEEM and not self.settlement_client:
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
        
        # Check trigger (bundle only; cheap-side removed)
        action, reason = self.calculator.get_trigger_action()
        should_enter = action is not None
        
        # Also check abort condition
        should_abort, abort_reason = self.calculator.should_abort()
        is_imbalanced = self.position.imbalance > self.config.max_imbalance
        
        # Abort only when balanced
        if should_abort and not is_imbalanced:
            if self.sweeps_executed > 0:  # Only log if we were trading
                print(f"[BundleArb] 🛑 ABORT: {abort_reason}")
            return False
        
        # Exposure cap only when balanced
        if not is_imbalanced and self.position.total_exposure >= self.config.max_total_exposure:
            self._log_state(f"⚠️ Max exposure ${self.position.total_exposure:.2f} reached")
            return False
        
        # Entry gate: if not panic and no trigger, skip
        if not should_enter and not is_imbalanced:
            self.sweeps_skipped += 1
            now = time.time()
            if now - self.last_log_time > 10:
                self._log_state(f"⏸️ Waiting: {reason}")
                self.last_log_time = now
            return False
        
        # Force panic action when imbalanced but no trigger
        if is_imbalanced and not should_enter:
            action = "PANIC"
            reason = f"Forcing hedge for {self.position.imbalance*100:.1f}% imbalance"
        
        # EXECUTE SWEEP!
        print(f"[BundleArb] 🎯 SWEEP {action}! BidBundle=${book.bundle_cost_bid:.3f} AskBundle=${book.bundle_cost_ask:.3f}")
        # Log sweep attempt with current state
        self.event_logger.log({
            "event_type": "sweep_attempt",
            "market_slug": self.market_slug,
            "condition_id": self.condition_id,
            "bundle_cost": book.bundle_cost_ask,
            "bundle_cost_bid": book.bundle_cost_bid,
            "best_ask_yes": book.best_ask_yes,
            "best_ask_no": book.best_ask_no,
            "best_bid_yes": book.best_bid_yes,
            "best_bid_no": book.best_bid_no,
            "position": {
                "yes_shares": self.position.yes_shares,
                "no_shares": self.position.no_shares,
                "bundle_cost": self.position.bundle_cost,
                "imbalance": self.position.imbalance,
            },
            "reason": reason,
            "action": action,
        })
        
        if action in ("bundle", "PANIC"):
            yes_fills, no_fills, yes_vol, no_vol = await self.sweeper.sweep_both_sides(
                self.yes_token,
                self.no_token,
                book,
                self.position,
                self.balancer,
                self.market_slug,
                self.condition_id,
                force_panic=(action == "PANIC"),
            )
        else:
            return False
        
        self.sweeps_executed += 1
        total_fills = yes_fills + no_fills
        
        if total_fills > 0:
            # Log sweep result
            self.event_logger.log({
                "event_type": "sweep_result",
                "market_slug": self.market_slug,
                "condition_id": self.condition_id,
                "bundle_cost": book.bundle_cost_ask,
                "yes_fills": yes_fills,
                "no_fills": no_fills,
                "yes_volume": yes_vol,
                "no_volume": no_vol,
                "position": {
                    "yes_shares": self.position.yes_shares,
                    "no_shares": self.position.no_shares,
                    "bundle_cost": self.position.bundle_cost,
                    "imbalance": self.position.imbalance,
                },
            })
            
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

