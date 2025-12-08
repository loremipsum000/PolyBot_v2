#!/usr/bin/env python3
"""
Live Test Script for Bundle Arbitrage Strategy.

This script runs the bundle arbitrageur with:
- WEBSOCKET REAL-TIME streaming (sub-100ms opportunity detection)
- ISOLATED config (small position sizes for extended testing)
- DETAILED logging (market snapshots vs fills for analysis)
- Market movement tracking (understand execution quality)

Output: data/test_logs/test_session_<timestamp>.jsonl
"""

import asyncio
import sys
import time
import json
import signal
import aiohttp
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Add project to path
sys.path.insert(0, '.')

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

from config import (
    PRIVATE_KEY,
    POLYGON_ADDRESS,
    SIGNATURE_TYPE,
    BTC_SLUG_PREFIX,
    ETH_SLUG_PREFIX,
    WS_URL,
    BA_MAX_BUNDLE_COST,
    BA_MAX_YES_PRICE,
    BA_MAX_NO_PRICE,
    BA_DEPTH_MIN,
)
from utils import get_target_markets

from bundle_arbitrageur import (
    BundleArbitrageur, BundleConfig, OrderBookState, Position,
    BundleCostCalculator, PositionBalancer, BundleJsonLogger
)


# =============================================================================
# ISOLATED TEST CONFIG (small sizes for extended testing)
# =============================================================================

def create_test_config() -> BundleConfig:
    """
    Create an ISOLATED config for testing.
    Uses smaller sizes to run longer and collect more data.
    Does NOT modify the main config.py.
    """
    return BundleConfig(
        # Bundle Cost Thresholds
        # CRITICAL: For profitable arbitrage, bundle_cost_ask must be < $1.00
        # We use 1.02 to allow slight overpay near expiry (aggressive mode)
        max_bundle_cost=1.02,          # Max we'll pay for a bundle (YES + NO asks)
        target_bundle_cost=0.985,      # Target bundle cost for entries
        abort_bundle_cost=1.02,        # Abort if bundle cost exceeds this
        
        # Position Balance
        max_imbalance=0.10,
        min_position_size=0,           # Allow immediate trading
        max_total_exposure=100,        # SMALL: $100 max for testing
        
        # Entry Price Thresholds
        max_yes_price=0.99,
        max_no_price=0.99,
        cheap_side_threshold=0.40,     # One-sided trigger: only if YES/NO ask <= $0.40
        cheap_yes_bid=0.45,
        cheap_no_bid=0.45,
        
        # Execution (SMALLER clips for more fills, more data)
        order_size=3,                  # Small fallback
        clip_ladder=(3, 4, 5),         # Small clips for testing
        burst_child_count=2,           # 2 pairs = 4 orders per sweep
        sweep_burst_size=8,            # Per-second cap
        sweep_interval_ms=10,          # 10ms between orders
        min_order_value=1.0,
        fills_per_sweep=4,
        spread_min=0.0,                # Don't gate on spread (we're takers)
        depth_min=2.0,                 # Minimum liquidity at touch
        
        # Merge settings
        merge_min_hedged=10,           # Low threshold for testing
        merge_min_unrealized=0.5,
    )


# =============================================================================
# DETAILED TEST LOGGER (for analysis)
# =============================================================================

@dataclass
class MarketSnapshot:
    """Captures full market state at a point in time."""
    timestamp: float
    market_slug: str
    
    # YES orderbook
    best_bid_yes: float
    best_ask_yes: float
    best_bid_yes_size: float
    best_ask_yes_size: float
    spread_yes: float
    
    # NO orderbook
    best_bid_no: float
    best_ask_no: float
    best_bid_no_size: float
    best_ask_no_size: float
    spread_no: float
    
    # Bundle metrics
    bundle_cost_bid: float
    bundle_cost_ask: float
    has_arb_opportunity: bool


@dataclass
class FillEvent:
    """Records a fill with market context."""
    timestamp: float
    market_slug: str
    side: str  # YES or NO
    price: float
    size: float
    order_id: str
    status: str
    
    # Market state AT TIME OF FILL
    market_best_ask: float
    market_best_bid: float
    bundle_cost_at_fill: float
    
    # Execution quality metrics
    slippage_vs_best: float  # price - best_ask (should be 0 for IOC at ask)
    
    # Position state after fill
    position_yes: float
    position_no: float
    position_imbalance: float
    position_bundle_cost: float


class TestLogger:
    """
    Comprehensive logger for test analysis.
    Captures market snapshots and fill events for post-hoc analysis.
    """
    
    def __init__(self, log_dir: str = "data/test_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"test_session_{self.session_id}.jsonl"
        
        # In-memory tracking for summary
        self.snapshots: List[MarketSnapshot] = []
        self.fills: List[FillEvent] = []
        self.trigger_events: List[Dict] = []
        self.skip_events: List[Dict] = []
        
        self._write({"event_type": "session_start", "session_id": self.session_id})
        print(f"[TestLog] 📝 Logging to {self.session_file}")
    
    def _write(self, record: Dict):
        """Write a record to the JSONL log."""
        record["ts"] = time.time()
        record["session_id"] = self.session_id
        try:
            with open(self.session_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[TestLog] Write error: {e}")
    
    def log_config(self, config: BundleConfig):
        """Log the test configuration."""
        self._write({
            "event_type": "config",
            "config": {
                "max_bundle_cost": config.max_bundle_cost,
                "abort_bundle_cost": config.abort_bundle_cost,
                "max_total_exposure": config.max_total_exposure,
                "max_yes_price": config.max_yes_price,
                "max_no_price": config.max_no_price,
                "clip_ladder": list(config.clip_ladder),
                "burst_child_count": config.burst_child_count,
                "sweep_burst_size": config.sweep_burst_size,
                "spread_min": config.spread_min,
                "depth_min": config.depth_min,
            }
        })
    
    def log_market_snapshot(self, book: OrderBookState, market_slug: str):
        """Log a market snapshot for later analysis."""
        snapshot = MarketSnapshot(
            timestamp=time.time(),
            market_slug=market_slug,
            best_bid_yes=book.best_bid_yes,
            best_ask_yes=book.best_ask_yes,
            best_bid_yes_size=book.best_bid_yes_size,
            best_ask_yes_size=book.best_ask_yes_size,
            spread_yes=book.spread_yes,
            best_bid_no=book.best_bid_no,
            best_ask_no=book.best_ask_no,
            best_bid_no_size=book.best_bid_no_size,
            best_ask_no_size=book.best_ask_no_size,
            spread_no=book.spread_no,
            bundle_cost_bid=book.bundle_cost_bid,
            bundle_cost_ask=book.bundle_cost_ask,
            has_arb_opportunity=book.has_arbitrage_opportunity,
        )
        self.snapshots.append(snapshot)
        self._write({"event_type": "market_snapshot", **asdict(snapshot)})
    
    def log_trigger(self, action: str, reason: str, book: OrderBookState, position: Position):
        """Log a trigger event (sweep attempt)."""
        event = {
            "event_type": "trigger",
            "action": action,
            "reason": reason,
            "bundle_cost_ask": book.bundle_cost_ask,
            "bundle_cost_bid": book.bundle_cost_bid,
            "best_ask_yes": book.best_ask_yes,
            "best_ask_no": book.best_ask_no,
            "position_yes": position.yes_shares,
            "position_no": position.no_shares,
            "position_imbalance": position.imbalance,
        }
        self.trigger_events.append(event)
        self._write(event)
    
    def log_skip(self, reason: str, book: Optional[OrderBookState], position: Position):
        """Log a skipped sweep with reason."""
        event = {
            "event_type": "skip",
            "reason": reason,
            "bundle_cost_ask": book.bundle_cost_ask if book else None,
            "position_yes": position.yes_shares,
            "position_no": position.no_shares,
        }
        self.skip_events.append(event)
        # Only write every 10th skip to avoid log spam
        if len(self.skip_events) % 10 == 0:
            self._write(event)
    
    def log_fill(
        self,
        side: str,
        price: float,
        size: float,
        order_id: str,
        status: str,
        book: OrderBookState,
        position: Position,
        market_slug: str,
    ):
        """Log a fill event with full context."""
        best_ask = book.best_ask_yes if side == "YES" else book.best_ask_no
        best_bid = book.best_bid_yes if side == "YES" else book.best_bid_no
        
        fill = FillEvent(
            timestamp=time.time(),
            market_slug=market_slug,
            side=side,
            price=price,
            size=size,
            order_id=order_id,
            status=status,
            market_best_ask=best_ask,
            market_best_bid=best_bid,
            bundle_cost_at_fill=book.bundle_cost_ask,
            slippage_vs_best=price - best_ask,
            position_yes=position.yes_shares,
            position_no=position.no_shares,
            position_imbalance=position.imbalance,
            position_bundle_cost=position.bundle_cost,
        )
        self.fills.append(fill)
        self._write({"event_type": "fill", **asdict(fill)})
    
    def log_sweep_result(
        self,
        yes_fills: int,
        no_fills: int,
        yes_volume: float,
        no_volume: float,
        book: OrderBookState,
        position: Position,
    ):
        """Log sweep result summary."""
        self._write({
            "event_type": "sweep_result",
            "yes_fills": yes_fills,
            "no_fills": no_fills,
            "yes_volume": yes_volume,
            "no_volume": no_volume,
            "bundle_cost_ask": book.bundle_cost_ask,
            "position_yes": position.yes_shares,
            "position_no": position.no_shares,
            "position_imbalance": position.imbalance,
            "position_bundle_cost": position.bundle_cost,
            "unrealized_pnl": position.unrealized_pnl,
        })
    
    def print_summary(self, position: Position, duration_seconds: float):
        """Print a summary of the test session."""
        print("\n" + "="*70)
        print("📊 TEST SESSION SUMMARY")
        print("="*70)
        print(f"Session ID:           {self.session_id}")
        print(f"Duration:             {duration_seconds/60:.1f} minutes")
        print(f"Log file:             {self.session_file}")
        print("-"*70)
        print(f"Market Snapshots:     {len(self.snapshots)}")
        print(f"Trigger Events:       {len(self.trigger_events)}")
        print(f"Skip Events:          {len(self.skip_events)}")
        print(f"Total Fills:          {len(self.fills)}")
        
        if self.fills:
            yes_fills = [f for f in self.fills if f.side == "YES"]
            no_fills = [f for f in self.fills if f.side == "NO"]
            print(f"  YES Fills:          {len(yes_fills)}")
            print(f"  NO Fills:           {len(no_fills)}")
            
            avg_slippage = sum(f.slippage_vs_best for f in self.fills) / len(self.fills)
            print(f"  Avg Slippage:       ${avg_slippage:.4f}")
        
        print("-"*70)
        print(f"Final Position YES:   {position.yes_shares:.2f} @ ${position.yes_avg_price:.4f}")
        print(f"Final Position NO:    {position.no_shares:.2f} @ ${position.no_avg_price:.4f}")
        print(f"Hedged Pairs:         {position.hedged_shares:.2f}")
        print(f"Bundle Cost (avg):    ${position.bundle_cost:.4f}")
        print(f"Position Imbalance:   {position.imbalance*100:.1f}%")
        print("-"*70)
        print(f"Total Exposure:       ${position.total_exposure:.2f}")
        print(f"Unrealized P&L:       ${position.unrealized_pnl:+.2f}")
        print(f"Realized P&L:         ${position.realized_pnl:+.2f}")
        print(f"TOTAL P&L:            ${position.total_pnl:+.2f}")
        if position.total_exposure > 0:
            roi = (position.total_pnl / position.total_exposure) * 100
            print(f"ROI:                  {roi:+.2f}%")
        print("="*70)
        
        # Log final summary
        self._write({
            "event_type": "session_end",
            "duration_seconds": duration_seconds,
            "total_snapshots": len(self.snapshots),
            "total_triggers": len(self.trigger_events),
            "total_skips": len(self.skip_events),
            "total_fills": len(self.fills),
            "final_position": {
                "yes_shares": position.yes_shares,
                "no_shares": position.no_shares,
                "bundle_cost": position.bundle_cost,
                "imbalance": position.imbalance,
                "total_exposure": position.total_exposure,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "total_pnl": position.total_pnl,
            }
        })


# =============================================================================
# LIVE TEST RUNNER (WebSocket Real-Time)
# =============================================================================

class LiveTestRunner:
    """
    Runs the bundle arbitrageur with WebSocket real-time streaming.
    Uses reactive execution for sub-100ms opportunity detection.
    """
    
    def __init__(self, config: BundleConfig, test_logger: TestLogger):
        self.config = config
        self.test_logger = test_logger
        self.arbitrageur = BundleArbitrageur(config)
        self.clob_client: Optional[ClobClient] = None
        self.running = True
        self.start_time = time.time()
        
        # Market info
        self.markets: Dict[str, Dict] = {}  # label -> {slug, condition_id, yes_token, no_token}
        self.current_market: Optional[str] = None
        self.token_to_market: Dict[str, str] = {}  # token_id -> market_label
        self.token_to_side: Dict[str, str] = {}    # token_id -> 'YES'/'NO'
        
        # WebSocket state
        self.ws_connected = False
        self._sweep_in_flight = False
        
        # Current book state (updated by WS)
        self.current_book: Optional[OrderBookState] = None
        
        # Stats
        self.ws_updates = 0
        self.opportunities_seen = 0
        self.sweeps_attempted = 0
    
    def init_client(self):
        """Initialize CLOB client."""
        print("[Test] 🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=SIGNATURE_TYPE,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[Test] ✅ CLOB client authenticated")
        
        # Also init the arbitrageur's client
        self.arbitrageur.clob_client = self.clob_client
        self.arbitrageur.init_clob_client()
    
    async def discover_markets(self):
        """Discover available markets."""
        print("[Test] 🔍 Discovering markets...")
        
        for label, prefix in [('BTC', BTC_SLUG_PREFIX), ('ETH', ETH_SLUG_PREFIX)]:
            found = get_target_markets(slug_prefix=prefix, asset_label=label)
            if not found:
                print(f"[Test] ⚠️ No {label} market found")
                continue
            
            m = found[0]
            c_id = m.get('condition_id')
            
            try:
                market_resp = self.clob_client.get_market(c_id)
                accepting = market_resp.get('accepting_orders') if isinstance(market_resp, dict) else getattr(market_resp, 'accepting_orders', None)
                if not accepting:
                    print(f"[Test] ⚠️ {label} market not accepting orders")
                    continue
                
                tokens = market_resp.get('tokens', []) if isinstance(market_resp, dict) else market_resp.tokens
                
                yes_id, no_id = "", ""
                for t in tokens:
                    outcome = (t.get('outcome') or "").upper()
                    if outcome in ['YES', 'UP']:
                        yes_id = t.get('token_id')
                    elif outcome in ['NO', 'DOWN']:
                        no_id = t.get('token_id')
                
                if yes_id and no_id:
                    self.markets[label] = {
                        'slug': m.get('market_slug'),
                        'condition_id': c_id,
                        'yes_token': yes_id,
                        'no_token': no_id,
                    }
                    # Map tokens to market for WS handling
                    self.token_to_market[yes_id] = label
                    self.token_to_market[no_id] = label
                    self.token_to_side[yes_id] = 'YES'
                    self.token_to_side[no_id] = 'NO'
                    print(f"[Test] ✅ {label}: {m.get('question', '')[:50]}...")
                    
            except Exception as e:
                print(f"[Test] ❌ Error with {label}: {e}")
    
    @staticmethod
    def _extract_top(side_data, default_price: float, default_size: float, *, is_ask: bool) -> Tuple[float, float]:
        """Extract best price/size from orderbook side."""
        if not side_data:
            return default_price, default_size
        
        best_price = None
        best_size = default_size
        
        for lvl in side_data:
            try:
                p = float(lvl.get("price")) if isinstance(lvl, dict) else float(lvl[0])
                s = float(lvl.get("size")) if isinstance(lvl, dict) else float(lvl[1])
            except Exception:
                continue
            
            if best_price is None:
                best_price, best_size = p, s
            else:
                if is_ask and p < best_price:
                    best_price, best_size = p, s
                elif not is_ask and p > best_price:
                    best_price, best_size = p, s
        
        return (best_price if best_price is not None else default_price, best_size)
    
    def _check_arbitrage(self, book: OrderBookState) -> bool:
        """Check if there's an arbitrage opportunity (same logic as fixed market_hopper)."""
        depth_yes = min(book.best_bid_yes_size, book.best_ask_yes_size)
        depth_no = min(book.best_bid_no_size, book.best_ask_no_size)
        depth_ok = depth_yes >= self.config.depth_min and depth_no >= self.config.depth_min
        
        return (
            book.bundle_cost_ask <= self.config.max_bundle_cost
            and depth_ok
            and book.best_ask_yes <= self.config.max_yes_price
            and book.best_ask_no <= self.config.max_no_price
        )
    
    async def _handle_ws_message(self, data: Dict):
        """Handle a WebSocket orderbook update - REACTIVE EXECUTION."""
        event_type = data.get("event_type")
        
        # Filter by event_type
        if event_type not in ("book", "price_change", None):
            return
        
        # ============================================================
        # PRICE_CHANGE: Nested 'price_changes' array
        # Structure: {"market": "0x...", "price_changes": [{"asset_id": "...", "price": "0.5", "side": "BUY"}]}
        # ============================================================
        if event_type == "price_change":
            price_changes = data.get("price_changes", [])
            if price_changes:
                for change in price_changes:
                    await self._process_price_change(change)
                return
            # Fallback for flat format
            if data.get("asset_id") and data.get("price"):
                await self._process_price_change(data)
            return
        
        # ============================================================
        # BOOK: Full snapshot with bids/asks arrays
        # ============================================================
        token_id = data.get("asset_id") or data.get("id") or data.get("token_id")
        market_label = self.token_to_market.get(token_id)
        if not market_label or market_label != self.current_market:
            return
        
        self.ws_updates += 1
        market = self.markets[market_label]
        side = self.token_to_side.get(token_id)
        
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        
        bid_price, bid_size = self._extract_top(bids, 0.01, 0.0, is_ask=False)
        ask_price, ask_size = self._extract_top(asks, 0.99, 0.0, is_ask=True)
        
        # Update current book state
        if self.current_book is None:
            self.current_book = OrderBookState()
        
        if side == 'YES':
            self.current_book = OrderBookState(
                timestamp=time.time(),
                best_bid_yes=bid_price,
                best_ask_yes=ask_price,
                best_bid_yes_size=bid_size,
                best_ask_yes_size=ask_size,
                best_bid_no=self.current_book.best_bid_no,
                best_ask_no=self.current_book.best_ask_no,
                best_bid_no_size=self.current_book.best_bid_no_size,
                best_ask_no_size=self.current_book.best_ask_no_size,
            )
        else:  # NO
            self.current_book = OrderBookState(
                timestamp=time.time(),
                best_bid_yes=self.current_book.best_bid_yes,
                best_ask_yes=self.current_book.best_ask_yes,
                best_bid_yes_size=self.current_book.best_bid_yes_size,
                best_ask_yes_size=self.current_book.best_ask_yes_size,
                best_bid_no=bid_price,
                best_ask_no=ask_price,
                best_bid_no_size=bid_size,
                best_ask_no_size=ask_size,
            )
        
        book = self.current_book
        
        # Log interesting bundle prices
        if book.bundle_cost_ask <= 1.05:
            self.test_logger._write({
                "event_type": "ws_bundle_update",
                "bundle_cost_ask": book.bundle_cost_ask,
                "ask_yes": book.best_ask_yes,
                "ask_no": book.best_ask_no,
                "has_opportunity": self._check_arbitrage(book),
                "is_profitable": book.bundle_cost_ask < 1.00,
            })
        
        # 🚀 REACTIVE EXECUTION: Check for arbitrage opportunity
        if self._check_arbitrage(book):
            self.opportunities_seen += 1
            
            # Guard against concurrent sweeps
            if not self._sweep_in_flight:
                self._sweep_in_flight = True
                asyncio.create_task(self._execute_reactive_sweep(book, market['slug']))
    
    async def _process_price_change(self, change: Dict):
        """
        Process a single price_change item from the nested 'price_changes' array.
        Structure: {"asset_id": "...", "price": "0.42", "side": "BUY", "size": "100"}
        """
        token_id = change.get("asset_id") or change.get("id")
        if not token_id:
            return
        
        market_label = self.token_to_market.get(token_id)
        if not market_label or market_label != self.current_market:
            return
        
        try:
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "").upper()
        except (ValueError, TypeError):
            return
        
        if price <= 0:
            return
        
        self.ws_updates += 1
        market = self.markets[market_label]
        token_side = self.token_to_side.get(token_id)
        
        if self.current_book is None:
            self.current_book = OrderBookState()
        
        # Update the appropriate price level based on side
        # BUY = bid, SELL = ask
        if token_side == 'YES':
            if side == "BUY" and size > 0:
                self.current_book = OrderBookState(
                    timestamp=time.time(),
                    best_bid_yes=price,
                    best_ask_yes=self.current_book.best_ask_yes,
                    best_bid_yes_size=size,
                    best_ask_yes_size=self.current_book.best_ask_yes_size,
                    best_bid_no=self.current_book.best_bid_no,
                    best_ask_no=self.current_book.best_ask_no,
                    best_bid_no_size=self.current_book.best_bid_no_size,
                    best_ask_no_size=self.current_book.best_ask_no_size,
                )
            elif side == "SELL" and size > 0:
                self.current_book = OrderBookState(
                    timestamp=time.time(),
                    best_bid_yes=self.current_book.best_bid_yes,
                    best_ask_yes=price,
                    best_bid_yes_size=self.current_book.best_bid_yes_size,
                    best_ask_yes_size=size,
                    best_bid_no=self.current_book.best_bid_no,
                    best_ask_no=self.current_book.best_ask_no,
                    best_bid_no_size=self.current_book.best_bid_no_size,
                    best_ask_no_size=self.current_book.best_ask_no_size,
                )
        else:  # NO
            if side == "BUY" and size > 0:
                self.current_book = OrderBookState(
                    timestamp=time.time(),
                    best_bid_yes=self.current_book.best_bid_yes,
                    best_ask_yes=self.current_book.best_ask_yes,
                    best_bid_yes_size=self.current_book.best_bid_yes_size,
                    best_ask_yes_size=self.current_book.best_ask_yes_size,
                    best_bid_no=price,
                    best_ask_no=self.current_book.best_ask_no,
                    best_bid_no_size=size,
                    best_ask_no_size=self.current_book.best_ask_no_size,
                )
            elif side == "SELL" and size > 0:
                self.current_book = OrderBookState(
                    timestamp=time.time(),
                    best_bid_yes=self.current_book.best_bid_yes,
                    best_ask_yes=self.current_book.best_ask_yes,
                    best_bid_yes_size=self.current_book.best_bid_yes_size,
                    best_ask_yes_size=self.current_book.best_ask_yes_size,
                    best_bid_no=self.current_book.best_bid_no,
                    best_ask_no=price,
                    best_bid_no_size=self.current_book.best_bid_no_size,
                    best_ask_no_size=size,
                )
        
        book = self.current_book
        
        # Log for analysis
        if self.test_logger:
            self.test_logger._write({
                "event_type": "ws_price_change",
                "token_side": token_side,
                "price_side": side,
                "price": price,
                "size": size,
                "bundle_cost_ask": book.bundle_cost_ask,
            })
        
        # 🚀 REACTIVE EXECUTION: Check for arbitrage opportunity
        if self._check_arbitrage(book):
            self.opportunities_seen += 1
            
            if not self._sweep_in_flight:
                self._sweep_in_flight = True
                asyncio.create_task(self._execute_reactive_sweep(book, market['slug']))
    
    async def _execute_reactive_sweep(self, book: OrderBookState, market_slug: str):
        """Execute a sweep reactively when WS detects opportunity."""
        try:
            # Double-check we're still good
            if book.bundle_cost_ask > self.config.max_bundle_cost:
                return
            
            # Check exposure
            if self.arbitrageur.position.total_exposure >= self.config.max_total_exposure:
                return
            
            self.sweeps_attempted += 1
            
            # Log trigger
            action, reason = self.arbitrageur.calculator.get_trigger_action()
            self.test_logger.log_trigger(action or "reactive", reason, book, self.arbitrageur.position)
            
            # Update book and execute
            self.arbitrageur.update_book(book)
            
            start_ts = time.time()
            had_fills = await self.arbitrageur.execute_sweep()
            exec_time_ms = (time.time() - start_ts) * 1000
            
            # Log result
            self.test_logger._write({
                "event_type": "reactive_sweep_result",
                "bundle_cost": book.bundle_cost_ask,
                "had_fills": had_fills,
                "exec_time_ms": exec_time_ms,
                "position_yes": self.arbitrageur.position.yes_shares,
                "position_no": self.arbitrageur.position.no_shares,
            })
            
            if had_fills:
                print(f"[Test] 🎯 FILL! Bundle ${book.bundle_cost_ask:.3f} | "
                      f"Pos: {self.arbitrageur.position.yes_shares:.0f}Y/{self.arbitrageur.position.no_shares:.0f}N | "
                      f"Exec: {exec_time_ms:.0f}ms")
        
        except Exception as e:
            print(f"[Test] ⚠️ Sweep error: {e}")
        finally:
            self._sweep_in_flight = False
    
    async def _ws_ping_loop(self, ws):
        """Send explicit PING messages every 10 seconds as per Polymarket docs."""
        try:
            while True:
                await ws.send_str("PING")
                await asyncio.sleep(10)
        except Exception:
            pass  # Connection closed
    
    async def _ws_listener(self, end_time: float):
        """WebSocket listener for real-time orderbook updates."""
        if not WS_URL:
            print("[Test] ❌ WS_URL not configured")
            return
        
        while self.running and time.time() < end_time:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                        # Subscribe to current market tokens
                        market = self.markets[self.current_market]
                        tokens = [market['yes_token'], market['no_token']]
                        
                        # Per Polymarket docs: https://docs.polymarket.com/quickstart/websocket/WSS-Quickstart
                        payload = {
                            "assets_ids": tokens,
                            "type": "market",
                        }
                        await ws.send_json(payload)
                        print(f"[Test] 🔗 WS subscribed to {self.current_market} tokens (market channel)")
                        self.ws_connected = True
                        
                        # Start ping loop per Polymarket docs
                        ping_task = asyncio.create_task(self._ws_ping_loop(ws))
                        
                        try:
                            async for msg in ws:
                                if not self.running or time.time() >= end_time:
                                    break
                                
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    raw = msg.data.strip()
                                    # Handle non-JSON responses (like PONG)
                                    if raw in ("PONG", ""):
                                        continue
                                    try:
                                        parsed = json.loads(raw)
                                        # Skip empty arrays
                                        if parsed == []:
                                            continue
                                        # Handle both single message (dict) and batch messages (list)
                                        messages = parsed if isinstance(parsed, list) else [parsed]
                                        for data in messages:
                                            if isinstance(data, dict):
                                                await self._handle_ws_message(data)
                                    except Exception:
                                        continue
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                        finally:
                            ping_task.cancel()
                            try:
                                await ping_task
                            except asyncio.CancelledError:
                                pass
            
            except Exception as e:
                self.ws_connected = False
                print(f"[Test] ⚠️ WS error: {e} (reconnecting...)")
                await asyncio.sleep(1)
            finally:
                self.ws_connected = False
    
    def fetch_orderbook(self, label: str) -> Optional[OrderBookState]:
        """Fetch current orderbook for initial state."""
        if label not in self.markets:
            return None
        
        market = self.markets[label]
        try:
            yes_book = self.clob_client.get_order_book(market['yes_token'])
            no_book = self.clob_client.get_order_book(market['no_token'])
            
            return OrderBookState(
                timestamp=time.time(),
                best_bid_yes=float(yes_book.bids[0].price) if yes_book.bids else 0.0,
                best_ask_yes=float(yes_book.asks[0].price) if yes_book.asks else 1.0,
                best_bid_yes_size=float(yes_book.bids[0].size) if yes_book.bids else 0.0,
                best_ask_yes_size=float(yes_book.asks[0].size) if yes_book.asks else 0.0,
                best_bid_no=float(no_book.bids[0].price) if no_book.bids else 0.0,
                best_ask_no=float(no_book.asks[0].price) if no_book.asks else 1.0,
                best_bid_no_size=float(no_book.bids[0].size) if no_book.bids else 0.0,
                best_ask_no_size=float(no_book.asks[0].size) if no_book.asks else 0.0,
            )
        except Exception as e:
            print(f"[Test] ⚠️ Orderbook fetch error: {e}")
            return None
    
    async def run_test_loop(self, duration_minutes: float = 5.0, poll_interval: float = 0.5):
        """
        Run the test loop with WebSocket real-time streaming.
        
        Args:
            duration_minutes: How long to run the test (default 5 min)
            poll_interval: Fallback poll interval (WS is primary)
        """
        self.init_client()
        await self.discover_markets()
        
        if not self.markets:
            print("[Test] ❌ No markets available. Exiting.")
            return
        
        # Log config
        self.test_logger.log_config(self.config)
        
        # Pick first available market
        self.current_market = list(self.markets.keys())[0]
        market = self.markets[self.current_market]
        
        self.arbitrageur.set_market(
            yes_token=market['yes_token'],
            no_token=market['no_token'],
            market_slug=market['slug'],
            condition_id=market['condition_id'],
        )
        
        # Fetch initial orderbook state
        initial_book = self.fetch_orderbook(self.current_market)
        if initial_book:
            self.current_book = initial_book
            self.arbitrageur.update_book(initial_book)
            print(f"[Test] 📊 Initial bundle cost: ${initial_book.bundle_cost_ask:.3f}")
        
        print(f"\n[Test] 🚀 Starting REAL-TIME test on {self.current_market} market")
        print(f"[Test]    Mode: WebSocket streaming (sub-100ms detection)")
        print(f"[Test]    Duration: {duration_minutes} minutes")
        print(f"[Test]    Max Bundle Cost: ${self.config.max_bundle_cost}")
        print(f"[Test]    Max Exposure: ${self.config.max_total_exposure}")
        print(f"[Test]    Press Ctrl+C to stop early\n")
        
        end_time = self.start_time + (duration_minutes * 60)
        
        # Start WebSocket listener
        ws_task = asyncio.create_task(self._ws_listener(end_time))
        
        try:
            # Main status loop
            last_status = time.time()
            while self.running and time.time() < end_time:
                await asyncio.sleep(1)
                
                # Status update every 30 seconds
                elapsed = time.time() - self.start_time
                if time.time() - last_status >= 30:
                    last_status = time.time()
                    remaining = (end_time - time.time()) / 60
                    pos = self.arbitrageur.position
                    bundle_str = f"${self.current_book.bundle_cost_ask:.3f}" if self.current_book else "N/A"
                    print(f"[Test] ⏱️ {elapsed/60:.1f}m elapsed, {remaining:.1f}m remaining | "
                          f"WS updates: {self.ws_updates} | "
                          f"Opps: {self.opportunities_seen} | "
                          f"Sweeps: {self.sweeps_attempted} | "
                          f"Bundle: {bundle_str} | "
                          f"Pos: {pos.yes_shares:.0f}Y/{pos.no_shares:.0f}N")
                
        except KeyboardInterrupt:
            print("\n[Test] 🛑 Interrupted by user")
        except Exception as e:
            print(f"\n[Test] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            
            duration = time.time() - self.start_time
            
            # Print additional WS stats
            print(f"\n[Test] 📡 WebSocket Stats:")
            print(f"       Total updates: {self.ws_updates}")
            print(f"       Opportunities seen: {self.opportunities_seen}")
            print(f"       Sweeps attempted: {self.sweeps_attempted}")
            
            self.test_logger.print_summary(self.arbitrageur.position, duration)
            self.arbitrageur.print_summary()
    
    def stop(self):
        """Stop the test loop."""
        self.running = False


# =============================================================================
# UNIT TESTS (quick validation)
# =============================================================================

def run_unit_tests():
    """Run quick unit tests to validate logic."""
    print("\n" + "="*60)
    print("🧪 UNIT TESTS (Quick Validation)")
    print("="*60)
    
    config = create_test_config()
    
    # Test 1: Config isolation
    print("\nTest 1: Config isolation")
    assert config.max_total_exposure == 100, f"Expected 100, got {config.max_total_exposure}"
    assert config.clip_ladder == (3, 4, 5), f"Expected (3,4,5), got {config.clip_ladder}"
    print("  ✅ Test config is isolated from production")
    
    # Test 2: Bundle cost calculation
    print("\nTest 2: Bundle cost calculation")
    book = OrderBookState(
        best_bid_yes=0.65,
        best_ask_yes=0.67,
        best_bid_no=0.28,
        best_ask_no=0.30,
    )
    assert abs(book.bundle_cost_ask - 0.97) < 0.001
    print(f"  Bundle ask: ${book.bundle_cost_ask:.3f} ✅")
    
    # Test 3: Position tracking
    print("\nTest 3: Position tracking")
    pos = Position()
    pos.update_from_fill('YES', 0.65, 10)
    pos.update_from_fill('NO', 0.30, 10)
    assert pos.hedged_shares == 10
    assert pos.imbalance == 0.0
    assert abs(pos.bundle_cost - 0.95) < 0.001
    print(f"  Hedged: {pos.hedged_shares}, Bundle: ${pos.bundle_cost:.3f} ✅")
    
    # Test 4: P&L calculation
    print("\nTest 4: P&L calculation")
    expected_pnl = (1.0 - 0.95) * 10  # $0.50
    assert abs(pos.unrealized_pnl - expected_pnl) < 0.01
    print(f"  Unrealized P&L: ${pos.unrealized_pnl:.2f} ✅")
    
    # Test 5: Trigger logic - arbitrage opportunity
    print("\nTest 5: Trigger logic - arbitrage opportunity")
    calc = BundleCostCalculator(config)
    # Book with arbitrage: bundle_cost_ask = 0.62 + 0.32 = 0.94 < 1.00 (profitable!)
    book_arb = OrderBookState(
        best_bid_yes=0.60,
        best_ask_yes=0.62,
        best_bid_no=0.30,
        best_ask_no=0.32,
        best_bid_yes_size=10,
        best_ask_yes_size=10,
        best_bid_no_size=10,
        best_ask_no_size=10,
    )
    calc.update(book_arb)
    action, reason = calc.get_trigger_action()
    print(f"  Bundle cost ASK: ${book_arb.bundle_cost_ask:.3f}")
    print(f"  Action: {action}, Reason: {reason}")
    # With bundle_cost_ask = 0.94 <= 1.02, should trigger
    assert action == "bundle", f"Expected 'bundle', got {action}"
    assert "profit" in reason.lower(), f"Reason should mention profit: {reason}"
    print("  ✅ Trigger fires on profitable bundle")
    
    # Test 6: NO trigger on expensive bundle
    print("\nTest 6: NO trigger on expensive bundle (like the test market)")
    # This mimics the market from the failed test: YES@0.99 + NO@0.99 = $1.98
    book_expensive = OrderBookState(
        best_bid_yes=0.01,
        best_ask_yes=0.99,
        best_bid_no=0.01,
        best_ask_no=0.99,
        best_bid_yes_size=1000,
        best_ask_yes_size=1000,
        best_bid_no_size=1000,
        best_ask_no_size=1000,
    )
    calc.update(book_expensive)
    action2, reason2 = calc.get_trigger_action()
    print(f"  Bundle cost ASK: ${book_expensive.bundle_cost_ask:.3f}")
    print(f"  Action: {action2}, Reason: {reason2}")
    # With bundle_cost_ask = 1.98 > 1.02, should NOT trigger
    assert action2 is None, f"Expected None (no trigger), got {action2}"
    print("  ✅ Correctly skips expensive bundle (no arbitrage opportunity)")
    
    print("\n" + "="*60)
    print("✅ ALL UNIT TESTS PASSED")
    print("="*60)
    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Bundle Arbitrage Strategy (WebSocket Real-Time)")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--live", action="store_true", help="Run live WebSocket test")
    parser.add_argument("--duration", type=float, default=5, help="Test duration in minutes (default: 5)")
    args = parser.parse_args()
    
    if args.unit or (not args.live and not args.unit):
        # Run unit tests first
        if not run_unit_tests():
            return 1
    
    if args.live:
        # Run live WebSocket test
        print("\n" + "="*60)
        print("🚀 BUNDLE ARBITRAGE - WEBSOCKET REAL-TIME TEST")
        print("="*60)
        print("Mode: Reactive execution via WebSocket streaming")
        print("Latency target: <100ms from opportunity to order")
        print("="*60 + "\n")
        
        config = create_test_config()
        logger = TestLogger()
        runner = LiveTestRunner(config, logger)
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n[Test] Received stop signal...")
            runner.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        await runner.run_test_loop(duration_minutes=args.duration)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
