"""
Market Hopper - Coordinates trading across multiple markets using Bundle Arbitrage.

STRATEGY OVERHAUL (Dec 5, 2025):
Based on PNL_ANALYSIS_REPORT.md, gabagool22's actual profitable strategy is:
- Bundle Arbitrage: Buy BOTH YES and NO for less than $1.00
- NOT momentum-following (that's what he did when losing money)

Key Insight from gabagool22:
- Lost money for 5+ weeks with directional bets
- Started winning Dec 5 with bundle arbitrage
- PROFIT = (1.00 - bundle_cost) × min(yes_shares, no_shares)

Implementation:
1. Monitor both BTC and ETH markets
2. Calculate bundle cost in real-time  
3. When bundle_cost < 0.98: sweep BOTH sides
4. Maintain position balance (< 10% imbalance)
5. Abort if bundle_cost > 0.99
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta
import aiohttp
from contextlib import suppress

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

from config import (
    PRIVATE_KEY,
    POLYGON_ADDRESS,
    SIGNATURE_TYPE,
    BTC_SLUG_PREFIX,
    ETH_SLUG_PREFIX,
    ENABLE_BUNDLE_ARBITRAGE,
    BA_MAX_BUNDLE_COST,
    BA_TARGET_BUNDLE_COST,
    BA_ABORT_BUNDLE_COST,
    BA_MAX_IMBALANCE,
    BA_MAX_TOTAL_EXPOSURE,
    BA_MAX_YES_PRICE,
    BA_MAX_NO_PRICE,
    BA_CHEAP_SIDE_THRESHOLD,
    BA_SWEEP_BURST_SIZE,
    BA_SWEEP_INTERVAL_MS,
    BA_ORDER_SIZE,
    BA_MORNING_BONUS_START,
    BA_MORNING_BONUS_END,
    BA_MORNING_BONUS_MULT,
    BA_FILLS_PER_SWEEP,
    BA_BID_BUNDLE_THRESHOLD,
    BA_SPREAD_MIN,
    BA_DEPTH_MIN,
    BA_CHEAP_YES_BID,
    BA_CHEAP_NO_BID,
    BA_CLIP_LADDER,
    BA_BURST_CHILD_COUNT,
    WS_URL,
    ENABLE_ONCHAIN_REDEEM,
)
from utils import get_target_markets, warm_connections
from trade_logger import TradeLogger
from bundle_arbitrageur import (
    BundleArbitrageur, BundleConfig, OrderBookState, Position
)


@dataclass
class MarketState:
    """Real-time state of a single market"""
    slug: str
    condition_id: str
    asset_label: str  # 'BTC' or 'ETH'
    question: str = ""
    end_date_iso: str = ""  # Market expiry time for window monitoring
    
    # Token IDs
    yes_token: str = ""
    no_token: str = ""
    
    # Book data
    best_bid_yes: float = 0.0
    best_ask_yes: float = 1.0
    best_bid_yes_size: float = 0.0
    best_ask_yes_size: float = 0.0
    best_bid_no: float = 0.0
    best_ask_no: float = 1.0
    best_bid_no_size: float = 0.0
    best_ask_no_size: float = 0.0
    
    # Derived metrics
    spread_yes: float = 1.0
    spread_no: float = 1.0
    bundle_cost: float = 2.0  # best_ask_yes + best_ask_no
    bundle_cost_bid: float = 2.0  # best_bid_yes + best_bid_no
    cheap_yes: bool = False
    cheap_no: bool = False
    
    # Liquidity flag
    has_liquidity: bool = False
    has_arbitrage: bool = False
    
    # Opportunity score
    score: float = 0.0
    
    def update_book(
        self,
        bid_yes: float,
        ask_yes: float,
        bid_no: float,
        ask_no: float,
        bid_yes_size: float = 0.0,
        ask_yes_size: float = 0.0,
        bid_no_size: float = 0.0,
        ask_no_size: float = 0.0,
    ):
        """Update order book data"""
        self.best_bid_yes = bid_yes
        self.best_ask_yes = ask_yes
        self.best_bid_no = bid_no
        self.best_ask_no = ask_no
        self.best_bid_yes_size = bid_yes_size
        self.best_ask_yes_size = ask_yes_size
        self.best_bid_no_size = bid_no_size
        self.best_ask_no_size = ask_no_size
        self.spread_yes = ask_yes - bid_yes
        self.spread_no = ask_no - bid_no
        
        # Bundle cost for arbitrage calculation
        self.bundle_cost = ask_yes + ask_no
        self.bundle_cost_bid = bid_yes + bid_no
        
        # Check liquidity
        yes_ok = 0.02 < ask_yes < 0.98
        no_ok = 0.02 < ask_no < 0.98
        self.has_liquidity = yes_ok or no_ok
        
        # Cheap-side flags (bid-based)
        self.cheap_yes = bid_yes < BA_CHEAP_YES_BID
        self.cheap_no = bid_no < BA_CHEAP_NO_BID
        
        # Check arbitrage opportunity (ASK-based for taker strategy)
        # We BUY at asks, so bundle_cost (ask_yes + ask_no) is what we pay
        depth_yes = min(self.best_bid_yes_size, self.best_ask_yes_size)
        depth_no = min(self.best_bid_no_size, self.best_ask_no_size)
        depth_ok = depth_yes >= BA_DEPTH_MIN and depth_no >= BA_DEPTH_MIN
        
        self.has_arbitrage = (
            self.bundle_cost <= BA_MAX_BUNDLE_COST
            and depth_ok
            and self.best_ask_yes <= BA_MAX_YES_PRICE
            and self.best_ask_no <= BA_MAX_NO_PRICE
        )
    
    def to_orderbook_state(self) -> OrderBookState:
        """Convert to OrderBookState for BundleArbitrageur"""
        return OrderBookState(
            timestamp=time.time(),
            best_bid_yes=self.best_bid_yes,
            best_ask_yes=self.best_ask_yes,
            best_bid_yes_size=self.best_bid_yes_size,
            best_ask_yes_size=self.best_ask_yes_size,
            best_bid_no=self.best_bid_no,
            best_ask_no=self.best_ask_no,
            best_bid_no_size=self.best_bid_no_size,
            best_ask_no_size=self.best_ask_no_size,
        )


class MarketHopper:
    """
    Coordinates trading across multiple markets using Bundle Arbitrage.
    
    Core Logic (based on gabagool22's Dec 5 strategy):
    1. Monitor both BTC and ETH markets simultaneously
    2. Score each market based on bundle arbitrage opportunity
    3. Focus capital on the BEST opportunity
    4. Execute dual-side sweeps (buy BOTH YES and NO)
    """
    
    def __init__(self):
        # Market states
        self.markets: Dict[str, MarketState] = {}  # asset_label -> MarketState
        
        # Focus tracking
        self.current_focus: Optional[str] = None  # 'BTC' or 'ETH'
        self.last_switch_time: float = 0.0
        self.switch_count: int = 0
        
        # CLOB client
        self.clob_client: Optional[ClobClient] = None
        
        # Bundle Arbitrageur (the new strategy!)
        self.arbitrageur: Optional[BundleArbitrageur] = None
        
        # Execution
        self.stop_event = asyncio.Event()
        
        # Logging
        self.logger = TradeLogger(log_dir="data/hopper_logs")
        
        # Stats
        self.trades_by_market: Dict[str, int] = {'BTC': 0, 'ETH': 0}
        self.focus_time: Dict[str, float] = {'BTC': 0.0, 'ETH': 0.0}
        self.focus_start_time: float = 0.0
        self.total_sweeps: int = 0
        self.target_fills: int = 20
        self.token_side: Dict[str, Tuple[str, str]] = {}  # token_id -> (market_label, 'YES'/'NO')
        self.last_ws_update: Dict[str, float] = {}
        self.ws_task: Optional[asyncio.Task] = None
        self.ws_connected: bool = False
        self.last_resolution_check: Dict[str, float] = {}
        self.last_resolution_poll: float = 0.0
        self.last_market_refresh: float = 0.0
        self.needs_refresh: bool = False
        
        # Session tracking
        self.session_start: float = time.time()
        
        # Concurrency guard for reactive execution
        self._sweep_in_flight: bool = False
        
        # Diagnostic counters for debugging
        self._ws_message_count: int = 0
        self._ws_verbose_until: int = 5  # Log first N messages verbosely
    
    def _init_clob_client(self):
        """Initialize CLOB client with pre-warmed connections"""
        # Pre-warm HTTP connections first (eliminates cold-start latency)
        warm_connections()
        
        print("[Hopper] 🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=SIGNATURE_TYPE,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[Hopper] ✅ CLOB client authenticated (key: {creds.api_key[:12]}...)")
        
        # Initialize Bundle Arbitrageur with UPDATED config from VWAP analysis
        config = BundleConfig(
            max_bundle_cost=BA_MAX_BUNDLE_COST,
            target_bundle_cost=BA_TARGET_BUNDLE_COST,
            abort_bundle_cost=BA_ABORT_BUNDLE_COST,
            bid_bundle_threshold=BA_BID_BUNDLE_THRESHOLD,
            max_imbalance=BA_MAX_IMBALANCE,
            max_total_exposure=BA_MAX_TOTAL_EXPOSURE,
            max_yes_price=BA_MAX_YES_PRICE,
            max_no_price=BA_MAX_NO_PRICE,
            cheap_side_threshold=BA_CHEAP_SIDE_THRESHOLD,
            cheap_yes_bid=BA_CHEAP_YES_BID,
            cheap_no_bid=BA_CHEAP_NO_BID,
            order_size=BA_ORDER_SIZE,
            clip_ladder=BA_CLIP_LADDER,
            burst_child_count=BA_BURST_CHILD_COUNT,
            sweep_burst_size=BA_SWEEP_BURST_SIZE,
            sweep_interval_ms=BA_SWEEP_INTERVAL_MS,
            fills_per_sweep=BA_FILLS_PER_SWEEP,
            spread_min=BA_SPREAD_MIN,
            depth_min=BA_DEPTH_MIN,
            morning_bonus_start=BA_MORNING_BONUS_START,
            morning_bonus_end=BA_MORNING_BONUS_END,
            morning_bonus_mult=BA_MORNING_BONUS_MULT,
        )
        self.arbitrageur = BundleArbitrageur(config)
        self.arbitrageur.clob_client = self.clob_client
        self.arbitrageur.init_clob_client()
        
        print("[Hopper] 🐙 Bundle Arbitrage strategy initialized (VWAP-optimized)")
        print(f"[Hopper]    Max bundle cost: ${BA_MAX_BUNDLE_COST:.3f}")
        print(f"[Hopper]    Target fills/sweep: {BA_FILLS_PER_SWEEP}")
        print(f"[Hopper]    Max imbalance: {BA_MAX_IMBALANCE*100:.0f}%")
        print(f"[Hopper]    Max exposure: ${BA_MAX_TOTAL_EXPOSURE:.0f}")
        print(f"[Hopper]    Morning bonus: {BA_MORNING_BONUS_START}-{BA_MORNING_BONUS_END}h UTC (+{(BA_MORNING_BONUS_MULT-1)*100:.0f}%)")
    
    def _get_time_bonus(self) -> float:
        """
        Calculate time-based scoring bonus.
        
        From 25K trade VWAP analysis:
        - ETH 9AM: bundle VWAP $0.712 (+40% ROI)
        - ETH 10:30AM: bundle VWAP $0.830 (+20% ROI)
        - Early morning (9-11 UTC) has best arbitrage spreads
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        if BA_MORNING_BONUS_START <= hour <= BA_MORNING_BONUS_END:
            return BA_MORNING_BONUS_MULT
        return 1.0
    
    def calculate_opportunity_score(self, market: MarketState) -> float:
        """
        Calculate opportunity score for a market.
        
        UPDATED SCORING (from 25K trade VWAP analysis):
        - Primary: How much below $1.00 is the bundle cost?
        - Secondary: Individual price thresholds
        - Bonus: Time-based (early morning has better spreads)
        """
        if not market.has_liquidity:
            market.score = 0.0
            return 0.0
        
        # Primary: Bundle cost arbitrage opportunity
        # Lower bundle cost = higher score
        # Score = 0 at bundle_cost = 1.00, Score = 1 at bundle_cost = 0.90
        if market.bundle_cost >= 1.0:
            bundle_score = 0.0
        else:
            bundle_score = (1.0 - market.bundle_cost) * 10  # 0.98 -> 0.2, 0.95 -> 0.5
            bundle_score = min(1.0, bundle_score)
        
        # Secondary: Are individual prices within our thresholds?
        yes_ok = 1.0 if market.best_ask_yes <= BA_MAX_YES_PRICE else 0.5
        no_ok = 1.0 if market.best_ask_no <= BA_MAX_NO_PRICE else 0.5
        cheap_side = min(market.best_ask_yes, market.best_ask_no)
        cheap_bonus = 0.0
        if cheap_side < BA_CHEAP_SIDE_THRESHOLD:
            cheap_bonus = (BA_CHEAP_SIDE_THRESHOLD - cheap_side) / BA_CHEAP_SIDE_THRESHOLD
            cheap_bonus = min(1.0, max(0.0, cheap_bonus))
        
        # Weighted score
        score = (
            bundle_score * 0.6
            + (yes_ok * 0.1)
            + (no_ok * 0.1)
            + (cheap_bonus * 0.2)
        )
        
        # Apply time-based bonus (early morning = better spreads)
        time_bonus = self._get_time_bonus()
        score = score * time_bonus
        
        market.score = score
        return score
    
    def decide_focus(self) -> Optional[str]:
        """
        Decide which market to focus on.
        Based on bundle arbitrage opportunity.
        """
        if not self.markets:
            return None
        
        if len(self.markets) == 1:
            new_focus = list(self.markets.keys())[0]
            if new_focus != self.current_focus:
                self._switch_focus(new_focus)
            return new_focus
        
        # Calculate scores for all markets
        scores = {}
        for label, state in self.markets.items():
            scores[label] = self.calculate_opportunity_score(state)
        
        # Log scores with bundle costs
        score_str = " | ".join([
            f"{k}: {v:.2f} (bc=${self.markets[k].bundle_cost:.3f})" 
            for k, v in scores.items()
        ])
        print(f"[Hopper] 📊 Scores: {score_str}")
        
        # Find best market with arbitrage opportunity
        arb_markets = {k: v for k, v in scores.items() 
                       if self.markets[k].has_arbitrage and v > 0}
        
        if not arb_markets:
            # No arbitrage - check if any has liquidity at all
            liq_markets = {k: v for k, v in scores.items() 
                          if self.markets[k].has_liquidity}
            if liq_markets:
                best_label = max(liq_markets.items(), key=lambda x: x[1])[0]
            elif self.markets:
                best_label = list(self.markets.keys())[0]
            else:
                return self.current_focus
        else:
            best_label = max(arb_markets.items(), key=lambda x: x[1])[0]
        
        # Switch focus if different
        if best_label != self.current_focus:
            self._switch_focus(best_label)
        
        return self.current_focus
    
    def _switch_focus(self, new_focus: str):
        """Execute a focus switch"""
        now = time.time()
        
        # Update time tracking for old focus
        if self.current_focus and self.focus_start_time > 0:
            elapsed = now - self.focus_start_time
            self.focus_time[self.current_focus] = \
                self.focus_time.get(self.current_focus, 0) + elapsed
        
        # Log the switch
        old = self.current_focus or 'NONE'
        old_score = self.markets[old].score if old in self.markets else 0
        new_score = self.markets[new_focus].score
        old_bc = self.markets[old].bundle_cost if old in self.markets else 0
        new_bc = self.markets[new_focus].bundle_cost
        
        print(f"[Hopper] 🔄 SWITCH: {old} (bc=${old_bc:.3f}) → "
              f"{new_focus} (bc=${new_bc:.3f})")
        
        # Log to trade logger
        self.logger.log_market_switch(old, new_focus, old_score, new_score)
        
        # Update arbitrageur with new market tokens
        market = self.markets[new_focus]
        if self.arbitrageur:
            self.arbitrageur.set_market(
                market.yes_token,
                market.no_token,
                market.slug,
                market.condition_id,
            )
        
        # Update state
        self.current_focus = new_focus
        self.last_switch_time = now
        self.focus_start_time = now
        self.switch_count += 1
    
    async def initialize_markets(self):
        """Discover and initialize both BTC and ETH markets"""
        print("[Hopper] 🔍 Discovering markets...")
        # Clock sanity: print UTC and Zagreb time for debugging window alignment
        now_utc = datetime.now(timezone.utc)
        try:
            zagreb = now_utc.astimezone(timezone(timedelta(hours=1)))
            print(f"[Hopper] 🕒 Clock check: UTC {now_utc.isoformat()} | Zagreb {zagreb.isoformat()}")
        except Exception:
            print(f"[Hopper] 🕒 Clock check (UTC): {now_utc.isoformat()}")
        
        # Initialize CLOB client (only once)
        if not self.clob_client:
            self._init_clob_client()
        
        # Clear old markets for refresh
        self.markets.clear()
        self.token_side.clear()
        self.last_ws_update.clear()
        self.last_resolution_check.clear()
        
        # Get BTC market
        btc_markets = get_target_markets(
            slug_prefix=BTC_SLUG_PREFIX,
            asset_label='BTC',
            keywords=['btc', 'bitcoin']
        )
        if btc_markets:
            m = btc_markets[0]
            self.markets['BTC'] = MarketState(
                slug=m.get('market_slug', ''),
                condition_id=m.get('condition_id', ''),
                asset_label='BTC',
                question=m.get('question', '')[:60],
                end_date_iso=m.get('end_date_iso', ''),
            )
            print(f"[Hopper] ✅ BTC: {m.get('question', '')[:50]}...")
            print(f"[Hopper]    Expires: {m.get('end_date_iso', 'unknown')}")
            
            # Fetch token IDs
            await self._fetch_tokens('BTC')
        
        # Get ETH market
        eth_markets = get_target_markets(
            slug_prefix=ETH_SLUG_PREFIX,
            asset_label='ETH',
            keywords=['eth', 'ethereum']
        )
        if eth_markets:
            m = eth_markets[0]
            self.markets['ETH'] = MarketState(
                slug=m.get('market_slug', ''),
                condition_id=m.get('condition_id', ''),
                asset_label='ETH',
                question=m.get('question', '')[:60],
                end_date_iso=m.get('end_date_iso', ''),
            )
            print(f"[Hopper] ✅ ETH: {m.get('question', '')[:50]}...")
            print(f"[Hopper]    Expires: {m.get('end_date_iso', 'unknown')}")
            
            # Fetch token IDs
            await self._fetch_tokens('ETH')
        
        if not self.markets:
            raise ValueError("No markets discovered!")
        
        print(f"[Hopper] 📊 Monitoring {len(self.markets)} markets")
        print(f"[Hopper] 🐙 Strategy: BUNDLE ARBITRAGE")
        print(f"[Hopper]    Target: bundle_cost < ${BA_MAX_BUNDLE_COST:.2f}")
        await self._restart_ws_listener()
    
    async def _fetch_tokens(self, asset_label: str):
        """Fetch token IDs for a market with validation."""
        market = self.markets.get(asset_label)
        if not market or not self.clob_client:
            return
        
        try:
            market_data = self.clob_client.get_market(market.condition_id)
            
            # Validate market is accepting orders
            accepting = market_data.get('accepting_orders') if isinstance(market_data, dict) else getattr(market_data, 'accepting_orders', None)
            if accepting is False:
                print(f"[Hopper] ⚠️ {asset_label} market NOT accepting orders - skipping!")
                return
            print(f"[Hopper] ✅ {asset_label} market accepting_orders={accepting}")
            
            tokens = market_data.get('tokens', []) if isinstance(market_data, dict) else getattr(market_data, 'tokens', [])
            
            print(f"[Hopper] 🔍 {asset_label} market response: {len(tokens)} tokens found")
            
            for token in tokens:
                outcome = token.get('outcome', '') if isinstance(token, dict) else getattr(token, 'outcome', '')
                token_id = token.get('token_id', '') if isinstance(token, dict) else getattr(token, 'token_id', '')
                
                if outcome.upper() in ('YES', 'UP'):
                    market.yes_token = token_id
                elif outcome.upper() in ('NO', 'DOWN'):
                    market.no_token = token_id
            
            if market.yes_token and market.no_token:
                print(f"[Hopper] 🔑 {asset_label} tokens: YES={market.yes_token[:16]}... NO={market.no_token[:16]}...")
                self.token_side[market.yes_token] = (asset_label, 'YES')
                self.token_side[market.no_token] = (asset_label, 'NO')
            else:
                print(f"[Hopper] ⚠️ {asset_label} MISSING TOKENS!")
        except Exception as e:
            print(f"[Hopper] ⚠️ Failed to fetch {asset_label} tokens: {e}")

    async def _restart_ws_listener(self):
        """Restart websocket listener after markets refresh."""
        if not WS_URL:
            return
        if self.ws_task:
            self.ws_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.ws_task
        self.ws_task = asyncio.create_task(self._start_ws_listener())

    async def _ws_ping_loop(self, ws):
        """Send explicit PING messages every 10 seconds as per Polymarket docs."""
        try:
            while True:
                await ws.send_str("PING")
                await asyncio.sleep(10)
        except Exception:
            pass  # Connection closed
    
    async def _start_ws_listener(self):
        """Listen to Polymarket live orderbook updates for all tracked tokens."""
        if not WS_URL:
            return
        while not self.stop_event.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                        await self._subscribe_tokens(ws)
                        self.ws_connected = True
                        
                        # Start ping loop per Polymarket docs
                        ping_task = asyncio.create_task(self._ws_ping_loop(ws))
                        
                        try:
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    raw = msg.data.strip()
                                    # Handle non-JSON responses (like PONG)
                                    if raw in ("PONG", ""):
                                        continue
                                    try:
                                        parsed = json.loads(raw)
                                    except Exception:
                                        continue
                                    # Skip empty arrays
                                    if parsed == []:
                                        continue
                                    # Handle both single message (dict) and batch messages (list)
                                    messages = parsed if isinstance(parsed, list) else [parsed]
                                    for data in messages:
                                        if isinstance(data, dict):
                                            await self._handle_ws_message(data)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                        finally:
                            ping_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await ping_task
            except Exception as e:
                self.ws_connected = False
                print(f"[Hopper] ⚠️ WS error: {e} (reconnecting in 1s)")
                await asyncio.sleep(1)
            finally:
                self.ws_connected = False
        self.ws_connected = False

    async def _subscribe_tokens(self, ws):
        """Subscribe to orderbook updates for all known tokens."""
        tokens = list(self.token_side.keys())
        if not tokens:
            return
        # Per Polymarket docs: https://docs.polymarket.com/quickstart/websocket/WSS-Quickstart
        payload = {
            "assets_ids": tokens,
            "type": "market",
        }
        try:
            await ws.send_json(payload)
            print(f"[Hopper] 🔗 WS subscribed to {len(tokens)} tokens (market channel)")
        except Exception as e:
            print(f"[Hopper] ⚠️ WS subscribe failed: {e}")

    @staticmethod
    def _extract_top(side_data, default_price: float, default_size: float, *, is_ask: bool) -> Tuple[float, float]:
        """
        Parse best-of-book price/size. Some endpoints return levels in descending order;
        compute min(ask) / max(bid) defensively.
        """
        if not side_data:
            return default_price, default_size

        def _ps(x):
            try:
                if isinstance(x, dict):
                    return float(x.get("price", default_price)), float(x.get("size", default_size))
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    return float(x[0]), float(x[1])
                return float(getattr(x, "price", default_price)), float(getattr(x, "size", default_size))
            except Exception:
                return default_price, default_size

        best_price = None
        best_size = default_size
        for lvl in side_data:
            p, s = _ps(lvl)
            if best_price is None:
                best_price, best_size = p, s
            else:
                if is_ask and p < best_price:
                    best_price, best_size = p, s
                if (not is_ask) and p > best_price:
                    best_price, best_size = p, s

        return (best_price if best_price is not None else default_price, best_size)

    async def _handle_ws_message(self, data: Dict):
        """Handle a single websocket orderbook delta."""
        self._ws_message_count += 1
        
        event_type = data.get("event_type")
        
        # Diagnostic: log first few messages verbosely
        if self._ws_message_count <= self._ws_verbose_until:
            print(f"[Hopper] 📨 WS msg #{self._ws_message_count}: event={event_type} keys={list(data.keys())[:6]}")
        
        # Per Polymarket docs: https://docs.polymarket.com/developers/CLOB/websocket/market-channel
        # Filter by event_type: only process 'book' (full snapshot) and 'price_change' (delta)
        if event_type not in ("book", "price_change", None):  # None for backwards compat
            if self._ws_message_count <= self._ws_verbose_until:
                print(f"[Hopper]    ↳ skipping event_type={event_type}")
            return
        
        # ============================================================
        # PRICE_CHANGE EVENTS: Contains nested 'price_changes' array
        # Structure: {"market": "0x...", "price_changes": [{"asset_id": "...", "price": "0.5", "side": "BUY", "size": "100"}, ...]}
        # See: https://docs.polymarket.com/developers/CLOB/websocket/market-channel
        # ============================================================
        if event_type == "price_change":
            price_changes = data.get("price_changes", [])
            if price_changes:
                for change in price_changes:
                    await self._process_price_change(change)
                return
            # Fallback: some price_change events may have flat structure
            # (single price update at top level)
            if data.get("asset_id") and data.get("price"):
                await self._process_price_change(data)
            return
        
        # ============================================================
        # BOOK EVENTS: Full orderbook snapshot with 'asset_id' at top level
        # Structure: {"asset_id": "...", "bids": [...], "asks": [...]}
        # ============================================================
        token_id = (
            data.get("asset_id") or 
            data.get("id") or 
            data.get("token_id") or
            data.get("marketId") or 
            data.get("tokenId")
        )
        if not token_id or token_id not in self.token_side:
            if self._ws_message_count <= self._ws_verbose_until:
                print(f"[Hopper]    ↳ token_id not tracked: {str(token_id)[:20] if token_id else 'None'}...")
            return

        bids = data.get("bids") or data.get("bid") or data.get("buy") or []
        asks = data.get("asks") or data.get("ask") or data.get("sell") or []

        bid_price, bid_size = self._extract_top(bids, default_price=0.01, default_size=0.0, is_ask=False)
        ask_price, ask_size = self._extract_top(asks, default_price=0.99, default_size=0.0, is_ask=True)
        
        # Diagnostic: log parsed prices for first few messages
        if self._ws_message_count <= self._ws_verbose_until:
            print(f"[Hopper]    ↳ parsed: bid=${bid_price:.3f} ask=${ask_price:.3f} (bids={len(bids)}, asks={len(asks)})")

        label, side = self.token_side[token_id]
        market = self.markets.get(label)
        if not market:
            return

        # Update the correct side while preserving the other side's latest values.
        if side == 'YES':
            market.update_book(
                bid_yes=bid_price,
                ask_yes=ask_price,
                bid_no=market.best_bid_no,
                ask_no=market.best_ask_no,
                bid_yes_size=bid_size,
                ask_yes_size=ask_size,
                bid_no_size=market.best_bid_no_size,
                ask_no_size=market.best_ask_no_size,
            )
        else:
            market.update_book(
                bid_yes=market.best_bid_yes,
                ask_yes=market.best_ask_yes,
                bid_no=bid_price,
                ask_no=ask_price,
                bid_yes_size=market.best_bid_yes_size,
                ask_yes_size=market.best_ask_yes_size,
                bid_no_size=bid_size,
                ask_no_size=ask_size,
            )

        self.last_ws_update[label] = time.time()
        
        # 📊 Log interesting bundle prices for opportunity analysis
        if market.bundle_cost <= 1.05 and self.arbitrageur:  # Only log interesting prices
            self.arbitrageur.event_logger.log({
                "event_type": "ws_bundle_update",
                "label": label,
                "bundle_cost_ask": market.bundle_cost,
                "ask_yes": market.best_ask_yes,
                "ask_no": market.best_ask_no,
                "is_arb": market.has_arbitrage,
                "is_profitable": market.bundle_cost < 1.00,
            })
        
        # 🚀 REACTIVE TRADING: Execute immediately if arbitrage detected
        # This removes the 0.5s polling lag
        # Guard against concurrent sweeps which could corrupt position tracking
        if market.has_arbitrage and self.current_focus == label and not self._sweep_in_flight:
            self._sweep_in_flight = True
            asyncio.create_task(self._guarded_execute_arb())

    async def _process_price_change(self, change: Dict):
        """
        Process a single price_change item from the nested 'price_changes' array.
        
        Structure per Polymarket docs:
        {
            "asset_id": "34474299...",  # Token ID (decimal string)
            "price": "0.42",            # New price level
            "side": "BUY",              # BUY = bid, SELL = ask
            "size": "100"               # Size at this level (may be 0 for removal)
        }
        
        Note: price_change events are INCREMENTAL updates, not full snapshots.
        We update only the specific level that changed.
        """
        token_id = change.get("asset_id") or change.get("id") or change.get("token_id")
        if not token_id or token_id not in self.token_side:
            return
        
        try:
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "").upper()
        except (ValueError, TypeError):
            return
        
        if price <= 0:
            return
        
        label, token_side = self.token_side[token_id]
        market = self.markets.get(label)
        if not market:
            return
        
        # Diagnostic logging
        if self._ws_message_count <= self._ws_verbose_until:
            print(f"[Hopper]    ↳ price_change: {token_side} {side}=${price:.3f} size={size:.0f}")
        
        # Update the appropriate price level
        # BUY = bid side, SELL = ask side
        if token_side == 'YES':
            if side == "BUY":
                # Update bid if this is a better price or significant size
                if price > market.best_bid_yes or size > 0:
                    market.update_book(
                        bid_yes=price if size > 0 else market.best_bid_yes,
                        ask_yes=market.best_ask_yes,
                        bid_no=market.best_bid_no,
                        ask_no=market.best_ask_no,
                        bid_yes_size=size if size > 0 else market.best_bid_yes_size,
                        ask_yes_size=market.best_ask_yes_size,
                        bid_no_size=market.best_bid_no_size,
                        ask_no_size=market.best_ask_no_size,
                    )
            else:  # SELL = ask
                # Update ask if this is a better price or significant size
                if price < market.best_ask_yes or size > 0:
                    market.update_book(
                        bid_yes=market.best_bid_yes,
                        ask_yes=price if size > 0 else market.best_ask_yes,
                        bid_no=market.best_bid_no,
                        ask_no=market.best_ask_no,
                        bid_yes_size=market.best_bid_yes_size,
                        ask_yes_size=size if size > 0 else market.best_ask_yes_size,
                        bid_no_size=market.best_bid_no_size,
                        ask_no_size=market.best_ask_no_size,
                    )
        else:  # NO side
            if side == "BUY":
                if price > market.best_bid_no or size > 0:
                    market.update_book(
                        bid_yes=market.best_bid_yes,
                        ask_yes=market.best_ask_yes,
                        bid_no=price if size > 0 else market.best_bid_no,
                        ask_no=market.best_ask_no,
                        bid_yes_size=market.best_bid_yes_size,
                        ask_yes_size=market.best_ask_yes_size,
                        bid_no_size=size if size > 0 else market.best_bid_no_size,
                        ask_no_size=market.best_ask_no_size,
                    )
            else:  # SELL = ask
                if price < market.best_ask_no or size > 0:
                    market.update_book(
                        bid_yes=market.best_bid_yes,
                        ask_yes=market.best_ask_yes,
                        bid_no=market.best_bid_no,
                        ask_no=price if size > 0 else market.best_ask_no,
                        bid_yes_size=market.best_bid_yes_size,
                        ask_yes_size=market.best_ask_yes_size,
                        bid_no_size=market.best_bid_no_size,
                        ask_no_size=size if size > 0 else market.best_ask_no_size,
                    )
        
        self.last_ws_update[label] = time.time()
        
        # 🚀 REACTIVE TRADING: Check for arbitrage after price update
        if market.has_arbitrage and self.current_focus == label and not self._sweep_in_flight:
            self._sweep_in_flight = True
            asyncio.create_task(self._guarded_execute_arb())

        # Opportunistically check resolution occasionally (no more than once every 60s per market)
        now = time.time()
        if ENABLE_ONCHAIN_REDEEM and (now - self.last_resolution_check.get(label, 0) > 60):
            self.last_resolution_check[label] = now
            asyncio.create_task(self._maybe_redeem_if_resolved(label))

    def _is_book_stale(self, label: Optional[str], stale_after: float = 2.0) -> bool:
        """Detect if websocket data is stale for a market label."""
        if not label:
            return True
        ts = self.last_ws_update.get(label)
        if ts is None:
            return True
        return (time.time() - ts) > stale_after

    async def _maybe_redeem_if_resolved(self, label: str):
        """
        Check if a market is resolved and submit redemption for the winning index set.
        Uses CLOB market metadata to detect resolution and winner; falls back to skip if unknown.
        """
        if not ENABLE_ONCHAIN_REDEEM or not self.clob_client or not self.arbitrageur:
            return
        market = self.markets.get(label)
        if not market or not market.condition_id:
            return
        try:
            market_data = self.clob_client.get_market(market.condition_id)
        except Exception as e:
            print(f"[Hopper] ⚠️ Resolution check failed for {label}: {e}")
            return

        resolved, winner_index = self._extract_resolution(market_data)
        if not resolved or winner_index is None:
            return

        try:
            await self.arbitrageur.redeem_winning_tokens(market.condition_id, [winner_index])
        except Exception as e:
            print(f"[Hopper] ⚠️ Redemption failed for {label}: {e}")

    @staticmethod
    def _extract_resolution(market_data) -> Tuple[bool, Optional[int]]:
        """
        Best-effort resolution parsing:
        - Expect a 'resolved' flag or similar.
        - Try to infer winning index from token payouts.
        """
        if not market_data:
            return False, None

        resolved = bool(market_data.get("resolved") or market_data.get("isResolved"))
        if not resolved:
            # Fallback: check payout numerators or winning outcome hints
            payout_nums = market_data.get("payoutNumerators") or market_data.get("payout_numerators")
            if isinstance(payout_nums, list) and any(payout_nums):
                try:
                    idx = max(range(len(payout_nums)), key=lambda i: payout_nums[i])
                    return True, idx
                except Exception:
                    pass
            return False, None

        tokens = market_data.get("tokens", []) if isinstance(market_data, dict) else getattr(market_data, "tokens", [])
        winner_index = None
        for token in tokens:
            outcome = (token.get("outcome") or token.get("label") or "").upper()
            payout = token.get("payout") or token.get("payoutNumerator") or token.get("payout_num") or 0
            # Consider payout > 0 as winner
            if payout and float(payout) > 0:
                if outcome in ("YES", "UP"):
                    winner_index = 1
                elif outcome in ("NO", "DOWN"):
                    winner_index = 2
                else:
                    # fallback to index_set if present
                    idx = token.get("index_set") or token.get("indexSet")
                    if idx:
                        winner_index = int(idx)
                if winner_index:
                    break
        
        # If still unknown, check payoutNumerators as last resort
        if winner_index is None:
            payout_nums = market_data.get("payoutNumerators") or market_data.get("payout_numerators")
            if isinstance(payout_nums, list) and any(payout_nums):
                try:
                    winner_index = max(range(len(payout_nums)), key=lambda i: payout_nums[i])
                except Exception:
                    winner_index = None

        return resolved, winner_index
    
    async def poll_market_data(self):
        """Poll REST APIs for market data (sequential to avoid rate limits)"""
        for label, market in self.markets.items():
            if not market.yes_token or not market.no_token:
                continue
            
            try:
                # Fetch order books
                yes_book = self.clob_client.get_order_book(market.yes_token)
                no_book = self.clob_client.get_order_book(market.no_token)
                
                # Extract best prices (defensive: min ask, max bid)
                def best(side, is_ask, default_price, default_size):
                    if not side:
                        return default_price, default_size
                    best_price = None
                    best_size = default_size
                    for lvl in side:
                        p = float(getattr(lvl, "price", getattr(lvl, "p", default_price)))
                        s = float(getattr(lvl, "size", getattr(lvl, "s", default_size)))
                        if best_price is None:
                            best_price, best_size = p, s
                        else:
                            if is_ask and p < best_price:
                                best_price, best_size = p, s
                            if (not is_ask) and p > best_price:
                                best_price, best_size = p, s
                    return best_price, best_size

                bid_yes, bid_yes_size = best(yes_book.bids, is_ask=False, default_price=0.01, default_size=0.0)
                ask_yes, ask_yes_size = best(yes_book.asks, is_ask=True, default_price=0.99, default_size=0.0)
                bid_no, bid_no_size = best(no_book.bids, is_ask=False, default_price=0.01, default_size=0.0)
                ask_no, ask_no_size = best(no_book.asks, is_ask=True, default_price=0.99, default_size=0.0)
                
                # Spread sanity check - warn if spreads are suspiciously large (degenerate book)
                spread_yes = ask_yes - bid_yes
                spread_no = ask_no - bid_no
                if spread_yes >= 0.90 or spread_no >= 0.90:
                    print(f"[Hopper] ⚠️ {label} DEGENERATE BOOK DETECTED!")
                    print(f"[Hopper]    YES: bid=${bid_yes:.2f} ask=${ask_yes:.2f} spread=${spread_yes:.2f}")
                    print(f"[Hopper]    NO:  bid=${bid_no:.2f} ask=${ask_no:.2f} spread=${spread_no:.2f}")
                    print(f"[Hopper]    This usually means the market is resolved/expired or has no liquidity")
                    self.needs_refresh = True
                
                # Update market state
                market.update_book(
                    bid_yes, ask_yes, bid_no, ask_no,
                    bid_yes_size, ask_yes_size, bid_no_size, ask_no_size
                )
                
            except Exception as e:
                err_str = str(e)
                if "404" in err_str:
                    print(f"[Hopper] 🔴 {label} Token not found - need market refresh")
                    self.needs_refresh = True
                else:
                    print(f"[Hopper] ⚠️ {label} poll error: {e}")
    
    async def _guarded_execute_arb(self):
        """Wrapper for reactive execution with concurrency guard."""
        try:
            await self.execute_bundle_arbitrage()
        finally:
            self._sweep_in_flight = False
    
    async def execute_bundle_arbitrage(self):
        """Execute bundle arbitrage on the focused market"""
        if not self.current_focus or not self.arbitrageur:
            return
        
        market = self.markets.get(self.current_focus)
        if not market:
            return
        
        # Update arbitrageur with current book state
        book_state = market.to_orderbook_state()
        self.arbitrageur.update_book(book_state)
        
        # Ensure arbitrageur has correct market tokens
        if (self.arbitrageur.yes_token != market.yes_token or 
            self.arbitrageur.no_token != market.no_token):
            self.arbitrageur.set_market(
                market.yes_token,
                market.no_token,
                market.slug,
                market.condition_id,
            )
        
        # Execute sweep!
        filled = await self.arbitrageur.execute_sweep()
        
        if filled:
            self.total_sweeps += 1
            self.trades_by_market[self.current_focus] += 1
    
    async def run(self, target_fills: int = 20):
        """Main execution loop"""
        self.target_fills = target_fills
        
        await self.initialize_markets()
        
        # Initial poll and focus decision
        await self.poll_market_data()
        self.decide_focus()
        
        print(f"\n[Hopper] 🎯 Initial focus: {self.current_focus}")
        print(f"[Hopper] 🎯 Strategy: BUNDLE ARBITRAGE")
        print(f"[Hopper]    Looking for bundle_cost < ${BA_MAX_BUNDLE_COST:.2f}")
        print(f"[Hopper] Starting trading loop...\n")
        
        iteration = 0
        while not self.stop_event.is_set():
            try:
                # Check if any market window is expiring soon
                if self._should_refresh_markets():
                    print("[Hopper] ⏰ Market window expiring, refreshing markets...")
                    await asyncio.sleep(5)
                    await self.initialize_markets()
                    await self.poll_market_data()
                    self.decide_focus()
                    print("[Hopper] ✅ Markets refreshed for new window")
                    self.last_market_refresh = time.time()
                    self.needs_refresh = False
                    continue
                
                # AGGRESSIVE refresh for 15-minute windows
                now = time.time()
                refresh_interval = 60  # Check every 1 minute for 15m windows
                
                # Immediate refresh if needs_refresh is set (404 detected, degenerate book, etc.)
                if self.needs_refresh:
                    print("[Hopper] 🔴 IMMEDIATE REFRESH triggered (404 or degenerate book detected)")
                    await asyncio.sleep(1)  # Brief pause before refresh
                    await self.initialize_markets()
                    await self.poll_market_data()
                    self.decide_focus()
                    self.last_market_refresh = now
                    self.needs_refresh = False
                    self._ws_message_count = 0  # Reset WS diagnostics
                    continue
                
                # Periodic refresh guard
                if (now - self.last_market_refresh > refresh_interval):
                    # Check for stale/degenerate markets
                    if any(self._market_needs_redisco(m) for m in self.markets.values()):
                        print("[Hopper] 🔄 Refreshing markets (periodic check found stale/degenerate)...")
                        await asyncio.sleep(1)
                        await self.initialize_markets()
                        await self.poll_market_data()
                        self.decide_focus()
                        self.last_market_refresh = now
                        self.needs_refresh = False
                        continue
                
                # Poll market data; if WS is down or stale, fall back to faster REST polling
                ws_stale = (not self.ws_connected) or self._is_book_stale(self.current_focus, stale_after=1.0)
                if ws_stale:
                    await self.poll_market_data()
                
                rest_sleep = 0.1 if ws_stale else 0.5
                
                # Log market state periodically
                if iteration % 10 == 0:
                    self._log_market_state()
                
                # Decide focus
                self.decide_focus()
                
                # Execute bundle arbitrage
                await self.execute_bundle_arbitrage()

                # Periodic resolution poller to trigger redemption if markets closed
                now = time.time()
                if now - self.last_resolution_poll > 60:
                    self.last_resolution_poll = now
                    for label in list(self.markets.keys()):
                        await self._maybe_redeem_if_resolved(label)
                
                # Check if we've hit exposure limit
                if self.arbitrageur and self.arbitrageur.position.total_exposure >= BA_MAX_TOTAL_EXPOSURE:
                    print(f"[Hopper] 💰 Max exposure reached (${BA_MAX_TOTAL_EXPOSURE})")
                    break
                
                # Brief sleep between iterations
                await asyncio.sleep(rest_sleep)
                iteration += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Hopper] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
        
        print(f"\n[Hopper] 🏁 Session ended")
        if self.ws_task:
            self.ws_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.ws_task
        self._print_stats()
    
    def _should_refresh_markets(self) -> bool:
        """Check if any market window is about to expire"""
        now = datetime.now(timezone.utc)
        
        for label, market in self.markets.items():
            if not market.end_date_iso:
                continue
            
            try:
                end_str = market.end_date_iso.replace('Z', '+00:00')
                end_time = datetime.fromisoformat(end_str)
                time_remaining = (end_time - now).total_seconds()
                
                if time_remaining < 30:
                    print(f"[Hopper] ⚠️ {label} market expires in {time_remaining:.0f}s")
                    return True
                    
            except Exception as e:
                print(f"[Hopper] ⚠️ Failed to parse end_date for {label}: {e}")
        
        return False

    def _market_needs_redisco(self, market: MarketState) -> bool:
        """
        Detect stale/invalid markets needing rediscovery:
        - Expired window (end_date <= now)
        - Degenerate book shape (0 bid, ~1 ask on both sides)
        """
        now = datetime.now(timezone.utc)
        # Expired
        if market.end_date_iso:
            try:
                end_str = market.end_date_iso.replace('Z', '+00:00')
                end_time = datetime.fromisoformat(end_str)
                if end_time <= now:
                    return True
            except Exception:
                pass
        # Degenerate book
        if (
            market.best_bid_yes == 0 and market.best_bid_no == 0 and
            market.best_ask_yes >= 0.99 and market.best_ask_no >= 0.99
        ):
            return True
        return False
    
    def _log_market_state(self):
        """Log current market state"""
        for label, market in self.markets.items():
            arb = "🎯 ARB!" if market.has_arbitrage else "❌ no-arb"
            focus = "👉" if label == self.current_focus else "  "
            print(f"[Hopper] {focus} {label}: Bundle=${market.bundle_cost:.3f} (Y=${market.best_ask_yes:.3f} N=${market.best_ask_no:.3f}) | {arb}")
        
        # Print position and P&L
        if self.arbitrageur:
            pos = self.arbitrageur.position
            print(f"[Hopper]    Position: {pos.yes_shares:.0f}Y/{pos.no_shares:.0f}N | "
                  f"Bundle=${pos.bundle_cost:.3f} | P&L=${pos.total_pnl:+.2f}")
    
    def _print_stats(self):
        """Print session statistics"""
        # Update final focus time
        if self.current_focus and self.focus_start_time > 0:
            elapsed = time.time() - self.focus_start_time
            self.focus_time[self.current_focus] = \
                self.focus_time.get(self.current_focus, 0) + elapsed
        
        session_duration = time.time() - self.session_start
        total_time = sum(self.focus_time.values())
        
        print("\n" + "="*65)
        print("🐙 BUNDLE ARBITRAGE SESSION STATS")
        print("="*65)
        print(f"Session Duration: {session_duration/60:.1f} minutes")
        print(f"Market Switches:  {self.switch_count}")
        print(f"Total Sweeps:     {self.total_sweeps}")
        print("-"*65)
        print("Focus Time:")
        for market, t in self.focus_time.items():
            pct = (t / total_time * 100) if total_time > 0 else 0
            print(f"  {market}: {t:.1f}s ({pct:.1f}%)")
        print("-"*65)
        
        # Print arbitrageur stats
        if self.arbitrageur:
            self.arbitrageur.print_summary()
        
        # Save logger stats
        self.logger.save_summary()


# Factory function
def create_hopper() -> MarketHopper:
    """Factory function to create a MarketHopper instance"""
    return MarketHopper()
