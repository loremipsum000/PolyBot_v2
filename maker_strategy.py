import asyncio
import os
import time
import json
import threading
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from config import (
    PRIVATE_KEY,
    POLYGON_ADDRESS,
    SIGNATURE_TYPE,
    BTC_SLUG_PREFIX,
    ETH_SLUG_PREFIX,
    WS_URL,
    CHAIN_ID,
    RPC_URL,
    PAPER_TRADING,
)
from utils import get_target_markets, warm_connections, get_http_session
from binance_client import BinanceFeed
from binance_ohlc import get_open_price_at_ts
from pricing_engine import calculate_fair_probability, get_time_to_expiry

# Load env so MAKER_* are populated
load_dotenv()

# --- MAKER CONFIGURATION ---
MAKER_MAX_BUNDLE_COST = float(os.getenv("MAKER_MAX_BUNDLE_COST", "0.99"))  # Max combined cost we are willing to pay
MAKER_ORDER_SIZE = float(os.getenv("MAKER_ORDER_SIZE", "25.0"))          # Size of our limit orders (default halved)
MAKER_MIN_SPREAD = float(os.getenv("MAKER_MIN_SPREAD", "0.01"))          # Min spread we want to capture
MAKER_TICK_SIZE = 0.01                                                   # Polymarket tick size
MAKER_REFRESH_INTERVAL = 2.0                                             # How often to re-evaluate orders (seconds)
MAKER_PRICE_DELTA = float(os.getenv("MAKER_PRICE_DELTA", "0.01"))        # Move/cancel threshold
MAKER_VOL = float(os.getenv("MAKER_VOL", "0.6"))                         # Annualized vol for fair pricing

@dataclass
class MarketData:
    slug: str
    condition_id: str
    yes_token: str
    no_token: str
    end_date_iso: str = ""
    strike_price: Optional[float] = None
    best_bid_yes: float = 0.0
    best_ask_yes: float = 0.0
    best_bid_no: float = 0.0
    best_ask_no: float = 0.0
    
    @property
    def mid_price_yes(self) -> float:
        if self.best_bid_yes > 0 and self.best_ask_yes < 1:
            return (self.best_bid_yes + self.best_ask_yes) / 2
        return 0.5
        
    @property
    def implied_probability(self) -> float:
        # Estimate true probability from both sides
        # P_yes + P_no = 1
        # Mid_yes / (Mid_yes + Mid_no)
        mid_yes = self.mid_price_yes
        mid_no = (self.best_bid_no + self.best_ask_no) / 2 if (self.best_bid_no > 0 and self.best_ask_no < 1) else 0.5
        total = mid_yes + mid_no
        if total == 0: return 0.5
        return mid_yes / total


@dataclass
class QuoteContext:
    ref_price: Optional[float] = None
    strike: Optional[float] = None
    t_years: Optional[float] = None
    fair_prob: Optional[float] = None
    vol: float = MAKER_VOL
    bid_yes_raw: float = 0.0
    bid_no_raw: float = 0.0
    bid_yes: float = 0.0
    bid_no: float = 0.0
    bundle_sum_raw: float = 0.0
    bundle_sum_capped: float = 0.0
    adjustments: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class MakerJsonLogger:
    """
    Comprehensive JSONL logger for maker events.
    Writes to data/maker_logs/maker_session_<ts>.jsonl
    """

    def __init__(self, log_dir: str = "data/maker_logs", *, paper_trading: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"maker_session_{self.session_id}.jsonl"
        self._lock = threading.Lock()
        self.paper_trading = paper_trading
        self.log({"event_type": "session_start"})
        self.log({
            "event_type": "session_config",
            "maker_params": {
                "max_bundle_cost": MAKER_MAX_BUNDLE_COST,
                "order_size": MAKER_ORDER_SIZE,
                "min_spread": MAKER_MIN_SPREAD,
                "tick_size": MAKER_TICK_SIZE,
                "refresh_interval": MAKER_REFRESH_INTERVAL,
                "price_delta": MAKER_PRICE_DELTA,
                "vol": MAKER_VOL,
            },
        })

    def log(self, record: Dict):
        rec = record.copy()
        rec.setdefault("ts", int(time.time()))
        rec.setdefault("session_id", self.session_id)
        rec.setdefault("paper_trading", self.paper_trading)
        try:
            with self._lock:
                with open(self.session_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[MakerLog] write error: {e}")

class MakerStrategy:
    def __init__(self):
        self.clob_client: Optional[ClobClient] = None
        self.markets: Dict[str, MarketData] = {}  # 'BTC', 'ETH' -> MarketData
        self.open_orders: Dict[str, List[str]] = {} # token_id -> list of order_ids
        self.last_quotes: Dict[str, float] = {}     # token_id -> last quoted price
        self.paper_trading: bool = PAPER_TRADING
        # Simple inventory tracking per asset
        self.position: Dict[str, Dict[str, float]] = {  # label -> {'YES': sh, 'NO': sh}
            'BTC': {'YES': 0.0, 'NO': 0.0},
            'ETH': {'YES': 0.0, 'NO': 0.0},
        }
        self.running = True
        self.binance_feed: Optional[BinanceFeed] = None
        self.event_logger = MakerJsonLogger(log_dir="data/maker_logs", paper_trading=self.paper_trading)

    def init_client(self):
        print("[Maker] 🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=SIGNATURE_TYPE,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[Maker] ✅ Authenticated (Addr: {POLYGON_ADDRESS})")
        if self.paper_trading:
            print("[Maker] 🧪 PAPER TRADING MODE: orders will NOT be sent to CLOB.")

    async def discover_markets(self):
        print("[Maker] 🔍 Discovering 15m markets...")
        self.markets = {}
        
        for label, prefix in [('BTC', BTC_SLUG_PREFIX), ('ETH', ETH_SLUG_PREFIX)]:
            found = get_target_markets(slug_prefix=prefix, asset_label=label)
            if not found:
                print(f"[Maker] ⚠️ No {label} market found.")
                continue
                
            m = found[0]
            print(f"[Maker] ✅ Found {label}: {m.get('question')}")
            self.event_logger.log({
                "event_type": "market_found",
                "label": label,
                "market_slug": m.get('market_slug'),
                "condition_id": m.get('condition_id'),
                "question": m.get('question'),
            })
            
            # Get Token IDs and ensure market is tradable on CLOB
            try:
                c_id = m.get('condition_id')
                market_resp = self.clob_client.get_market(c_id)
                accepting = market_resp.get('accepting_orders') if isinstance(market_resp, dict) else getattr(market_resp, 'accepting_orders', None)
                closed = market_resp.get('closed') if isinstance(market_resp, dict) else getattr(market_resp, 'closed', None)
                if accepting is False or closed is True:
                    print(f"[Maker] ❌ {label} market not accepting orders or closed")
                    continue
                tokens = market_resp.get('tokens', []) if isinstance(market_resp, dict) else market_resp.tokens
                
                yes_id = ""
                no_id = ""
                for t in tokens:
                    outcome = (t.get('outcome') or t.get('label') or "").upper()
                    tid = t.get('token_id')
                    if outcome in ['YES', 'UP']: yes_id = tid
                    elif outcome in ['NO', 'DOWN']: no_id = tid
                
                if yes_id and no_id:
                    end_iso = m.get('end_date_iso') or m.get('endDate') or ""
                    start_ts = None
                    if end_iso:
                        try:
                            from datetime import datetime, timezone, timedelta
                            end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                            start_ts = int((end - timedelta(minutes=15)).timestamp())
                        except Exception:
                            start_ts = None

                    strike = None
                    if start_ts:
                        symbol = "BTCUSDT" if label == "BTC" else "ETHUSDT"
                        strike = get_open_price_at_ts(symbol, start_ts)

                    # Fallback: if we started late and could not fetch the window open,
                    # seed strike with the current Binance price so we can still quote.
                    if strike is None and self.binance_feed:
                        live_price = self.binance_feed.get_price(label)
                        if live_price:
                            strike = live_price
                            print(f"[Maker] ⚠️ {label}: Missing strike; falling back to live price {strike:.2f}")

                    # Quick sanity check: ensure orderbooks exist before trading
                    try:
                        self.clob_client.get_order_book(yes_id)
                        self.clob_client.get_order_book(no_id)
                    except Exception:
                        print(f"[Maker] ⚠️ {label}: Skipping - token orderbook not available (404)")
                        self.event_logger.log({
                            "event_type": "market_skip",
                            "label": label,
                            "reason": "orderbook_missing",
                            "yes_token": yes_id,
                            "no_token": no_id,
                        })
                        continue

                    self.markets[label] = MarketData(
                        slug=m.get('market_slug'),
                        condition_id=c_id,
                        yes_token=yes_id,
                        no_token=no_id,
                        strike_price=strike,
                        end_date_iso=end_iso
                    )
                    print(f"[Maker]    Tokens: YES={yes_id[:10]}... NO={no_id[:10]}... Strike={strike}")
                else:
                    print(f"[Maker] ❌ Could not identify tokens for {label}")
                    self.event_logger.log({
                        "event_type": "market_error",
                        "label": label,
                        "reason": "token_id_missing",
                    })
                    
            except Exception as e:
                print(f"[Maker] ❌ Error fetching details for {label}: {e}")
                self.event_logger.log({
                    "event_type": "market_error",
                    "label": label,
                    "reason": str(e),
                })

    async def cancel_all_orders(self, token_ids: List[str]):
        """Cancel open orders for provided token_ids by tracked order IDs."""
        if not token_ids or not self.clob_client:
            return

        order_ids: List[str] = []
        for tid in token_ids:
            order_ids.extend(self.open_orders.get(tid, []))
            self.open_orders.pop(tid, None)

        # In paper mode, just clear and log.
        if self.paper_trading:
            if order_ids:
                self.event_logger.log({
                    "event_type": "paper_cancel",
                    "order_ids": order_ids,
                    "token_ids": token_ids,
                })
            return

        if not order_ids:
            return

        # Prefer batch cancel; fall back to per-order cancel variants.
        try:
            if hasattr(self.clob_client, "cancel_orders"):
                self.clob_client.cancel_orders(order_ids=order_ids)
                return
        except Exception:
            pass
        
        for oid in order_ids:
            try:
                if hasattr(self.clob_client, "cancel_order"):
                    self.clob_client.cancel_order(order_id=oid)
                elif hasattr(self.clob_client, "cancel"):
                    # Some client versions expose `cancel(order_id=...)`
                    self.clob_client.cancel(order_id=oid)
                else:
                    raise AttributeError("No cancel method available on client")
            except Exception as e:
                print(f"[Maker] ⚠️ Cancel fallback failed for {oid}: {e}")

    def _should_replace(self, token_id: str, new_price: float) -> bool:
        """Decide if we should replace an existing quote based on price delta."""
        if new_price <= 0:
            return False
        prev = self.last_quotes.get(token_id)
        if prev is None:
            return True
        return abs(prev - new_price) >= MAKER_PRICE_DELTA

    def _record_quote(self, token_id: str, price: float, order_id: Optional[str] = None):
        if price > 0:
            self.last_quotes[token_id] = price
        if order_id:
            self.open_orders.setdefault(token_id, []).append(order_id)

    async def update_market_books(self):
        """Fetch latest orderbook depth"""
        to_remove: List[str] = []

        for label, market in list(self.markets.items()):
            try:
                yes_book = self.clob_client.get_order_book(market.yes_token)
                no_book = self.clob_client.get_order_book(market.no_token)
                
                market.best_bid_yes = float(yes_book.bids[0].price) if yes_book.bids else 0.0
                market.best_ask_yes = float(yes_book.asks[0].price) if yes_book.asks else 1.0
                market.best_bid_no = float(no_book.bids[0].price) if no_book.bids else 0.0
                market.best_ask_no = float(no_book.asks[0].price) if no_book.asks else 1.0
                
                print(f"[Maker] {label} Book: YES {market.best_bid_yes:.2f}/{market.best_ask_yes:.2f} | NO {market.best_bid_no:.2f}/{market.best_ask_no:.2f} (ImpProb: {market.implied_probability:.2f})")
            except Exception as e:
                print(f"[Maker] ⚠️ Error updating book for {label}: {e}")
                if "No orderbook exists" in str(e):
                    to_remove.append(label)
                self.event_logger.log({
                    "event_type": "book_error",
                    "label": label,
                    "reason": str(e),
                })

        if to_remove:
            for label in to_remove:
                market = self.markets.get(label)
                if market:
                    await self.cancel_all_orders([market.yes_token, market.no_token])
                self.markets.pop(label, None)
                print(f"[Maker] ❌ Dropping {label} - orderbook missing/expired")
                self.event_logger.log({
                    "event_type": "market_drop",
                    "label": label,
                    "reason": "orderbook_missing",
                })

    def calculate_my_bids(self, market: MarketData, asset_label: str) -> Tuple[Optional[float], Optional[float], QuoteContext]:
        """
        Calculate optimal bid prices.
        Strategy:
        1. Determine fair probability (P).
        2. Distribute MAKER_MAX_BUNDLE_COST according to P.
        3. Round down to tick size.
        4. Ensure we are not crossing the spread (taking liquidity) unless intended (here we are Makers).
        5. Optimize: Try to be Best Bid if it's "cheap enough", otherwise just sit at our max value.
        """
        ctx = QuoteContext()
        
        # External price feed (Binance). If unavailable, skip quoting.
        ref_price = self.binance_feed.get_price(asset_label) if self.binance_feed else None
        ctx.ref_price = ref_price
        if not ref_price:
            return 0.0, 0.0, ctx
        
        # Require strike; if missing, fall back to live price to keep the bot quoting in tests.
        strike = market.strike_price
        if not strike and ref_price:
            strike = ref_price
            ctx.adjustments["strike_source"] = "binance_fallback"
            market.strike_price = strike
            print(f"[Maker] ⚠️ {asset_label}: Strike missing; using live price {strike:.2f} as fallback")
        if not strike:
            return 0.0, 0.0, ctx
        
        t_years = get_time_to_expiry(market.end_date_iso)
        ctx.strike = strike
        ctx.t_years = t_years
        # Compute fair probability from external price
        fair_prob = calculate_fair_probability(ref_price, strike, t_years, volatility=MAKER_VOL)
        ctx.fair_prob = fair_prob
        
        # Add maker edge: quote inside fair by MIN_SPREAD
        bid_yes = max(0.0, round(fair_prob - MAKER_MIN_SPREAD, 2))
        bid_no = max(0.0, round((1.0 - fair_prob) - MAKER_MIN_SPREAD, 2))
        ctx.bid_yes_raw = bid_yes
        ctx.bid_no_raw = bid_no
        ctx.bundle_sum_raw = round(bid_yes + bid_no, 4)
        
        # Safety checks
        if bid_yes <= 0.01: bid_yes = 0.0
        if bid_no <= 0.01: bid_no = 0.0
        if bid_yes >= 0.99: bid_yes = 0.0 # Too risky
        if bid_no >= 0.99: bid_no = 0.0
        
        # Keep bids on the passive side and respect min spread
        if market.best_bid_yes and bid_yes > market.best_bid_yes:
            bid_yes = max(market.best_bid_yes, bid_yes - MAKER_TICK_SIZE)
        if market.best_bid_no and bid_no > market.best_bid_no:
            bid_no = max(market.best_bid_no, bid_no - MAKER_TICK_SIZE)
        
        # If spread is tight, step back a tick
        if (market.best_ask_yes - market.best_bid_yes) < MAKER_MIN_SPREAD:
            bid_yes = max(0.0, min(bid_yes, market.best_bid_yes - MAKER_TICK_SIZE))
        if (market.best_ask_no - market.best_bid_no) < MAKER_MIN_SPREAD:
            bid_no = max(0.0, min(bid_no, market.best_bid_no - MAKER_TICK_SIZE))
        
        # Final bundle safety: don't exceed max bundle with current best asks
        bundle_sum = bid_yes + bid_no
        if bundle_sum > MAKER_MAX_BUNDLE_COST:
            scale = MAKER_MAX_BUNDLE_COST / max(0.01, bundle_sum)
            bid_yes = round(bid_yes * scale, 2)
            bid_no = round(bid_no * scale, 2)
            ctx.adjustments["bundle_scale"] = round(scale, 4)
        
        ctx.bid_yes = bid_yes
        ctx.bid_no = bid_no
        ctx.bundle_sum_capped = round(bid_yes + bid_no, 4)
        
        return bid_yes, bid_no, ctx

    async def manage_orders(self):
        """Place or update orders (batched, with replace-on-delta and simple inventory skew)"""
        for label, market in self.markets.items():
            target_yes, target_no, quote_ctx = self.calculate_my_bids(market, label)

            # Simple inventory skew: if long YES, shade YES down and NO up; vice versa
            pre_inventory_yes = target_yes
            pre_inventory_no = target_no
            pos_yes = self.position.get(label, {}).get('YES', 0.0)
            pos_no = self.position.get(label, {}).get('NO', 0.0)
            imbalance = pos_yes - pos_no
            if imbalance > 0:  # long YES, so bid less aggressively on YES, more on NO
                target_yes = max(0.0, target_yes - MAKER_TICK_SIZE)
                target_no = round(min(0.99, target_no + MAKER_TICK_SIZE), 2)
            elif imbalance < 0:  # long NO
                target_no = max(0.0, target_no - MAKER_TICK_SIZE)
                target_yes = round(min(0.99, target_yes + MAKER_TICK_SIZE), 2)

            quote_ctx.adjustments["inventory_skew"] = round(imbalance, 4)
            quote_ctx.bid_yes = target_yes
            quote_ctx.bid_no = target_no
            quote_ctx.bundle_sum_capped = round(target_yes + target_no, 4)
            quote_ctx.adjustments["inventory_yes_delta"] = round(target_yes - pre_inventory_yes, 4)
            quote_ctx.adjustments["inventory_no_delta"] = round(target_no - pre_inventory_no, 4)

            # Log planned quotes and current book/position
            self.event_logger.log({
                "event_type": "quote_plan",
                "label": label,
                "market_slug": market.slug,
                "condition_id": market.condition_id,
                "bids": {
                    "yes": target_yes,
                    "no": target_no,
                    "size": MAKER_ORDER_SIZE,
                    "max_bundle": MAKER_MAX_BUNDLE_COST,
                },
                "book": {
                    "best_bid_yes": market.best_bid_yes,
                    "best_ask_yes": market.best_ask_yes,
                    "best_bid_no": market.best_bid_no,
                    "best_ask_no": market.best_ask_no,
                    "implied_prob": market.implied_probability,
                },
                "position": {
                    "yes": self.position.get(label, {}).get('YES', 0.0),
                    "no": self.position.get(label, {}).get('NO', 0.0),
                },
                "pricing": quote_ctx.to_dict(),
            })

            orders_to_place = []
            tokens_to_cancel: List[str] = []

            if target_yes > 0 and self._should_replace(market.yes_token, target_yes):
                tokens_to_cancel.append(market.yes_token)
                orders_to_place.append(OrderArgs(
                    price=target_yes,
                    size=MAKER_ORDER_SIZE,
                    side=BUY,
                    token_id=market.yes_token
                ))

            if target_no > 0 and self._should_replace(market.no_token, target_no):
                tokens_to_cancel.append(market.no_token)
                orders_to_place.append(OrderArgs(
                    price=target_no,
                    size=MAKER_ORDER_SIZE,
                    side=BUY,
                    token_id=market.no_token
                ))

            if tokens_to_cancel:
                await self.cancel_all_orders(tokens_to_cancel)

            if orders_to_place:
                fair = "n/a"
                if target_yes > 0 or target_no > 0:
                    fair = f"{target_yes+target_no:.2f}"
                print(f"[Maker] 💸 {label}: Bids YES@{target_yes:.2f}, NO@{target_no:.2f} (Sum: {fair})")
                try:
                    for o in orders_to_place:
                        if self.paper_trading:
                            oid = f"paper-{int(time.time()*1000)}"
                            self._record_quote(o.token_id, o.price, oid)
                            self.event_logger.log({
                                "event_type": "paper_order",
                                "label": label,
                                "token_id": o.token_id,
                                "price": o.price,
                                "size": o.size,
                                "side": "BUY",
                                "order_id": oid,
                            })
                            continue

                        order = self.clob_client.create_order(o)
                        resp = self.clob_client.post_order(order, orderType=OrderType.GTC)
                        oid = None
                        if isinstance(resp, dict):
                            oid = resp.get('orderID') or resp.get('id')
                        self._record_quote(o.token_id, o.price, oid)
                        self.event_logger.log({
                            "event_type": "order_post",
                            "label": label,
                            "token_id": o.token_id,
                            "price": o.price,
                            "size": o.size,
                            "side": "BUY",
                            "order_id": oid,
                            "status": resp.get('status') if isinstance(resp, dict) else None,
                        })
                except Exception as e:
                    print(f"[Maker] ⚠️ Order placement failed: {e}")
                    self.event_logger.log({
                        "event_type": "order_post_error",
                        "label": label,
                        "error": str(e),
                        "tokens": tokens_to_cancel,
                        "bids": {"yes": target_yes, "no": target_no},
                    })
            else:
                print(f"[Maker] 💤 {label}: No quote changes (holding book)")

    async def run(self):
        self.init_client()
        warm_connections()
        # Start Binance feed
        self.binance_feed = BinanceFeed()
        feed_task = asyncio.create_task(self.binance_feed.start())

        # Wait for feed readiness
        for _ in range(15):
            if self.binance_feed.get_price("BTC") and self.binance_feed.get_price("ETH"):
                break
            print("[Maker] Waiting for Binance feed...")
            await asyncio.sleep(1)

        if self.paper_trading:
            print("[Maker] PAPER MODE: reading live books/prices, not posting orders.")

        await self.discover_markets()
        
        if not self.markets:
            print("[Maker] ❌ No markets to trade. Exiting.")
            self.binance_feed.stop()
            feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feed_task
            return
        
        print(f"[Maker] 🤖 Starting Maker Loop (Max Bundle Cost: ${MAKER_MAX_BUNDLE_COST})")
        print(f"[Maker]    Order Size: {MAKER_ORDER_SIZE} shares")
        
        try:
            while self.running:
                # 2. Get latest prices
                await self.update_market_books()
                
                # 3. Place new orders
                await self.manage_orders()
                
                # 4. Sleep
                await asyncio.sleep(MAKER_REFRESH_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n[Maker] 🛑 Stopping...")
        finally:
            await self.cancel_all_orders(list(self.last_quotes.keys()))
            if self.binance_feed:
                self.binance_feed.stop()
            feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feed_task
            print("[Maker] 👋 Shutdown complete.")

if __name__ == "__main__":
    # Simple event loop
    strategy = MakerStrategy()
    asyncio.run(strategy.run())

