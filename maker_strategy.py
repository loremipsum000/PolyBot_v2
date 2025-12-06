import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from config import (
    PRIVATE_KEY, POLYGON_ADDRESS,
    BTC_SLUG_PREFIX, ETH_SLUG_PREFIX,
    WS_URL,
    CHAIN_ID,
    RPC_URL
)
from utils import get_target_markets, warm_connections, get_http_session
from binance_client import BinanceFeed
from binance_ohlc import get_open_price_at_ts
from pricing_engine import calculate_fair_probability, get_time_to_expiry

# Load env so MAKER_* are populated
load_dotenv()

# --- MAKER CONFIGURATION ---
MAKER_MAX_BUNDLE_COST = float(os.getenv("MAKER_MAX_BUNDLE_COST", "0.99"))  # Max combined cost we are willing to pay
MAKER_ORDER_SIZE = float(os.getenv("MAKER_ORDER_SIZE", "50.0"))          # Size of our limit orders
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

class MakerStrategy:
    def __init__(self):
        self.clob_client: Optional[ClobClient] = None
        self.markets: Dict[str, MarketData] = {}  # 'BTC', 'ETH' -> MarketData
        self.open_orders: Dict[str, List[str]] = {} # token_id -> list of order_ids
        self.last_quotes: Dict[str, float] = {}     # token_id -> last quoted price
        # Simple inventory tracking per asset
        self.position: Dict[str, Dict[str, float]] = {  # label -> {'YES': sh, 'NO': sh}
            'BTC': {'YES': 0.0, 'NO': 0.0},
            'ETH': {'YES': 0.0, 'NO': 0.0},
        }
        self.running = True
        self.binance_feed: Optional[BinanceFeed] = None

    def init_client(self):
        print("[Maker] 🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=2,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print(f"[Maker] ✅ Authenticated (Addr: {POLYGON_ADDRESS})")

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
                    
            except Exception as e:
                print(f"[Maker] ❌ Error fetching details for {label}: {e}")

    async def cancel_all_orders(self, token_ids: List[str]):
        """Cancel open orders for provided token_ids."""
        if not token_ids:
            return
        try:
            for tid in token_ids:
                self.clob_client.cancel_orders(token_id=tid)
                self.open_orders.pop(tid, None)
        except Exception as e:
            print(f"[Maker] ⚠️ Error cancelling orders: {e}")

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
        for label, market in self.markets.items():
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

    def calculate_my_bids(self, market: MarketData, asset_label: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate optimal bid prices.
        Strategy:
        1. Determine fair probability (P).
        2. Distribute MAKER_MAX_BUNDLE_COST according to P.
        3. Round down to tick size.
        4. Ensure we are not crossing the spread (taking liquidity) unless intended (here we are Makers).
        5. Optimize: Try to be Best Bid if it's "cheap enough", otherwise just sit at our max value.
        """
        
        # External price feed (Binance). If unavailable, skip quoting.
        ref_price = self.binance_feed.get_price(asset_label) if self.binance_feed else None
        if not ref_price:
            return 0.0, 0.0

        # Require strike; if missing, do not quote.
        if not market.strike_price:
            return 0.0, 0.0

        strike = market.strike_price
        t_years = get_time_to_expiry(market.end_date_iso)
        # Compute fair probability from external price
        fair_prob = calculate_fair_probability(ref_price, strike, t_years, volatility=MAKER_VOL)

        # Add maker edge: quote inside fair by MIN_SPREAD
        bid_yes = max(0.0, round(fair_prob - MAKER_MIN_SPREAD, 2))
        bid_no = max(0.0, round((1.0 - fair_prob) - MAKER_MIN_SPREAD, 2))
        
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
        if (bid_yes + bid_no) > MAKER_MAX_BUNDLE_COST:
            scale = MAKER_MAX_BUNDLE_COST / max(0.01, bid_yes + bid_no)
            bid_yes = round(bid_yes * scale, 2)
            bid_no = round(bid_no * scale, 2)
        
        return bid_yes, bid_no

    async def manage_orders(self):
        """Place or update orders (batched, with replace-on-delta and simple inventory skew)"""
        for label, market in self.markets.items():
            target_yes, target_no = self.calculate_my_bids(market, label)

            # Simple inventory skew: if long YES, shade YES down and NO up; vice versa
            pos_yes = self.position.get(label, {}).get('YES', 0.0)
            pos_no = self.position.get(label, {}).get('NO', 0.0)
            imbalance = pos_yes - pos_no
            if imbalance > 0:  # long YES, so bid less aggressively on YES, more on NO
                target_yes = max(0.0, target_yes - MAKER_TICK_SIZE)
                target_no = round(min(0.99, target_no + MAKER_TICK_SIZE), 2)
            elif imbalance < 0:  # long NO
                target_no = max(0.0, target_no - MAKER_TICK_SIZE)
                target_yes = round(min(0.99, target_yes + MAKER_TICK_SIZE), 2)

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
                    # Batch create and post
                    created = [self.clob_client.create_order(o) for o in orders_to_place]
                    resp = self.clob_client.post_orders(created, OrderType.GTC)
                    # Record quotes/order ids if returned
                    if isinstance(resp, list):
                        for r, o in zip(resp, orders_to_place):
                            oid = r.get('orderID') or r.get('id')
                            self._record_quote(o.token_id, o.price, oid)
                    else:
                        # fallback if single response
                        for o in orders_to_place:
                            self._record_quote(o.token_id, o.price)
                except Exception as e:
                    print(f"[Maker] ⚠️ Order placement failed: {e}")
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

