#!/usr/bin/env python3
"""
Monitor Bundle Cost Opportunities

A lightweight script that:
- Connects to WS for both BTC and ETH markets
- Logs every bundle_cost_ask change to JSONL file
- Runs for multiple market cycles (configurable duration)
- Produces summary statistics at the end

No trading - pure monitoring for market analysis.
"""

import asyncio
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

# Add project to path
sys.path.insert(0, ".")

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


@dataclass
class MonitoredMarket:
    """State for a monitored market."""
    slug: str
    condition_id: str
    yes_token: str = ""
    no_token: str = ""
    best_ask_yes: float = 1.0
    best_ask_no: float = 1.0
    best_ask_yes_size: float = 0.0
    best_ask_no_size: float = 0.0
    best_bid_yes_size: float = 0.0
    best_bid_no_size: float = 0.0
    
    @property
    def bundle_cost(self) -> float:
        return self.best_ask_yes + self.best_ask_no
    
    @property
    def is_profitable(self) -> bool:
        """True arbitrage: buy both for < $1.00"""
        return self.bundle_cost < 1.00
    
    @property
    def is_within_threshold(self) -> bool:
        """Within configured threshold"""
        return self.bundle_cost <= BA_MAX_BUNDLE_COST
    
    @property
    def has_depth(self) -> bool:
        depth_yes = min(self.best_bid_yes_size, self.best_ask_yes_size)
        depth_no = min(self.best_bid_no_size, self.best_ask_no_size)
        return depth_yes >= BA_DEPTH_MIN and depth_no >= BA_DEPTH_MIN
    
    @property
    def prices_ok(self) -> bool:
        return self.best_ask_yes <= BA_MAX_YES_PRICE and self.best_ask_no <= BA_MAX_NO_PRICE


@dataclass
class MonitorStats:
    """Accumulated statistics for a market."""
    total_updates: int = 0
    profitable_count: int = 0  # bundle < 1.00
    threshold_count: int = 0   # bundle <= BA_MAX_BUNDLE_COST
    min_bundle: float = 2.0
    max_bundle: float = 0.0
    sum_bundle: float = 0.0
    profitable_opportunities: List[Dict] = field(default_factory=list)
    
    @property
    def avg_bundle(self) -> float:
        return self.sum_bundle / self.total_updates if self.total_updates > 0 else 0.0


class BundleCostMonitor:
    def __init__(self, log_dir: str = "data/bundle_monitor", duration_minutes: int = 60):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"monitor_session_{self.session_id}.jsonl"
        
        self.duration_minutes = duration_minutes
        self.start_time = time.time()
        self.end_time = self.start_time + (duration_minutes * 60)
        
        self.clob_client: Optional[ClobClient] = None
        self.markets: Dict[str, MonitoredMarket] = {}
        self.token_to_slug: Dict[str, str] = {}
        self.token_to_side: Dict[str, str] = {}  # token_id -> 'YES'/'NO'
        
        self.stats: Dict[str, MonitorStats] = {}
        self.running = True
        self.ws_connected = False
        
        self._last_log: Dict[str, float] = {}  # Rate limit logging
        
        print(f"📝 Logging to: {self.session_file}")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"📊 Threshold: bundle_cost <= {BA_MAX_BUNDLE_COST}")
        
        self._log_event({
            "event_type": "session_start",
            "session_id": self.session_id,
            "duration_minutes": duration_minutes,
            "threshold": BA_MAX_BUNDLE_COST,
        })
    
    def _log_event(self, record: Dict):
        """Write event to JSONL file."""
        record["ts"] = time.time()
        record["ts_iso"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.session_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"⚠️ Write error: {e}")
    
    async def init_client(self):
        """Initialize CLOB client."""
        if self.clob_client:
            return
        
        print("🔐 Initializing CLOB client...")
        self.clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=POLYGON_ADDRESS,
            signature_type=SIGNATURE_TYPE,
        )
        creds = self.clob_client.create_or_derive_api_creds()
        self.clob_client.set_api_creds(creds)
        print("✅ CLOB client authenticated")
    
    async def discover_markets(self):
        """Find available BTC/ETH markets."""
        print("🔍 Discovering markets...")
        
        for label, prefix in [("BTC", BTC_SLUG_PREFIX), ("ETH", ETH_SLUG_PREFIX)]:
            found = get_target_markets(slug_prefix=prefix, asset_label=label)
            if not found:
                print(f"⚠️ No {label} markets found")
                continue
            
            for m in found:
                c_id = m.get("condition_id")
                slug = m.get("market_slug")
                if not c_id or not slug:
                    continue
                
                try:
                    market_resp = self.clob_client.get_market(c_id)
                    accepting = market_resp.get("accepting_orders") if isinstance(market_resp, dict) else getattr(market_resp, "accepting_orders", None)
                    
                    if not accepting:
                        print(f"⚠️ {slug} not accepting orders")
                        continue
                    
                    tokens = market_resp.get("tokens", []) if isinstance(market_resp, dict) else market_resp.tokens
                    
                    yes_id, no_id = "", ""
                    for t in tokens:
                        outcome = (t.get("outcome") or "").upper()
                        if outcome in ["YES", "UP"]:
                            yes_id = t.get("token_id")
                        elif outcome in ["NO", "DOWN"]:
                            no_id = t.get("token_id")
                    
                    if yes_id and no_id:
                        market = MonitoredMarket(
                            slug=slug,
                            condition_id=c_id,
                            yes_token=yes_id,
                            no_token=no_id,
                        )
                        self.markets[slug] = market
                        self.token_to_slug[yes_id] = slug
                        self.token_to_slug[no_id] = slug
                        self.token_to_side[yes_id] = "YES"
                        self.token_to_side[no_id] = "NO"
                        self.stats[slug] = MonitorStats()
                        print(f"✅ Monitoring: {slug}")
                
                except Exception as e:
                    print(f"❌ Error with {slug}: {e}")
        
        if not self.markets:
            print("❌ No markets found to monitor!")
            self.running = False
    
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
    
    async def _handle_ws_message(self, data: Dict):
        """Process WS orderbook update."""
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
                    self._process_price_change(change)
                return
            # Fallback for flat format
            if data.get("asset_id") and data.get("price"):
                self._process_price_change(data)
            return
        
        # ============================================================
        # BOOK: Full snapshot with bids/asks arrays
        # ============================================================
        token_id = data.get("asset_id") or data.get("id") or data.get("token_id")
        slug = self.token_to_slug.get(token_id)
        if not slug:
            return
        
        market = self.markets.get(slug)
        if not market:
            return
        
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        
        bid_price, bid_size = self._extract_top(bids, 0.01, 0.0, is_ask=False)
        ask_price, ask_size = self._extract_top(asks, 0.99, 0.0, is_ask=True)
        
        side = self.token_to_side.get(token_id)
        
        # Update market state
        if side == "YES":
            market.best_ask_yes = ask_price
            market.best_ask_yes_size = ask_size
            market.best_bid_yes_size = bid_size
        else:
            market.best_ask_no = ask_price
            market.best_ask_no_size = ask_size
            market.best_bid_no_size = bid_size
        
        # Update stats
        stats = self.stats[slug]
        stats.total_updates += 1
        stats.sum_bundle += market.bundle_cost
        stats.min_bundle = min(stats.min_bundle, market.bundle_cost)
        stats.max_bundle = max(stats.max_bundle, market.bundle_cost)
        
        if market.is_profitable:
            stats.profitable_count += 1
            stats.profitable_opportunities.append({
                "ts": time.time(),
                "bundle_cost": market.bundle_cost,
                "ask_yes": market.best_ask_yes,
                "ask_no": market.best_ask_no,
            })
        
        if market.is_within_threshold:
            stats.threshold_count += 1
        
        self._update_stats_and_log(slug, market)
    
    def _process_price_change(self, change: Dict):
        """
        Process a single price_change item from the nested 'price_changes' array.
        Structure: {"asset_id": "...", "price": "0.42", "side": "BUY", "size": "100"}
        """
        token_id = change.get("asset_id") or change.get("id")
        if not token_id:
            return
        
        slug = self.token_to_slug.get(token_id)
        if not slug:
            return
        
        market = self.markets.get(slug)
        if not market:
            return
        
        try:
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "").upper()
        except (ValueError, TypeError):
            return
        
        if price <= 0:
            return
        
        token_side = self.token_to_side.get(token_id)
        
        # Update the appropriate price level based on side
        # BUY = bid, SELL = ask
        if token_side == "YES":
            if side == "BUY" and size > 0:
                market.best_bid_yes_size = size
            elif side == "SELL" and size > 0:
                market.best_ask_yes = price
                market.best_ask_yes_size = size
        else:  # NO
            if side == "BUY" and size > 0:
                market.best_bid_no_size = size
            elif side == "SELL" and size > 0:
                market.best_ask_no = price
                market.best_ask_no_size = size
        
        # Update stats
        stats = self.stats[slug]
        stats.total_updates += 1
        stats.sum_bundle += market.bundle_cost
        stats.min_bundle = min(stats.min_bundle, market.bundle_cost)
        stats.max_bundle = max(stats.max_bundle, market.bundle_cost)
        
        if market.is_profitable:
            stats.profitable_count += 1
            stats.profitable_opportunities.append({
                "ts": time.time(),
                "bundle_cost": market.bundle_cost,
                "ask_yes": market.best_ask_yes,
                "ask_no": market.best_ask_no,
            })
        
        if market.is_within_threshold:
            stats.threshold_count += 1
        
        self._update_stats_and_log(slug, market)
    
    def _update_stats_and_log(self, slug: str, market):
        """Rate-limited logging for bundle updates."""
        # Rate-limited logging (every 100ms for interesting prices)
        now = time.time()
        if market.bundle_cost <= 1.05 and (now - self._last_log.get(slug, 0) > 0.1):
            self._last_log[slug] = now
            self._log_event({
                "event_type": "bundle_update",
                "slug": slug,
                "bundle_cost": round(market.bundle_cost, 4),
                "ask_yes": round(market.best_ask_yes, 4),
                "ask_no": round(market.best_ask_no, 4),
                "is_profitable": market.is_profitable,
                "is_within_threshold": market.is_within_threshold,
                "has_depth": market.has_depth,
                "prices_ok": market.prices_ok,
            })
            
            # Alert on profitable opportunity
            if market.is_profitable:
                profit = 1.00 - market.bundle_cost
                print(f"🎯 PROFITABLE: {slug} bundle=${market.bundle_cost:.4f} profit=${profit:.4f}/pair")
    
    async def _ws_ping_loop(self, ws):
        """Send explicit PING messages every 10 seconds as per Polymarket docs."""
        try:
            while True:
                await ws.send_str("PING")
                await asyncio.sleep(10)
        except Exception:
            pass  # Connection closed
    
    async def _ws_listener(self):
        """WebSocket connection and message handling."""
        if not WS_URL:
            print("❌ WS_URL not configured")
            return
        
        while self.running and time.time() < self.end_time:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                        # Subscribe to all tokens
                        tokens = []
                        for market in self.markets.values():
                            tokens.extend([market.yes_token, market.no_token])
                        
                        # Per Polymarket docs: https://docs.polymarket.com/quickstart/websocket/WSS-Quickstart
                        payload = {
                            "assets_ids": tokens,
                            "type": "market",
                        }
                        await ws.send_json(payload)
                        print(f"🔗 WS subscribed to {len(tokens)} tokens (market channel)")
                        self.ws_connected = True
                        
                        # Start ping loop per Polymarket docs
                        ping_task = asyncio.create_task(self._ws_ping_loop(ws))
                        
                        try:
                            async for msg in ws:
                                if not self.running or time.time() >= self.end_time:
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
                print(f"⚠️ WS error: {e} (reconnecting...)")
                await asyncio.sleep(1)
            finally:
                self.ws_connected = False
    
    def print_summary(self):
        """Print and log final statistics."""
        elapsed = time.time() - self.start_time
        elapsed_mins = elapsed / 60
        
        print("\n" + "=" * 60)
        print("📊 BUNDLE COST MONITORING SUMMARY")
        print("=" * 60)
        print(f"Duration: {elapsed_mins:.1f} minutes")
        print(f"Threshold: bundle_cost <= {BA_MAX_BUNDLE_COST}")
        print()
        
        summary = {
            "event_type": "session_summary",
            "session_id": self.session_id,
            "duration_seconds": elapsed,
            "markets": {},
        }
        
        for slug, stats in self.stats.items():
            if stats.total_updates == 0:
                continue
            
            print(f"--- {slug} ---")
            print(f"  Total updates: {stats.total_updates}")
            print(f"  Bundle cost: min=${stats.min_bundle:.4f}, max=${stats.max_bundle:.4f}, avg=${stats.avg_bundle:.4f}")
            print(f"  Profitable (< $1.00): {stats.profitable_count} ({100*stats.profitable_count/stats.total_updates:.2f}%)")
            print(f"  Within threshold (<= ${BA_MAX_BUNDLE_COST}): {stats.threshold_count} ({100*stats.threshold_count/stats.total_updates:.2f}%)")
            
            if stats.profitable_opportunities:
                print(f"  📈 {len(stats.profitable_opportunities)} profitable opportunities logged")
            
            print()
            
            summary["markets"][slug] = {
                "total_updates": stats.total_updates,
                "min_bundle": stats.min_bundle,
                "max_bundle": stats.max_bundle,
                "avg_bundle": stats.avg_bundle,
                "profitable_count": stats.profitable_count,
                "threshold_count": stats.threshold_count,
                "profitable_pct": 100 * stats.profitable_count / stats.total_updates if stats.total_updates > 0 else 0,
            }
        
        self._log_event(summary)
        
        # Summary file
        summary_file = self.log_dir / f"summary_{self.session_id}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Summary saved to: {summary_file}")
    
    def stop(self):
        self.running = False
    
    async def run(self):
        await self.init_client()
        await self.discover_markets()
        
        if not self.markets:
            return
        
        print(f"\n🚀 Starting monitor (Ctrl+C to stop early)...")
        print(f"⏱️  Will run until: {datetime.fromtimestamp(self.end_time).strftime('%H:%M:%S')}")
        print()
        
        ws_task = asyncio.create_task(self._ws_listener())
        
        try:
            while self.running and time.time() < self.end_time:
                await asyncio.sleep(1)
                
                # Progress update every 60 seconds
                elapsed = time.time() - self.start_time
                if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                    remaining = (self.end_time - time.time()) / 60
                    total_updates = sum(s.total_updates for s in self.stats.values())
                    total_profitable = sum(s.profitable_count for s in self.stats.values())
                    print(f"⏱️  {int(elapsed/60)}m elapsed, {remaining:.0f}m remaining | Updates: {total_updates}, Profitable: {total_profitable}")
        
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
        
        self.print_summary()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor bundle cost opportunities")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes (default: 60)")
    args = parser.parse_args()
    
    monitor = BundleCostMonitor(duration_minutes=args.duration)
    
    def signal_handler(sig, frame):
        print("\n🛑 Stopping monitor...")
        monitor.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await monitor.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

