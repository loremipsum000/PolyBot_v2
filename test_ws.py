#!/usr/bin/env python3
"""
WebSocket Diagnostic Tool for Polymarket CLOB API.

Tests:
1. WebSocket connectivity to correct endpoint
2. Subscription message format (per docs)
3. Message parsing (asset_id, event_type, bids/asks)
4. PING keepalive mechanism
5. REST API orderbook comparison

Per Polymarket docs:
- https://docs.polymarket.com/quickstart/websocket/WSS-Quickstart
- https://docs.polymarket.com/developers/CLOB/websocket/market-channel
"""

import asyncio
import aiohttp
import json
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple

EASTERN = ZoneInfo("America/New_York")
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
REST_URL = "https://clob.polymarket.com"


def get_current_tokens() -> Tuple[List[str], Dict[str, str]]:
    """
    Discover current BTC/ETH 15m UP/DOWN markets and return their token IDs.
    Returns (token_ids, token_to_label mapping).
    """
    tokens: List[str] = []
    token_labels: Dict[str, str] = {}
    
    # Use utils.py get_target_markets for proper 15m market discovery
    print("[Discovery] Using utils.get_target_markets for 15m BTC/ETH markets...")
    
    try:
        # Import the utility function
        import sys
        sys.path.insert(0, '.')
        from utils import get_target_markets
        from config import BTC_SLUG_PREFIX, ETH_SLUG_PREFIX
        
        # Try BTC first
        print("[Discovery] Looking for BTC 15m market...")
        btc_markets = get_target_markets(slug_prefix=BTC_SLUG_PREFIX, asset_label='BTC')
        
        if btc_markets:
            m = btc_markets[0]
            cond_id = m.get('condition_id')
            print(f"[Discovery] Found BTC market: {m.get('question', '')[:50]}...")
            print(f"[Discovery]   Condition ID: {cond_id}")
            print(f"[Discovery]   Expires: {m.get('end_date_iso')}")
            
            # Fetch tokens from CLOB
            try:
                market_resp = requests.get(f"{REST_URL}/markets/{cond_id}", timeout=5)
                market_resp.raise_for_status()
                data = market_resp.json()
                
                accepting = data.get("accepting_orders", True)
                print(f"[Discovery]   accepting_orders={accepting}")
                
                if accepting:
                    for t in data.get("tokens", []):
                        tid = t.get("token_id") or t.get("tokenId")
                        outcome = t.get("outcome", "?")
                        if tid:
                            tokens.append(tid)
                            token_labels[tid] = f"BTC-{outcome}"
                            print(f"[Discovery]   Token: {tid[:24]}... ({outcome})")
            except Exception as e:
                print(f"[Discovery]   CLOB fetch failed: {e}")
        else:
            print("[Discovery] No BTC 15m market found")
        
        # Try ETH
        print("[Discovery] Looking for ETH 15m market...")
        eth_markets = get_target_markets(slug_prefix=ETH_SLUG_PREFIX, asset_label='ETH')
        
        if eth_markets:
            m = eth_markets[0]
            cond_id = m.get('condition_id')
            print(f"[Discovery] Found ETH market: {m.get('question', '')[:50]}...")
            print(f"[Discovery]   Condition ID: {cond_id}")
            print(f"[Discovery]   Expires: {m.get('end_date_iso')}")
            
            # Fetch tokens from CLOB
            try:
                market_resp = requests.get(f"{REST_URL}/markets/{cond_id}", timeout=5)
                market_resp.raise_for_status()
                data = market_resp.json()
                
                accepting = data.get("accepting_orders", True)
                print(f"[Discovery]   accepting_orders={accepting}")
                
                if accepting:
                    for t in data.get("tokens", []):
                        tid = t.get("token_id") or t.get("tokenId")
                        outcome = t.get("outcome", "?")
                        if tid:
                            tokens.append(tid)
                            token_labels[tid] = f"ETH-{outcome}"
                            print(f"[Discovery]   Token: {tid[:24]}... ({outcome})")
            except Exception as e:
                print(f"[Discovery]   CLOB fetch failed: {e}")
        else:
            print("[Discovery] No ETH 15m market found")
            
    except Exception as e:
        print(f"[Discovery] Error using get_target_markets: {e}")
        import traceback
        traceback.print_exc()
    
    # No fallback - we only want 15m BTC/ETH markets
    
    if not tokens:
        print("❌ Could not discover any active tokens!")
    
    return tokens, token_labels


def fetch_rest_orderbook(token_id: str) -> Optional[Dict]:
    """Fetch orderbook via REST API for comparison."""
    try:
        resp = requests.get(f"{REST_URL}/book", params={"token_id": token_id}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[REST] Orderbook fetch failed: {e}")
        return None


def extract_best(side: List, is_ask: bool) -> Tuple[Optional[float], Optional[float]]:
    """Extract best price and size from orderbook side."""
    if not side:
        return None, None
    
    best_price = None
    best_size = None
    
    for lvl in side:
        try:
            if isinstance(lvl, dict):
                p = float(lvl.get("price", 0))
                s = float(lvl.get("size", 0))
            else:
                p = float(lvl[0])
                s = float(lvl[1]) if len(lvl) > 1 else 0
        except Exception:
            continue
        
        if best_price is None:
            best_price, best_size = p, s
        else:
            if is_ask and p < best_price:
                best_price, best_size = p, s
            elif not is_ask and p > best_price:
                best_price, best_size = p, s
    
    return best_price, best_size


async def ws_ping_loop(ws):
    """Send PING every 10s per Polymarket docs."""
    try:
        while True:
            await ws.send_str("PING")
            await asyncio.sleep(10)
    except Exception:
        pass


async def test_ws():
    """Main WebSocket diagnostic test."""
    print("=" * 70)
    print("POLYMARKET WEBSOCKET DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"Time: {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"WS URL: {WS_URL}")
    print(f"REST URL: {REST_URL}")
    print("=" * 70)
    
    # Step 1: Discover tokens
    print("\n[Step 1] Discovering market tokens...")
    tokens, labels = get_current_tokens()
    print(f"Found {len(tokens)} tokens")
    
    if not tokens:
        print("❌ No tokens found. Cannot proceed.")
        return
    
    # Step 2: Fetch REST orderbooks for comparison
    print("\n[Step 2] Fetching REST orderbooks for comparison...")
    rest_books = {}
    for tid in tokens[:2]:  # Just first 2 for comparison
        book = fetch_rest_orderbook(tid)
        if book:
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid, bid_size = extract_best(bids, is_ask=False)
            best_ask, ask_size = extract_best(asks, is_ask=True)
            rest_books[tid] = {"bid": best_bid, "ask": best_ask}
            print(f"  {labels.get(tid, '?')}: bid=${best_bid} ask=${best_ask} (bids={len(bids)}, asks={len(asks)})")
            
            # Check for suspicious spread
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid
                if spread >= 0.90:
                    print(f"  ⚠️ WARNING: Spread ${spread:.2f} is suspiciously large (degenerate book?)")
    
    # Step 3: Connect to WebSocket
    print(f"\n[Step 3] Connecting to WebSocket...")
    print(f"URL: {WS_URL}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL, heartbeat=20, timeout=15) as ws:
                print("✅ Connected!")
                
                # Step 4: Subscribe with correct format
                print("\n[Step 4] Subscribing to market channel...")
                payload = {
                    "assets_ids": tokens,
                    "type": "market",
                }
                print(f"Subscription payload: {json.dumps(payload)[:100]}...")
                await ws.send_json(payload)
                print("✅ Subscription sent")
                
                # Start ping loop
                ping_task = asyncio.create_task(ws_ping_loop(ws))
                
                # Step 5: Listen for messages
                print("\n[Step 5] Listening for WebSocket messages (30 seconds)...")
                print("-" * 70)
                
                updates = 0
                raw_messages = []
                start_time = asyncio.get_event_loop().time()
                timeout = 30
                
                try:
                    while asyncio.get_event_loop().time() - start_time < timeout and updates < 20:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=5)
                        except asyncio.TimeoutError:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            print(f"[{elapsed:.1f}s] No message received (timeout)")
                            continue
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            # Log raw message for first few
                            if len(raw_messages) < 3:
                                raw_messages.append(msg.data[:500])
                            
                            # Handle non-JSON responses (like PONG)
                            raw = msg.data.strip()
                            if raw == "PONG" or raw == "":
                                continue  # Expected keepalive response
                            
                            try:
                                parsed = json.loads(raw)
                            except Exception as e:
                                print(f"[warn] JSON parse error: {e} | raw={raw[:50]}")
                                continue
                            
                            # Skip empty arrays
                            if parsed == []:
                                continue
                            
                            # Handle both single message (dict) and batch messages (list)
                            messages = parsed if isinstance(parsed, list) else [parsed]
                            
                            for data in messages:
                                if not isinstance(data, dict):
                                    print(f"[skip] Non-dict item: {type(data)}")
                                    continue
                                
                                event_type = data.get("event_type")
                                now = datetime.now(EASTERN).strftime("%H:%M:%S.%f")[:-3]
                                
                                # Skip non-relevant events
                                if event_type not in ("book", "price_change", None):
                                    print(f"[skip] event_type={event_type}")
                                    continue
                                
                                # ============================================================
                                # PRICE_CHANGE: Nested 'price_changes' array
                                # Structure: {"market": "0x...", "price_changes": [{"asset_id": "...", "price": "0.5", "side": "BUY"}]}
                                # ============================================================
                                if event_type == "price_change":
                                    price_changes = data.get("price_changes", [])
                                    if price_changes:
                                        for change in price_changes:
                                            asset_id = change.get("asset_id") or change.get("id")
                                            price = change.get("price")
                                            side = change.get("side", "").upper()
                                            size = change.get("size", "0")
                                            label = labels.get(asset_id, "?")
                                            side_str = "bid" if side == "BUY" else "ask"
                                            print(f"{now} | {label:8} | price_change | {side_str}=${price} size={size} | asset={asset_id[:20] if asset_id else 'N/A'}...")
                                            updates += 1
                                        continue
                                    # Fallback for flat price_change (single update)
                                    asset_id = data.get("asset_id") or data.get("market")
                                    print(f"[debug] price_change (flat): keys={list(data.keys())}")
                                    continue
                                
                                # ============================================================
                                # BOOK: Full snapshot with bids/asks arrays
                                # Structure: {"asset_id": "...", "bids": [...], "asks": [...]}
                                # ============================================================
                                asset_id = data.get("asset_id") or data.get("id") or data.get("token_id")
                                
                                print(f"[debug] event={event_type or 'N/A'} keys={list(data.keys())}")
                                
                                bids = data.get("bids", [])
                                asks = data.get("asks", [])
                                
                                best_bid, _ = extract_best(bids, is_ask=False)
                                best_ask, _ = extract_best(asks, is_ask=True)
                                
                                label = labels.get(asset_id, "?")
                                
                                print(f"{now} | {label:8} | event={event_type or 'N/A':12} | "
                                      f"asset_id={asset_id[:16] if asset_id else 'MISSING'}... | "
                                      f"bid=${best_bid or 'N/A'} ask=${best_ask or 'N/A'} "
                                      f"(bids={len(bids)}, asks={len(asks)})")
                                
                                updates += 1
                            
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            print(f"❌ Connection closed/error: {msg.type}")
                            break
                        else:
                            print(f"[other] msg.type={msg.type}")
                
                finally:
                    ping_task.cancel()
                    try:
                        await ping_task
                    except asyncio.CancelledError:
                        pass
                
                # Summary
                print("-" * 70)
                print(f"\n[Summary]")
                print(f"  Total updates received: {updates}")
                print(f"  Duration: {asyncio.get_event_loop().time() - start_time:.1f}s")
                
                if updates == 0:
                    print("\n❌ NO WEBSOCKET UPDATES RECEIVED!")
                    print("   Possible causes:")
                    print("   1. Wrong WebSocket URL")
                    print("   2. Wrong subscription payload format")
                    print("   3. Tokens not active/found")
                    print("   4. Server issue")
                else:
                    print(f"\n✅ WebSocket is receiving data ({updates} updates)")
                
                # Show raw messages
                if raw_messages:
                    print("\n[Raw Messages (first 3)]:")
                    for i, raw in enumerate(raw_messages):
                        print(f"  [{i+1}] {raw[:200]}...")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ws())
