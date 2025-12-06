#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
WS_URL = "wss://ws-live-data.polymarket.com"

# Get fresh BTC tokens
condition_id = "0xfc3086a43d772f3e9a143e29fa9666ba70fcfc30686819c660b14da55c864e85"
resp = requests.get(f"https://clob.polymarket.com/markets/{condition_id}")
data = resp.json()
tokens = [t["token_id"] for t in data.get("tokens", [])]

print(f"Testing WebSocket at {datetime.now(EASTERN).strftime('%H:%M:%S ET')}")
print(f"Tokens: {len(tokens)}")
print(f"Token IDs: {[t[:20]+'...' for t in tokens]}")

async def test_ws():
    print(f"\nConnecting to {WS_URL}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL, heartbeat=20, timeout=10) as ws:
                print("Connected!")
                
                payload = {
                    "type": "subscribe",
                    "channel": "orderbook", 
                    "marketIds": tokens,
                }
                await ws.send_json(payload)
                print(f"Subscribed. Waiting for updates (10 sec max)...\n")
                
                count = 0
                start = asyncio.get_event_loop().time()
                
                async for msg in ws:
                    elapsed = asyncio.get_event_loop().time() - start
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                        except:
                            continue
                            
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        
                        if bids or asks:
                            now = datetime.now(EASTERN).strftime("%H:%M:%S.%f")[:-3]
                            top_bid = bids[0]["price"] if bids else "-"
                            top_ask = asks[0]["price"] if asks else "-"
                            print(f"{now} | bid={top_bid} ask={top_ask} (bids={len(bids)}, asks={len(asks)})")
                            count += 1
                        else:
                            # Print other message types
                            msg_type = data.get("type", "unknown")
                            print(f"[{elapsed:.1f}s] Message type: {msg_type}")
                            
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        print(f"Connection closed/error: {msg.type}")
                        break
                        
                    if elapsed > 10:
                        print("Timeout reached")
                        break
                        
                print(f"\nReceived {count} orderbook updates in {elapsed:.1f} seconds")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ws())
