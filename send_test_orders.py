#!/usr/bin/env python3
"""
One-off utility to post a couple of live test BUY orders on the current 15m
BTC/ETH markets, with loosened prices so we can verify orders land.

Defaults:
- Discover current BTC/ETH 15m markets via utils.get_target_markets
- Fetch token IDs from /markets/{condition_id}
- Compute safe resting prices just below best ask (or fall back to 0.05/0.10)
- Post up to --per-side orders for YES and NO at the chosen price and size

Usage examples (activate venv first):
  python3 send_test_orders.py --asset BTC --size 5 --per-side 1
  python3 send_test_orders.py --asset ALL --size 5 --per-side 2 --yes-price 0.10 --no-price 0.10

Environment overrides:
  TEST_YES_PRICE, TEST_NO_PRICE  (float)
"""

import argparse
import sys
import time
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from py_clob_client.constants import POLYGON

from config import PRIVATE_KEY, POLYGON_ADDRESS, SIGNATURE_TYPE
from utils import get_target_markets
from pricing_engine import calculate_fair_probability, get_time_to_expiry
from binance_ohlc import get_open_price_at_ts, get_window_start_ts

REST = "https://clob.polymarket.com"


def fetch_tokens(condition_id: str):
    resp = requests.get(f"{REST}/markets/{condition_id}", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    tokens = {}
    for t in data.get("tokens", []):
        tid = t.get("token_id") or t.get("tokenId")
        outcome = (t.get("outcome") or t.get("label") or "").upper()
        if tid and outcome in ("UP", "YES"):
            tokens["YES"] = tid
        elif tid and outcome in ("DOWN", "NO"):
            tokens["NO"] = tid
    return tokens


def fetch_book_top(token_id: str):
    try:
        r = requests.get(f"{REST}/book", params={"token_id": token_id}, timeout=5)
        r.raise_for_status()
        data = r.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        def best(side, is_ask):
            bp = None
            for lvl in side:
                try:
                    p = float(lvl.get("price")) if isinstance(lvl, dict) else float(lvl[0])
                except Exception:
                    continue
                if bp is None:
                    bp = p
                else:
                    if is_ask and p < bp:
                        bp = p
                    if (not is_ask) and p > bp:
                        bp = p
            return bp
        return best(bids, False), best(asks, True)
    except Exception:
        return None, None


def compute_passive_price(best_bid, best_ask, fallback=0.10):
    """
    Choose a passive quote:
    - If ask exists, rest one tick below ask (but not below 0.01)
    - If bid exists, ensure we are at least at the bid
    - Round to cents
    - Fall back to provided default if no book data
    """
    price = None
    if best_ask is not None:
        price = max(0.01, best_ask - 0.01)
    if best_bid is not None:
        price = max(price or 0, best_bid)
    if price is None:
        price = fallback
    return round(price, 2)


def fetch_binance_price(symbol: str) -> float:
    """
    Lightweight spot price fetch (no websockets).
    """
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return 0.0


def compute_model_prices(
    asset_label: str,
    end_date_iso: str,
    best_bid_yes: float,
    best_ask_yes: float,
    best_bid_no: float,
    best_ask_no: float,
    max_bundle: float,
    edge: float,
    vol: float,
    tick: float = 0.01,
):
    """
    Mirror the maker-style quoting:
    - Black-Scholes cash-or-nothing probability
    - Subtract edge on both legs
    - Enforce max bundle cost
    - Keep passive vs best asks
    """
    symbol = "BTCUSDT" if asset_label == "BTC" else "ETHUSDT"
    live_px = fetch_binance_price(symbol)
    if live_px <= 0:
        return None, None, "no_live_price"

    start_ts = get_window_start_ts(end_date_iso) if end_date_iso else None
    strike = get_open_price_at_ts(symbol, start_ts) if start_ts else None
    if strike is None:
        strike = live_px

    t_years = get_time_to_expiry(end_date_iso) if end_date_iso else 0.0
    fair = calculate_fair_probability(live_px, strike, t_years, volatility=vol)

    bid_yes = max(0.0, round(fair - edge, 2))
    bid_no = max(0.0, round((1.0 - fair) - edge, 2))

    # Clip tiny/too-high values
    if bid_yes < 0.01:
        bid_yes = 0.0
    if bid_no < 0.01:
        bid_no = 0.0
    if bid_yes > 0.99:
        bid_yes = 0.0
    if bid_no > 0.99:
        bid_no = 0.0

    # Respect max bundle
    if (bid_yes + bid_no) > max_bundle and (bid_yes + bid_no) > 0:
        scale = max_bundle / (bid_yes + bid_no)
        bid_yes = round(bid_yes * scale, 2)
        bid_no = round(bid_no * scale, 2)

    # Stay passive vs asks
    if best_ask_yes and bid_yes >= best_ask_yes:
        bid_yes = round(max(0.01, best_ask_yes - tick), 2)
    if best_ask_no and bid_no >= best_ask_no:
        bid_no = round(max(0.01, best_ask_no - tick), 2)

    # If both zero, signal fallback
    if bid_yes == 0 and bid_no == 0:
        return None, None, "both_zero"

    return bid_yes or None, bid_no or None, f"fair={fair:.3f}, strike={strike:.2f}, live={live_px:.2f}"


def discover_market(asset_label: str):
    res = get_target_markets(
        slug_prefix="btc-updown-15m" if asset_label == "BTC" else "eth-updown-15m",
        asset_label=asset_label,
        keywords=[asset_label.lower()],
    )
    return res[0] if res else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["BTC", "ETH", "ALL"], default="ALL")
    ap.add_argument("--size", type=float, default=10.0, help="Order size (shares)")
    ap.add_argument("--per-side", type=int, default=1, help="How many orders per side to send (legacy, ignored when wait-for-fill is enabled)")
    ap.add_argument("--yes-price", type=float, default=None, help="Override YES price")
    ap.add_argument("--no-price", type=float, default=None, help="Override NO price")
    ap.add_argument("--max-bundle", type=float, default=0.99, help="Max bundle cost for model quoting")
    ap.add_argument("--edge", type=float, default=0.01, help="Edge to subtract from each side")
    ap.add_argument("--vol", type=float, default=0.6, help="Annualized vol for fair pricing")
    ap.add_argument("--wait-seconds", type=int, default=60, help="Wait up to N seconds for both legs to fill; stop emitting after fill or timeout")
    args = ap.parse_args()

    assets = ["BTC", "ETH"] if args.asset == "ALL" else [args.asset]

    client = ClobClient(
        REST.rstrip("/"),
        key=PRIVATE_KEY,
        chain_id=POLYGON,
        funder=POLYGON_ADDRESS,
        signature_type=SIGNATURE_TYPE,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    print("=== Account context ===")
    print(f"Address: {client.get_address()}")
    try:
        bal = client.get_balance_allowance()
        print(f"Balance/Allowance response: {bal}")
    except Exception as e:
        print(f"Balance/Allowance fetch failed: {e}")

    total_sent = 0
    for asset in assets:
        m = discover_market(asset)
        if not m:
            print(f"[{asset}] No market found")
            continue
        cond = m.get("condition_id")
        tokens = fetch_tokens(cond)
        yes_id = tokens.get("YES")
        no_id = tokens.get("NO")
        if not yes_id or not no_id:
            print(f"[{asset}] Missing tokens; skipping")
            continue

        # Books
        yes_bid, yes_ask = fetch_book_top(yes_id)
        no_bid, no_ask = fetch_book_top(no_id)

        yes_price = args.yes_price
        no_price = args.no_price

        if yes_price is None and no_price is None:
            model_yes, model_no, reason = compute_model_prices(
                asset,
                m.get("end_date_iso", ""),
                best_bid_yes=yes_bid or 0.0,
                best_ask_yes=yes_ask or 0.0,
                best_bid_no=no_bid or 0.0,
                best_ask_no=no_ask or 0.0,
                max_bundle=args.max_bundle,
                edge=args.edge,
                vol=args.vol,
            )
            if model_yes is not None:
                yes_price = model_yes
            if model_no is not None:
                no_price = model_no
            print(f"[{asset}] Model quote reason: {reason}")

        # Fallback to passive if still missing
        if yes_price is None:
            yes_price = compute_passive_price(yes_bid, yes_ask, fallback=0.10)
        if no_price is None:
            no_price = compute_passive_price(no_bid, no_ask, fallback=0.10)

        print(f"[{asset}] YES token {yes_id[:16]}... bid={yes_bid} ask={yes_ask} -> price {yes_price:.3f}")
        print(f"[{asset}] NO  token {no_id[:16]}... bid={no_bid} ask={no_ask} -> price {no_price:.3f}")

        # Single YES/NO pair, then wait for fills
        orders = [
            OrderArgs(price=yes_price, size=args.size, side=BUY, token_id=yes_id),
            OrderArgs(price=no_price, size=args.size, side=BUY, token_id=no_id),
        ]
        ids = []
        for oa in orders:
            order = client.create_order(oa)
            resp = client.post_order(order, orderType=OrderType.GTC)
            total_sent += 1
            status = resp.get("status") if isinstance(resp, dict) else None
            oid = resp.get("orderID") or resp.get("id") if isinstance(resp, dict) else None
            ids.append(oid)
            print(f"[{asset}] Posted {oa.side} {oa.size} @ {oa.price} token={oa.token_id[:10]}... status={status} id={oid}")
            time.sleep(0.2)

        # Wait for both legs to fill, then stop emitting orders
        filled_yes = False
        filled_no = False
        deadline = time.time() + args.wait_seconds
        while time.time() < deadline and (not (filled_yes and filled_no)):
            for i, oid in enumerate(ids):
                try:
                    oresp = client.get_order(oid)
                    ost = oresp.get("status") if isinstance(oresp, dict) else None
                    if ost and str(ost).lower() in ("filled", "done"):
                        if i == 0:
                            filled_yes = True
                        else:
                            filled_no = True
                except Exception:
                    pass
            if filled_yes and filled_no:
                print(f"[{asset}] ✅ Both legs filled; stopping.")
                break
            time.sleep(1.0)

    print(f"Done. Orders sent: {total_sent}")


if __name__ == "__main__":
    sys.exit(main())

