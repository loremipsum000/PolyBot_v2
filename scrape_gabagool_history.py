#!/usr/bin/env python3
"""
Gabagool2 Historical Trade Scraper

Iterates through all BTC 15-minute markets from the past N days,
fetches Gabagool2's trades from each, and aggregates results into 
a single JSON file for comprehensive historical analysis.

Usage:
    python scrape_gabagool_history.py --days 30 --output data/gabagool_history_btc_30d.json
    python scrape_gabagool_history.py --days 1 --output data/gabagool_history_btc_1d.json  # Test run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
GABAGOOL_ADDRESS = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
TRADES_URL = "https://data-api.polymarket.com/trades"
MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Rate limiting delays (seconds)
DELAY_BETWEEN_MARKET_LOOKUPS = 0.3
DELAY_BETWEEN_TRADE_FETCHES = 0.5
PROGRESS_LOG_INTERVAL = 50  # Log every N markets

# Market slug patterns
BTC_SLUG_PREFIX = "btc-updown-15m"
ETH_SLUG_PREFIX = "eth-updown-15m"

# 15 minutes in seconds
MARKET_INTERVAL = 900


# ---------------------------------------------------
# TIMESTAMP GENERATION
# ---------------------------------------------------
def generate_market_timestamps(days: int = 30, asset: str = "BTC") -> List[Tuple[int, str]]:
    """
    Generate all market timestamps and slugs from N days ago to now.
    Returns list of (unix_timestamp, market_slug) tuples.
    """
    now = int(time.time())
    start = now - (days * 24 * 60 * 60)
    
    # Round to nearest 15-min boundary
    start = (start // MARKET_INTERVAL) * MARKET_INTERVAL
    
    prefix = BTC_SLUG_PREFIX if asset.upper() == "BTC" else ETH_SLUG_PREFIX
    
    markets = []
    ts = start
    while ts < now:
        slug = f"{prefix}-{ts}"
        markets.append((ts, slug))
        ts += MARKET_INTERVAL
    
    return markets


# ---------------------------------------------------
# MARKET DISCOVERY
# ---------------------------------------------------
def search_market_by_slug(slug: str) -> Optional[Dict]:
    """
    Search for a market by its slug using the gamma API.
    Returns market data dict if found, None otherwise.
    """
    try:
        # Try direct market lookup first
        resp = requests.get(f"{MARKETS_URL}", params={"slug": slug}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Response could be list or single object
        if isinstance(data, list) and data:
            return data[0]
        elif isinstance(data, dict) and data.get("conditionId"):
            return data
            
    except requests.RequestException:
        pass
    
    # Fallback: search API
    try:
        resp = requests.get(SEARCH_URL, params={"q": slug}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        events = data.get("events", []) if isinstance(data, dict) else []
        for event in events:
            markets = event.get("markets") or []
            for market in markets:
                if market.get("slug") == slug or slug in market.get("slug", ""):
                    return market
                    
    except requests.RequestException:
        pass
    
    return None


def get_condition_id(slug: str) -> Optional[str]:
    """
    Get condition_id for a market slug.
    Returns condition_id string if found, None otherwise.
    """
    market = search_market_by_slug(slug)
    if market:
        return market.get("conditionId") or market.get("condition_id")
    return None


# ---------------------------------------------------
# TRADE FETCHING
# ---------------------------------------------------
def fetch_trades(condition_id: str, user_address: str, page_limit: int = 500) -> List[Dict]:
    """
    Fetch all trades for a condition/user with simple pagination.
    Reused from analyze_gabagool.py with minor modifications.
    """
    all_trades = []
    offset = 0

    while True:
        params = {
            "limit": page_limit,
            "offset": offset,
            "takerOnly": "false",
            "market": condition_id,
            "user": user_address,
        }
        try:
            resp = requests.get(TRADES_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"  [WARN] Error fetching trades: {exc}")
            return all_trades

        if isinstance(data, dict):
            batch = data.get("trades", [])
        elif isinstance(data, list):
            batch = data
        else:
            batch = []

        all_trades.extend(batch)
        if len(batch) < page_limit:
            break
        offset += page_limit

    return all_trades


# ---------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------
def save_checkpoint(checkpoint_path: str, last_index: int, all_trades: List[Dict], 
                   markets_scanned: int, markets_with_trades: int):
    """Save progress checkpoint for resume capability."""
    checkpoint = {
        "last_index": last_index,
        "markets_scanned": markets_scanned,
        "markets_with_trades": markets_with_trades,
        "trades_count": len(all_trades),
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_partial_trades(output_path: str) -> List[Dict]:
    """Load existing trades from partial output file."""
    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r") as f:
                data = json.load(f)
                return data.get("trades", [])
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_partial_trades(output_path: str, trades: List[Dict], metadata: Dict):
    """Save partial progress to allow resume."""
    partial_path = output_path + ".partial"
    data = {
        "scrape_metadata": metadata,
        "trades": trades,
    }
    with open(partial_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------
# MAIN SCRAPER
# ---------------------------------------------------
def scrape_gabagool_history(
    days: int = 30,
    asset: str = "BTC",
    output_path: str = "data/gabagool_history_btc_30d.json",
    resume: bool = True,
):
    """
    Main scraper function that iterates through all markets and collects trades.
    """
    print("=" * 70)
    print("GABAGOOL2 HISTORICAL TRADE SCRAPER")
    print("=" * 70)
    print(f"Target Address: {GABAGOOL_ADDRESS}")
    print(f"Asset: {asset}")
    print(f"Days: {days}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Generate all market timestamps
    markets = generate_market_timestamps(days=days, asset=asset)
    total_markets = len(markets)
    print(f"\nGenerated {total_markets} market timestamps to scan")
    
    # Checkpoint paths
    checkpoint_path = output_path.replace(".json", "_checkpoint.json")
    
    # Check for resume
    start_index = 0
    all_trades = []
    markets_scanned = 0
    markets_with_trades = 0
    
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_index = checkpoint.get("last_index", 0) + 1
            markets_scanned = checkpoint.get("markets_scanned", 0)
            markets_with_trades = checkpoint.get("markets_with_trades", 0)
            all_trades = load_partial_trades(output_path)
            print(f"\n[RESUME] Continuing from index {start_index}")
            print(f"         Already scanned: {markets_scanned} markets")
            print(f"         Trades collected: {len(all_trades)}")
    
    # Calculate date range
    if markets:
        start_date = datetime.fromtimestamp(markets[0][0]).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(markets[-1][0]).strftime("%Y-%m-%d")
    else:
        start_date = end_date = "N/A"
    
    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Starting scan from index {start_index}...\n")
    
    start_time = time.time()
    
    try:
        for i, (ts, slug) in enumerate(markets[start_index:], start=start_index):
            # Progress logging
            if i % PROGRESS_LOG_INTERVAL == 0 or i == start_index:
                elapsed = time.time() - start_time
                rate = (i - start_index + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (total_markets - i) / rate if rate > 0 else 0
                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 0 else "N/A"
                
                print(f"[{i+1}/{total_markets}] Scanning {slug}... "
                      f"(rate: {rate:.1f}/s, ETA: {eta_str})")
            
            # Rate limit: delay before market lookup
            time.sleep(DELAY_BETWEEN_MARKET_LOOKUPS)
            
            # Get condition_id for this market
            condition_id = get_condition_id(slug)
            
            if not condition_id:
                markets_scanned += 1
                continue
            
            # Rate limit: delay before trade fetch
            time.sleep(DELAY_BETWEEN_TRADE_FETCHES)
            
            # Fetch trades for Gabagool2 in this market
            trades = fetch_trades(condition_id, GABAGOOL_ADDRESS)
            markets_scanned += 1
            
            if trades:
                markets_with_trades += 1
                # Add market metadata to each trade
                for trade in trades:
                    trade["_market_slug"] = slug
                    trade["_market_ts"] = ts
                    trade["_condition_id"] = condition_id
                
                all_trades.extend(trades)
                print(f"  → Found {len(trades)} trades in {slug}")
            
            # Save checkpoint every 100 markets
            if i % 100 == 0 and i > start_index:
                metadata = {
                    "gabagool_address": GABAGOOL_ADDRESS,
                    "asset": asset,
                    "date_range": f"{start_date} to {end_date}",
                    "markets_scanned": markets_scanned,
                    "markets_with_trades": markets_with_trades,
                    "total_trades": len(all_trades),
                    "status": "in_progress",
                }
                save_checkpoint(checkpoint_path, i, all_trades, markets_scanned, markets_with_trades)
                save_partial_trades(output_path, all_trades, metadata)
                
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving progress...")
        metadata = {
            "gabagool_address": GABAGOOL_ADDRESS,
            "asset": asset,
            "date_range": f"{start_date} to {end_date}",
            "markets_scanned": markets_scanned,
            "markets_with_trades": markets_with_trades,
            "total_trades": len(all_trades),
            "status": "interrupted",
        }
        save_checkpoint(checkpoint_path, i, all_trades, markets_scanned, markets_with_trades)
        save_partial_trades(output_path, all_trades, metadata)
        print(f"Progress saved. Run again with same arguments to resume.")
        sys.exit(1)
    
    # Final output
    elapsed_total = time.time() - start_time
    
    metadata = {
        "gabagool_address": GABAGOOL_ADDRESS,
        "asset": asset,
        "date_range": f"{start_date} to {end_date}",
        "markets_scanned": markets_scanned,
        "markets_with_trades": markets_with_trades,
        "total_trades": len(all_trades),
        "scrape_duration_seconds": round(elapsed_total, 2),
        "completed_at": datetime.now().isoformat(),
        "status": "complete",
    }
    
    output_data = {
        "scrape_metadata": metadata,
        "trades": all_trades,
    }
    
    # Save final output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Clean up checkpoint and partial files
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPE COMPLETE")
    print("=" * 70)
    print(f"Markets scanned: {markets_scanned}")
    print(f"Markets with trades: {markets_with_trades}")
    print(f"Total trades collected: {len(all_trades)}")
    print(f"Duration: {elapsed_total/60:.1f} minutes")
    print(f"Output saved to: {output_path}")
    print("=" * 70)
    
    return output_data


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape Gabagool2's historical trades from Polymarket BTC/ETH 15m markets"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30,
        help="Number of days to scrape (default: 30)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        choices=["BTC", "ETH"],
        help="Asset to scrape (default: BTC)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data/gabagool_history_{asset}_{days}d.json)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        args.output = f"data/gabagool_history_{args.asset.lower()}_{args.days}d.json"
    
    # Run scraper
    scrape_gabagool_history(
        days=args.days,
        asset=args.asset,
        output_path=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

