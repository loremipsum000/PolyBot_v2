import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests


def get_window_start_ts(end_date_iso: str, window_minutes: int = 15) -> Optional[int]:
    """
    Given an end date ISO string, compute the window start epoch (UTC).
    """
    if not end_date_iso:
        return None
    try:
        end = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        start = end - timedelta(minutes=window_minutes)
    except Exception:
        return None
    return int(start.timestamp())


def get_open_price_at_ts(symbol: str, start_ts: int) -> Optional[float]:
    """
    Fetch 1m klines from Binance Futures and return the candle open at/after start_ts.
    symbol: 'BTCUSDT' or 'ETHUSDT'
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 2,
        "startTime": start_ts * 1000,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        # Kline format: [open_time, open, high, low, close, volume, ...]
        open_price = float(data[0][1])
        return open_price
    except Exception:
        return None

