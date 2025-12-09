from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from config import PHASE_CONFIG  # single source of truth


@dataclass
class PhaseSnapshot:
    name: str
    config: Dict
    minutes_remaining: float


class PhaseManager:
    """Determines trading phase based on time remaining to expiry."""

    def __init__(self, market_end_iso: str, binance_feed=None):
        try:
            self.end_dt = datetime.fromisoformat(market_end_iso.replace("Z", "+00:00"))
        except Exception:
            # Fallback for missing/invalid dates
            self.end_dt = datetime.now(timezone.utc)
        self.binance_feed = binance_feed

    def get_current_phase(self) -> PhaseSnapshot:
        now = datetime.now(timezone.utc)
        minutes_remaining = (self.end_dt - now).total_seconds() / 60.0

        if minutes_remaining > 10.0:
            name = "FORMATION"
        elif minutes_remaining > 3.0:
            name = "GRIND"
        else:
            name = "RESOLUTION"

        cfg = PHASE_CONFIG.get(name, PHASE_CONFIG["GRIND"])
        return PhaseSnapshot(name=name, config=cfg, minutes_remaining=minutes_remaining)

