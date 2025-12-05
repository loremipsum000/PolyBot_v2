"""
Comprehensive Trade Logger for Polymarket Bot
Tracks all trades in a format compatible with copycat_script.py for analysis
"""

import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
import threading


@dataclass
class TradeRecord:
    """Single trade/order event - compatible with copycat_script.py"""
    timestamp: int                    # Unix timestamp
    event_type: str                   # 'order_placed', 'order_filled', 'order_canceled', 'merge'
    market_slug: str
    condition_id: str
    token_id: str
    outcome: str                      # 'Up' or 'Down' (YES/NO)
    side: str                         # 'BUY' or 'SELL'
    price: float                      # In decimal (0.54, not cents)
    size: float                       # Number of shares
    cost: float                       # price * size (USDC spent/received)
    order_id: Optional[str] = None
    status: Optional[str] = None      # 'live', 'matched', 'canceled', 'partial'
    fill_amount: Optional[float] = None
    remaining: Optional[float] = None
    
    # Market context at time of trade
    best_bid_yes: Optional[float] = None
    best_ask_yes: Optional[float] = None
    best_bid_no: Optional[float] = None
    best_ask_no: Optional[float] = None
    spread_yes: Optional[float] = None
    spread_no: Optional[float] = None
    bundle_cost: Optional[float] = None  # best_bid_yes + best_bid_no
    
    # Position tracking
    position_yes: Optional[float] = None
    position_no: Optional[float] = None
    net_exposure: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None


@dataclass 
class PositionSnapshot:
    """Current position state"""
    timestamp: int
    yes_shares: float
    no_shares: float
    yes_avg_cost: float
    no_avg_cost: float
    total_yes_cost: float
    total_no_cost: float
    net_exposure: float
    unrealized_pnl: float
    realized_pnl: float


class TradeLogger:
    """
    Comprehensive trade logger for analysis and strategy tuning.
    Outputs data compatible with copycat_script.py
    """
    
    def __init__(self, log_dir: str = "data/trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"
        
        # Position state
        self.position = {
            'yes_shares': 0.0,
            'no_shares': 0.0,
            'yes_cost': 0.0,
            'no_cost': 0.0,
            'realized_pnl': 0.0,
        }
        
        # Trade history for analysis
        self.trades: List[TradeRecord] = []
        self.orders: Dict[str, TradeRecord] = {}  # order_id -> record
        
        # Stats
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_canceled': 0,
            'orders_partial': 0,
            'orders_matched_immediate': 0,
            'total_volume_yes': 0.0,
            'total_volume_no': 0.0,
            'total_cost_yes': 0.0,
            'total_cost_no': 0.0,
            'avg_fill_price_yes': 0.0,
            'avg_fill_price_no': 0.0,
            'best_fill_price_yes': 1.0,
            'worst_fill_price_yes': 0.0,
            'best_fill_price_no': 1.0,
            'worst_fill_price_no': 0.0,
            'merges_count': 0,
            'merges_volume': 0.0,
        }
        
        self._lock = threading.Lock()
        
        print(f"[Logger] 📝 Session started: {self.session_id}")
        print(f"[Logger] 📂 Log file: {self.session_file}")

    def log_order_placed(
        self,
        market_slug: str,
        condition_id: str,
        token_id: str,
        outcome: str,  # 'Up' or 'Down'
        side: str,
        price: float,
        size: float,
        order_id: str,
        status: str,
        market_context: Optional[Dict] = None
    ) -> TradeRecord:
        """Log when an order is placed"""
        
        with self._lock:
            record = TradeRecord(
                timestamp=int(time.time()),
                event_type='order_placed',
                market_slug=market_slug,
                condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                side=side,
                price=price,
                size=size,
                cost=price * size,
                order_id=order_id,
                status=status,
                position_yes=self.position['yes_shares'],
                position_no=self.position['no_shares'],
                net_exposure=self.position['yes_cost'] + self.position['no_cost'],
                realized_pnl=self.position['realized_pnl'],
            )
            
            # Add market context
            if market_context:
                record.best_bid_yes = market_context.get('best_bid_yes')
                record.best_ask_yes = market_context.get('best_ask_yes')
                record.best_bid_no = market_context.get('best_bid_no')
                record.best_ask_no = market_context.get('best_ask_no')
                record.bundle_cost = market_context.get('bundle_cost')
                
                # Calculate spreads
                if record.best_bid_yes and record.best_ask_yes:
                    record.spread_yes = record.best_ask_yes - record.best_bid_yes
                if record.best_bid_no and record.best_ask_no:
                    record.spread_no = record.best_ask_no - record.best_bid_no
            
            self.orders[order_id] = record
            self.trades.append(record)
            self.stats['orders_placed'] += 1
            
            # Track immediate matches
            if status == 'matched':
                self.stats['orders_matched_immediate'] += 1
            
            self._write_record(record)
            self._print_trade(record)
            
            return record

    def log_order_filled(
        self,
        order_id: str,
        fill_price: float,
        fill_size: float,
        status: str = 'matched'
    ) -> Optional[TradeRecord]:
        """Log when an order is filled (fully or partially)"""
        
        with self._lock:
            if order_id not in self.orders:
                print(f"[Logger] ⚠️ Unknown order filled: {order_id[:16]}...")
                return None
            
            original = self.orders[order_id]
            
            record = TradeRecord(
                timestamp=int(time.time()),
                event_type='order_filled',
                market_slug=original.market_slug,
                condition_id=original.condition_id,
                token_id=original.token_id,
                outcome=original.outcome,
                side=original.side,
                price=fill_price,
                size=fill_size,
                cost=fill_price * fill_size,
                order_id=order_id,
                status=status,
                fill_amount=fill_size,
                remaining=original.size - fill_size if status == 'partial' else 0,
            )
            
            # Update position
            if original.side == 'BUY':
                if original.outcome == 'Up':
                    self.position['yes_shares'] += fill_size
                    self.position['yes_cost'] += fill_price * fill_size
                    self.stats['total_volume_yes'] += fill_size
                    self.stats['total_cost_yes'] += fill_price * fill_size
                    
                    # Track best/worst
                    self.stats['best_fill_price_yes'] = min(self.stats['best_fill_price_yes'], fill_price)
                    self.stats['worst_fill_price_yes'] = max(self.stats['worst_fill_price_yes'], fill_price)
                else:
                    self.position['no_shares'] += fill_size
                    self.position['no_cost'] += fill_price * fill_size
                    self.stats['total_volume_no'] += fill_size
                    self.stats['total_cost_no'] += fill_price * fill_size
                    
                    # Track best/worst
                    self.stats['best_fill_price_no'] = min(self.stats['best_fill_price_no'], fill_price)
                    self.stats['worst_fill_price_no'] = max(self.stats['worst_fill_price_no'], fill_price)
            
            record.position_yes = self.position['yes_shares']
            record.position_no = self.position['no_shares']
            record.net_exposure = self.position['yes_cost'] + self.position['no_cost']
            record.realized_pnl = self.position['realized_pnl']
            
            # Update stats
            self.stats['orders_filled'] += 1
            if status == 'partial':
                self.stats['orders_partial'] += 1
            
            self.trades.append(record)
            self._write_record(record)
            self._print_trade(record)
            
            return record

    def log_order_canceled(self, order_id: str) -> Optional[TradeRecord]:
        """Log when an order is canceled"""
        
        with self._lock:
            if order_id not in self.orders:
                return None
            
            original = self.orders[order_id]
            
            record = TradeRecord(
                timestamp=int(time.time()),
                event_type='order_canceled',
                market_slug=original.market_slug,
                condition_id=original.condition_id,
                token_id=original.token_id,
                outcome=original.outcome,
                side=original.side,
                price=original.price,
                size=original.size,
                cost=0,
                order_id=order_id,
                status='canceled',
                position_yes=self.position['yes_shares'],
                position_no=self.position['no_shares'],
            )
            
            self.stats['orders_canceled'] += 1
            self.trades.append(record)
            self._write_record(record)
            
            del self.orders[order_id]
            return record

    def log_merge(self, shares_merged: float, pnl: float):
        """Log when positions are merged for profit"""
        
        with self._lock:
            min_shares = min(self.position['yes_shares'], self.position['no_shares'])
            if shares_merged > min_shares:
                shares_merged = min_shares
            
            if shares_merged <= 0:
                return
            
            # Calculate average costs
            yes_avg = self.position['yes_cost'] / self.position['yes_shares'] if self.position['yes_shares'] > 0 else 0
            no_avg = self.position['no_cost'] / self.position['no_shares'] if self.position['no_shares'] > 0 else 0
            
            # Update positions
            self.position['yes_shares'] -= shares_merged
            self.position['no_shares'] -= shares_merged
            self.position['yes_cost'] -= yes_avg * shares_merged
            self.position['no_cost'] -= no_avg * shares_merged
            self.position['realized_pnl'] += pnl
            
            record = TradeRecord(
                timestamp=int(time.time()),
                event_type='merge',
                market_slug='',
                condition_id='',
                token_id='',
                outcome='MERGE',
                side='MERGE',
                price=1.0,
                size=shares_merged,
                cost=shares_merged,  # $1 per merged pair
                realized_pnl=self.position['realized_pnl'],
                position_yes=self.position['yes_shares'],
                position_no=self.position['no_shares'],
                net_exposure=self.position['yes_cost'] + self.position['no_cost'],
            )
            
            self.stats['merges_count'] += 1
            self.stats['merges_volume'] += shares_merged
            
            self.trades.append(record)
            self._write_record(record)
            
            print(f"[Logger] 🔀 Merged {shares_merged:.2f} pairs | PnL: ${pnl:.2f} | Total: ${self.position['realized_pnl']:.2f}")

    def update_position_from_fill(self, outcome: str, side: str, price: float, size: float):
        """
        Directly update position when we detect a fill (for immediate matches).
        Used when status='matched' on order placement.
        """
        with self._lock:
            if side == 'BUY':
                if outcome == 'Up':
                    self.position['yes_shares'] += size
                    self.position['yes_cost'] += price * size
                    self.stats['total_volume_yes'] += size
                    self.stats['total_cost_yes'] += price * size
                    self.stats['best_fill_price_yes'] = min(self.stats['best_fill_price_yes'], price)
                    self.stats['worst_fill_price_yes'] = max(self.stats['worst_fill_price_yes'], price)
                else:
                    self.position['no_shares'] += size
                    self.position['no_cost'] += price * size
                    self.stats['total_volume_no'] += size
                    self.stats['total_cost_no'] += price * size
                    self.stats['best_fill_price_no'] = min(self.stats['best_fill_price_no'], price)
                    self.stats['worst_fill_price_no'] = max(self.stats['worst_fill_price_no'], price)
                
                self.stats['orders_filled'] += 1

    def _write_record(self, record: TradeRecord):
        """Append record to JSONL file"""
        try:
            with open(self.session_file, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            print(f"[Logger] ⚠️ Write error: {e}")

    def _print_trade(self, record: TradeRecord):
        """Pretty print trade to console"""
        emoji = {
            'order_placed': '📤',
            'order_filled': '✅',
            'order_canceled': '❌',
            'merge': '🔀',
        }.get(record.event_type, '📝')
        
        outcome_str = 'YES' if record.outcome == 'Up' else ('NO' if record.outcome == 'Down' else record.outcome)
        status_str = f" [{record.status}]" if record.status else ""
        
        print(
            f"[Logger] {emoji} {record.event_type.upper()}{status_str} | "
            f"{record.side} {outcome_str} @ ${record.price:.3f} x{record.size:.1f} "
            f"(${record.cost:.2f}) | "
            f"Pos: Y={record.position_yes:.1f} N={record.position_no:.1f}"
        )

    def get_position_snapshot(self) -> PositionSnapshot:
        """Get current position state"""
        with self._lock:
            yes_avg = self.position['yes_cost'] / self.position['yes_shares'] if self.position['yes_shares'] > 0 else 0
            no_avg = self.position['no_cost'] / self.position['no_shares'] if self.position['no_shares'] > 0 else 0
            
            # Unrealized P&L: if we could merge all pairs at $1
            min_shares = min(self.position['yes_shares'], self.position['no_shares'])
            merge_value = min_shares * 1.0
            merge_cost = (yes_avg * min_shares) + (no_avg * min_shares)
            unrealized = merge_value - merge_cost
            
            return PositionSnapshot(
                timestamp=int(time.time()),
                yes_shares=self.position['yes_shares'],
                no_shares=self.position['no_shares'],
                yes_avg_cost=yes_avg,
                no_avg_cost=no_avg,
                total_yes_cost=self.position['yes_cost'],
                total_no_cost=self.position['no_cost'],
                net_exposure=self.position['yes_cost'] + self.position['no_cost'],
                unrealized_pnl=unrealized,
                realized_pnl=self.position['realized_pnl'],
            )

    def export_for_copycat(self, output_file: str = "trades.json") -> str:
        """Export trades in format compatible with copycat_script.py"""
        
        export_data = []
        for trade in self.trades:
            if trade.event_type in ('order_filled', 'order_placed') and trade.status == 'matched':
                export_data.append({
                    "timestamp": trade.timestamp,
                    "title": trade.market_slug,
                    "conditionId": trade.condition_id,
                    "side": trade.side,
                    "outcome": trade.outcome,
                    "price": trade.price,
                    "size": trade.size,
                })
        
        output_path = self.log_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[Logger] 📊 Exported {len(export_data)} trades to {output_path}")
        return str(output_path)

    def save_summary(self) -> Dict[str, Any]:
        """Save session summary"""
        
        with self._lock:
            # Calculate averages
            if self.stats['total_volume_yes'] > 0:
                self.stats['avg_fill_price_yes'] = self.stats['total_cost_yes'] / self.stats['total_volume_yes']
            if self.stats['total_volume_no'] > 0:
                self.stats['avg_fill_price_no'] = self.stats['total_cost_no'] / self.stats['total_volume_no']
            
            summary = {
                'session_id': self.session_id,
                'start_time': self.trades[0].timestamp if self.trades else None,
                'end_time': self.trades[-1].timestamp if self.trades else None,
                'duration_seconds': (self.trades[-1].timestamp - self.trades[0].timestamp) if len(self.trades) > 1 else 0,
                'stats': self.stats.copy(),
                'final_position': {
                    'yes_shares': self.position['yes_shares'],
                    'no_shares': self.position['no_shares'],
                    'yes_cost': self.position['yes_cost'],
                    'no_cost': self.position['no_cost'],
                    'net_exposure': self.position['yes_cost'] + self.position['no_cost'],
                    'realized_pnl': self.position['realized_pnl'],
                },
                'efficiency': {
                    'fill_rate': self.stats['orders_filled'] / self.stats['orders_placed'] if self.stats['orders_placed'] > 0 else 0,
                    'immediate_match_rate': self.stats['orders_matched_immediate'] / self.stats['orders_placed'] if self.stats['orders_placed'] > 0 else 0,
                    'cancel_rate': self.stats['orders_canceled'] / self.stats['orders_placed'] if self.stats['orders_placed'] > 0 else 0,
                }
            }
            
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[Logger] 📋 Summary saved to {self.summary_file}")
            return summary

    def print_session_stats(self):
        """Print current session statistics"""
        pos = self.get_position_snapshot()
        
        print("\n" + "="*65)
        print(f"📊 SESSION STATS - {self.session_id}")
        print("="*65)
        print(f"Orders Placed:        {self.stats['orders_placed']}")
        print(f"Orders Filled:        {self.stats['orders_filled']}")
        print(f"Immediate Matches:    {self.stats['orders_matched_immediate']}")
        print(f"Orders Canceled:      {self.stats['orders_canceled']}")
        print(f"Partial Fills:        {self.stats['orders_partial']}")
        print("-"*65)
        print(f"YES Volume:           {self.stats['total_volume_yes']:.2f} shares")
        print(f"YES Total Cost:       ${self.stats['total_cost_yes']:.2f}")
        print(f"YES Avg Fill Price:   ${self.stats['avg_fill_price_yes']:.4f}")
        if self.stats['total_volume_yes'] > 0:
            print(f"YES Best/Worst:       ${self.stats['best_fill_price_yes']:.3f} / ${self.stats['worst_fill_price_yes']:.3f}")
        print("-"*65)
        print(f"NO Volume:            {self.stats['total_volume_no']:.2f} shares")
        print(f"NO Total Cost:        ${self.stats['total_cost_no']:.2f}")
        print(f"NO Avg Fill Price:    ${self.stats['avg_fill_price_no']:.4f}")
        if self.stats['total_volume_no'] > 0:
            print(f"NO Best/Worst:        ${self.stats['best_fill_price_no']:.3f} / ${self.stats['worst_fill_price_no']:.3f}")
        print("-"*65)
        print(f"Merges:               {self.stats['merges_count']} ({self.stats['merges_volume']:.2f} shares)")
        print("-"*65)
        print(f"Current YES:          {pos.yes_shares:.2f} sh @ ${pos.yes_avg_cost:.4f} avg")
        print(f"Current NO:           {pos.no_shares:.2f} sh @ ${pos.no_avg_cost:.4f} avg")
        print(f"Bundle Cost (avg):    ${pos.yes_avg_cost + pos.no_avg_cost:.4f}")
        print(f"Net Exposure:         ${pos.net_exposure:.2f}")
        print(f"Unrealized P&L:       ${pos.unrealized_pnl:.2f}")
        print(f"Realized P&L:         ${pos.realized_pnl:.2f}")
        print("="*65 + "\n")

    def log_market_switch(self, old_market: str, new_market: str, old_score: float, new_score: float):
        """
        Log when the hopper switches focus from one market to another.
        This is key to matching gabagool22's market hopping behavior.
        """
        with self._lock:
            switch_record = {
                'timestamp': int(time.time()),
                'event_type': 'market_switch',
                'old_market': old_market,
                'new_market': new_market,
                'old_score': old_score,
                'new_score': new_score,
            }
            
            # Track switch stats
            if 'market_switches' not in self.stats:
                self.stats['market_switches'] = 0
                self.stats['switches_to_btc'] = 0
                self.stats['switches_to_eth'] = 0
            
            self.stats['market_switches'] += 1
            if new_market == 'BTC':
                self.stats['switches_to_btc'] += 1
            elif new_market == 'ETH':
                self.stats['switches_to_eth'] += 1
            
            self._write_record_dict(switch_record)
            print(f"[Logger] 🔄 SWITCH: {old_market} → {new_market} (score: {old_score:.2f} → {new_score:.2f})")
    
    def _write_record_dict(self, record: dict):
        """Write a dict record to JSONL file"""
        try:
            with open(self.session_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"[Logger] ⚠️ Write error: {e}")

    def get_comparison_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for comparison with target trader.
        Use this to tune knobs to match 0x6031B6eed1C97e853c6e0F03Ad3ce3529351F96d
        """
        pos = self.get_position_snapshot()
        
        return {
            # Price metrics (target: ~$0.54-0.55)
            'avg_fill_price_yes': self.stats['avg_fill_price_yes'],
            'avg_fill_price_no': self.stats['avg_fill_price_no'],
            'avg_bundle_cost': pos.yes_avg_cost + pos.no_avg_cost,
            
            # Volume metrics
            'total_volume_yes': self.stats['total_volume_yes'],
            'total_volume_no': self.stats['total_volume_no'],
            'volume_ratio': self.stats['total_volume_yes'] / self.stats['total_volume_no'] if self.stats['total_volume_no'] > 0 else 0,
            
            # Efficiency metrics
            'fill_rate': self.stats['orders_filled'] / self.stats['orders_placed'] if self.stats['orders_placed'] > 0 else 0,
            'immediate_match_rate': self.stats['orders_matched_immediate'] / self.stats['orders_placed'] if self.stats['orders_placed'] > 0 else 0,
            
            # Position balance (target trader stays balanced)
            'position_imbalance': abs(pos.yes_shares - pos.no_shares),
            'position_ratio': pos.yes_shares / pos.no_shares if pos.no_shares > 0 else 0,
            
            # P&L
            'realized_pnl': pos.realized_pnl,
            'unrealized_pnl': pos.unrealized_pnl,
            
            # Market hopping stats (if available)
            'market_switches': self.stats.get('market_switches', 0),
            'switches_to_btc': self.stats.get('switches_to_btc', 0),
            'switches_to_eth': self.stats.get('switches_to_eth', 0),
        }

