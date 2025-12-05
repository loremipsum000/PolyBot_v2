# Polymarket Bundle Arbitrage Bot

A bundle arbitrage trading bot for Polymarket 15-minute Bitcoin/Ethereum markets


## Strategy

**Bundle Arbitrage**: Buy BOTH YES and NO shares for less than $1.00 combined.

- **Guaranteed profit** = `(1.00 - bundle_cost) × min(yes_shares, no_shares)`
- **No directional risk** - market-neutral hedging
- **Auto-merge** positions to realize profits immediately

### Key Metrics (from analysis)
- Target bundle cost: < $0.99
- Average fills per transaction: ~17
- Position imbalance: < 10%
- Best arbitrage windows: 9-11 AM UTC

## Quick Start

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   - Copy `.env.example` to `.env` and fill in your keys
   - See [`docs/TUNING.md`](docs/TUNING.md) for configuration options

3. **Run the bot:**
   ```bash
   python main_hopper.py 10000  # Run for 10000 fills
   ```

## Architecture

```
main_hopper.py          # Entry point
├── market_hopper.py    # Multi-market coordination (BTC + ETH)
├── bundle_arbitrageur.py  # Core arbitrage strategy
├── trade_logger.py     # Trade logging & P&L tracking
├── utils.py            # Market discovery & HTTP pooling
└── config.py           # Configuration parameters
```

### Core Components

- **MarketHopper**: Monitors both BTC and ETH 15-minute markets, switches focus based on opportunity score
- **BundleArbitrageur**: Executes dual-side sweeps (buy YES + NO), maintains position balance
- **Connection Pooling**: HTTP session with keep-alive for ~40% faster API calls

## AWS Deployment

```bash
# Stop existing service
ssh -i poly-bot.pem ubuntu@YOUR_EC2_IP "sudo systemctl stop gabagool"

# Upload files
scp -i poly-bot.pem *.py ubuntu@YOUR_EC2_IP:~/gabagool-bot/

# Restart service
ssh -i poly-bot.pem ubuntu@YOUR_EC2_IP "sudo systemctl restart gabagool"

# Watch logs
ssh -i poly-bot.pem ubuntu@YOUR_EC2_IP "sudo journalctl -u gabagool -f"
```

## Testing

```bash
# Run bundle arbitrage unit tests
python test_bundle_arbitrage.py
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BA_MAX_BUNDLE_COST` | 0.99 | Max bundle cost to enter |
| `BA_MAX_IMBALANCE` | 0.10 | Max position imbalance (10%) |
| `BA_FILLS_PER_SWEEP` | 17 | Target fills per sweep |
| `BA_ORDER_SIZE` | 16 | Shares per order |

See [`docs/TUNING.md`](docs/TUNING.md) for all configuration options.

## License

MIT
